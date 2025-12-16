#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import bisect
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm
import rosbag
import sensor_msgs.point_cloud2 as pc2


# =========================
# CONFIGURATION
# =========================

# Root where the recorder wrote the bags via copy_overlapping_chunks
RAW_BASE_DIR = "/media/minghao/Data2TB/IndoorData/TwoNodeSapCopeExp"

# Root of the CoInfra dataset you want to build
COINFRA_ROOT = "/media/minghao/Data2TB/IndoorData/TwoNodeSapCopeExpCoInfra"

# Nodes to include in CoInfra slices
# INTERESTED_NODE_LIST = list(range(1, 15))
INTERESTED_NODE_LIST = [1,3]

# Topics (per node; same names on each node)
TOPICS = {
    "lidar": "/rslidar_points_front",
    "left_img": "/pylon_camera_node_left/image_rect/compressed",
    "right_img": "/pylon_camera_node_right/image_rect/compressed",
}

FUSION_RATE_HZ = 10.0           # 0.1s grid
MATCH_THRESHOLD_SEC = 0.05      # allowed offset when matching msgs to fusion ts
IMAGE_TIMESTAMP_SHIFT = -0.02   # 20ms shift for images to align with LiDAR

# Date directories to process; if None -> auto-detect all under RAW_BASE_DIR
DATE_DIR_WHITELIST = None  # e.g. ["2025_11_18_sunny"] to restrict
DATE_DIR_WHITELIST = ['NoBed', 'WithBed']


# =========================
# UTILITIES
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def decode_image(msg):
    """Decode sensor_msgs/CompressedImage -> BGR np.array."""
    np_arr = np.frombuffer(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_np


def get_bag_time_range(bag_path):
    with rosbag.Bag(bag_path, "r") as bag:
        return bag.get_start_time(), bag.get_end_time()


def generate_fusion_timestamps(global_start, global_end, rate_hz=10.0):
    """
    Generate a regularly spaced fusion grid from global_start to global_end (inclusive),
    snapped to 0.1s for 10Hz.
    """
    step = 1.0 / rate_hz
    start_ts = np.ceil(global_start * 10) / 10.0 + 0.0
    end_ts = np.floor(global_end * 10) / 10.0 - 0.0
    timestamps = np.arange(start_ts, end_ts + 1e-6, step)
    return np.round(timestamps, 3)


def find_common_time_range_for_slice(node_bag_map):
    """
    node_bag_map: {node_id: [bag_paths...]} for ONE span_dir.
    Returns (global_start, global_end) overlapping across nodes, trimmed by 1s on each side.
    """
    node_time_ranges = {}
    for node_id, bag_files in node_bag_map.items():
        start_times = []
        end_times = []
        for bag_file in bag_files:
            s, e = get_bag_time_range(bag_file)
            start_times.append(s)
            end_times.append(e)
        if not start_times:
            raise ValueError(f"No bags for node {node_id} in this slice.")
        node_time_ranges[node_id] = (min(start_times), max(end_times))

    all_start = [s for s, _ in node_time_ranges.values()]
    all_end = [e for _, e in node_time_ranges.values()]

    global_start = max(all_start) + 1.0
    global_end = min(all_end) - 1.0

    if global_start >= global_end:
        raise ValueError("No overlapping time range across nodes for this slice.")

    return global_start, global_end


def index_messages_for_node(node_id, bag_files, topics):
    """
    For a single node across multiple bag_files, build:
      cache[topic_key] = [(stamp, topic_name, bag_file), ...] sorted by stamp.
    """
    cache = defaultdict(list)
    desc = f"Indexing Node {node_id}"
    for bag_file in tqdm(bag_files, desc=desc, leave=False):
        with rosbag.Bag(bag_file, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=list(topics.values())):
                stamp = (msg.header if hasattr(msg, "header") else msg).stamp.to_sec()
                for key, tname in topics.items():
                    if topic == tname:
                        cache[key].append((stamp, topic, bag_file))

    for k in cache:
        cache[k].sort(key=lambda x: x[0])
    return cache


def match_timestamps(fusion_timestamps, all_cache, threshold=0.05):
    """
    all_cache[node_id][topic_key] = [(stamp, topic, bag_file), ...]
    Returns:
      matched[node_id][topic_key][fusion_ts] = (stamp, topic, bag_file)

    NOTE: for images, matching is done using (stamp + IMAGE_TIMESTAMP_SHIFT),
    but we still store the *raw* stamp in matched.
    """
    matched = defaultdict(lambda: defaultdict(dict))

    for node_id, topic_data in all_cache.items():
        for topic_key, messages in topic_data.items():
            if not messages:
                continue

            # raw stamps for all messages
            raw_stamps = [s for (s, _, _) in messages]

            # aligned stamps used for matching
            if topic_key in ("left_img", "right_img"):
                stamps = [s + IMAGE_TIMESTAMP_SHIFT for s in raw_stamps]
            else:
                stamps = raw_stamps

            for ts in fusion_timestamps:
                # use aligned 'stamps' for bisect and distance
                idx = bisect.bisect_left(stamps, ts)
                candidate_idxs = []
                if idx < len(stamps):
                    candidate_idxs.append(idx)
                if idx > 0:
                    candidate_idxs.append(idx - 1)

                if not candidate_idxs:
                    continue

                # pick closest in aligned time domain
                best_i = min(candidate_idxs, key=lambda i: abs(stamps[i] - ts))
                if abs(stamps[best_i] - ts) <= threshold:
                    # store the original (raw-stamp, topic, bag_file)
                    matched[node_id][topic_key][ts] = messages[best_i]

    return matched


def group_matched_by_bag(matched):
    """
    bag_group[bag_file][topic_name] = list of (node_id, topic_key, ts, msg_time)
    """
    bag_group = defaultdict(lambda: defaultdict(list))
    for node_id, topic_data in matched.items():
        for topic_key, ts_map in topic_data.items():
            for ts, (msg_time, topic, bag_file) in ts_map.items():
                bag_group[bag_file][topic].append((node_id, topic_key, ts, msg_time))
    return bag_group


# =========================
# FILLING HELPERS
# =========================

def get_timestamp_from_filename(path):
    """
    Extract the floating timestamp from filenames like:
      '4_1742850299.199.npy'
      '4_left_1742850299.202.jpg'
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    parts = stem.split("_")
    ts_str = parts[-1]
    try:
        return float(ts_str)
    except ValueError:
        return None


def move_closest_file_with_log(src_files, dst_folder, target_ts, logf, desc):
    """
    Copy the file whose encoded timestamp is closest to target_ts into dst_folder.
    Log the operation. Return True if something was copied, False otherwise.
    """
    if not src_files:
        return False

    candidates = []
    for f in src_files:
        ts = get_timestamp_from_filename(f)
        if ts is not None:
            candidates.append((f, ts))

    if not candidates:
        return False

    closest_file, src_ts = min(candidates, key=lambda x: abs(x[1] - target_ts))
    ensure_dir(dst_folder)
    dst_path = os.path.join(dst_folder, os.path.basename(closest_file))
    shutil.copy2(closest_file, dst_path)
    logf.write(f"FILL {desc}: target_ts={target_ts:.3f}, "
               f"src_ts={src_ts:.3f}, src={closest_file}, dst={dst_path}\n")
    return True


def fill_missing_data_for_slice(pcimage_root, fusion_timestamps, expected_nodes, step, log_path):
    """
    For each (ts, node, modality) that is missing, try to fill from ts-step or ts+step.
    Log all fills and remaining misses into log_path.
    """
    ts_set = set(round(ts, 1) for ts in fusion_timestamps)

    with open(log_path, "w") as logf:
        logf.write("# Neighbor-fill log\n")
        logf.write("# Format:\n")
        logf.write("# FILL <modality> node=<id>: target_ts=..., src_ts=..., src=..., dst=...\n")
        logf.write("# MISS <modality> node=<id>: target_ts=... (no suitable neighbor found)\n\n")

        for ts in fusion_timestamps:
            ts = round(ts, 1)
            ts_str = f"{ts:.1f}"
            ts_dir = os.path.join(pcimage_root, ts_str)
            lidar_dir = os.path.join(ts_dir, "lidar")
            cam_dir = os.path.join(ts_dir, "camera")

            for node_id in expected_nodes:
                # lidar
                lidar_pattern = f"{node_id}_*.npy"
                lidar_files = glob.glob(os.path.join(lidar_dir, lidar_pattern))

                # left/right cameras
                left_pattern = f"{node_id}_left_*.jpg"
                right_pattern = f"{node_id}_right_*.jpg"
                left_files = glob.glob(os.path.join(cam_dir, left_pattern))
                right_files = glob.glob(os.path.join(cam_dir, right_pattern))

                # --- lidar ---
                if len(lidar_files) == 0:
                    filled = False
                    for offset in [-step, step]:
                        neighbor_ts = round(ts + offset, 1)
                        if neighbor_ts not in ts_set:
                            continue
                        neighbor_str = f"{neighbor_ts:.1f}"
                        neighbor_lidar_dir = os.path.join(
                            pcimage_root, neighbor_str, "lidar"
                        )
                        neighbor_files = glob.glob(
                            os.path.join(neighbor_lidar_dir, lidar_pattern)
                        )
                        if move_closest_file_with_log(
                            neighbor_files,
                            lidar_dir,
                            ts,
                            logf,
                            desc=f"lidar node={node_id}"
                        ):
                            filled = True
                            break
                    if not filled:
                        logf.write(
                            f"MISS lidar node={node_id}: target_ts={ts:.3f} "
                            f"(no neighbor lidar found)\n"
                        )

                # --- left image ---
                if len(left_files) == 0:
                    filled = False
                    for offset in [-step, step]:
                        neighbor_ts = round(ts + offset, 1)
                        if neighbor_ts not in ts_set:
                            continue
                        neighbor_str = f"{neighbor_ts:.1f}"
                        neighbor_cam_dir = os.path.join(
                            pcimage_root, neighbor_str, "camera"
                        )
                        neighbor_files = glob.glob(
                            os.path.join(neighbor_cam_dir, left_pattern)
                        )
                        if move_closest_file_with_log(
                            neighbor_files,
                            cam_dir,
                            ts,
                            logf,
                            desc=f"left_img node={node_id}"
                        ):
                            filled = True
                            break
                    if not filled:
                        logf.write(
                            f"MISS left_img node={node_id}: target_ts={ts:.3f} "
                            f"(no neighbor left image found)\n"
                        )

                # --- right image ---
                if len(right_files) == 0:
                    filled = False
                    for offset in [-step, step]:
                        neighbor_ts = round(ts + offset, 1)
                        if neighbor_ts not in ts_set:
                            continue
                        neighbor_str = f"{neighbor_ts:.1f}"
                        neighbor_cam_dir = os.path.join(
                            pcimage_root, neighbor_str, "camera"
                        )
                        neighbor_files = glob.glob(
                            os.path.join(neighbor_cam_dir, right_pattern)
                        )
                        if move_closest_file_with_log(
                            neighbor_files,
                            cam_dir,
                            ts,
                            logf,
                            desc=f"right_img node={node_id}"
                        ):
                            filled = True
                            break
                    if not filled:
                        logf.write(
                            f"MISS right_img node={node_id}: target_ts={ts:.3f} "
                            f"(no neighbor right image found)\n"
                        )


# =========================
# EXTRACTION → COINFRA
# =========================

def extract_and_save_coinfra_slice(
    matched,
    slice_output_dir,
    fusion_timestamps,
    expected_nodes,
    topic_keys,
    step=0.1,
):
    """
    Save matched data into CoInfra layout for ONE slice.

    Layout (for this script, we ONLY handle PcImage):

      slice_output_dir/
        timestamps.txt              # full fusion grid (0.1s)
        PcImage/
          {fusion_ts}/              # e.g., 1742850299.2
            lidar/
              {node_id}_{exact_timestamp}.npy
            camera/
              {node_id}_{left|right}_{exact_timestamp}.jpg

    where:
      - fusion_ts is the regular grid timestamp (0.1s).
      - exact_timestamp is the *actual* (possibly shifted) sensor time, e.g. 1742850299.199
    """
    print("  Preparing fusion timestamps and PcImage structure...")
    regular_ts = [round(ts, 3) for ts in fusion_timestamps]

    ensure_dir(slice_output_dir)

    # Save timestamps.txt (one ts per line, fixed 0.1 format)
    ts_txt = os.path.join(slice_output_dir, "timestamps.txt")
    with open(ts_txt, "w") as f:
        for ts in regular_ts:
            f.write(f"{ts:.1f}\n")

    # Prepare PcImage/{ts}/lidar, PcImage/{ts}/camera
    pcimage_root = os.path.join(slice_output_dir, "PcImage")
    for ts in regular_ts:
        ts_str = f"{ts:.1f}"
        ts_dir = os.path.join(pcimage_root, ts_str)
        ensure_dir(os.path.join(ts_dir, "lidar"))
        ensure_dir(os.path.join(ts_dir, "camera"))

    # Restrict matched to regular_ts only (in practice they already match fusion_timestamps)
    reduced = defaultdict(lambda: defaultdict(dict))
    ts_set = set(regular_ts)
    for node_id, topic_data in matched.items():
        for topic_key, ts_map in topic_data.items():
            for ts, v in ts_map.items():
                if round(ts, 3) in ts_set:
                    reduced[node_id][topic_key][round(ts, 3)] = v

    # Group by bag for efficient reading
    bag_group = group_matched_by_bag(reduced)

    # To avoid accidentally writing multiple times for same (node, topic_key, ts),
    # we track which ones we already wrote.
    written_flags = defaultdict(lambda: defaultdict(set))
    # written_flags[node_id][topic_key] = set of fusion_ts written

    # Read bags and dump into npy/jpg
    print("  Reading bags and writing CoInfra PcImage files (direct matches)...")
    for bag_file, topic_map in tqdm(bag_group.items(), desc="  Bags", leave=False):
        with rosbag.Bag(bag_file, "r") as bag:
            # pre-filter by ts set
            topic_records = {
                topic: [
                    (node_id, topic_key, ts, round(msg_time, 6))
                    for (node_id, topic_key, ts, msg_time) in records
                ]
                for topic, records in topic_map.items()
            }

            topics_to_read = list(topic_records.keys())
            if not topics_to_read:
                continue

            for topic, msg, t in bag.read_messages(topics=topics_to_read):
                header = msg.header if hasattr(msg, "header") else msg
                msg_time = round(header.stamp.to_sec(), 6)  # raw sensor time

                if topic not in topic_records:
                    continue

                for (node_id, topic_key, ts, expected_time) in topic_records[topic]:
                    ts = round(ts, 3)
                    # If this (node, topic_key, ts) is already written, skip.
                    if ts in written_flags[node_id][topic_key]:
                        continue

                    # expected_time is the raw stamp from indexing
                    if abs(msg_time - expected_time) < 5e-3:
                        # fusion timestamp for folder name
                        ts_str = f"{ts:.1f}"
                        ts_dir = os.path.join(pcimage_root, ts_str)

                        if topic_key == "lidar":
                            # filename uses raw lidar time
                            exact_ts_str = f"{msg_time:.3f}"
                            points = np.array(
                                list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
                            )
                            lidar_dir = os.path.join(ts_dir, "lidar")
                            out_path = os.path.join(
                                lidar_dir, f"{node_id}_{exact_ts_str}.npy"
                            )
                            np.save(out_path, points)

                        elif topic_key == "left_img":
                            # filename uses shifted image time
                            shifted_ts = msg_time + IMAGE_TIMESTAMP_SHIFT
                            exact_ts_str = f"{shifted_ts:.3f}"
                            img = decode_image(msg)
                            cam_dir = os.path.join(ts_dir, "camera")
                            out_path = os.path.join(
                                cam_dir, f"{node_id}_left_{exact_ts_str}.jpg"
                            )
                            cv2.imwrite(out_path, img)

                        elif topic_key == "right_img":
                            # filename uses shifted image time
                            shifted_ts = msg_time + IMAGE_TIMESTAMP_SHIFT
                            exact_ts_str = f"{shifted_ts:.3f}"
                            img = decode_image(msg)
                            cam_dir = os.path.join(ts_dir, "camera")
                            out_path = os.path.join(
                                cam_dir, f"{node_id}_right_{exact_ts_str}.jpg"
                            )
                            cv2.imwrite(out_path, img)

                        # mark as written so we don't write duplicates for this fusion ts
                        written_flags[node_id][topic_key].add(ts)

    # Neighbor-based filling + logging
    log_path = os.path.join(slice_output_dir, "neighbor_fill_log.txt")
    print("  Filling missing data from neighboring timestamps...")
    fill_missing_data_for_slice(
        pcimage_root,
        regular_ts,
        expected_nodes=expected_nodes,
        step=step,
        log_path=log_path,
    )
    print(f"  Neighbor fill log saved to: {log_path}")


# =========================
# HIGH-LEVEL PIPELINE
# =========================

def discover_date_dirs(raw_base_dir):
    all_dirs = [
        d for d in os.listdir(raw_base_dir)
        if os.path.isdir(os.path.join(raw_base_dir, d))
    ]
    all_dirs.sort()
    if DATE_DIR_WHITELIST is not None:
        return [d for d in all_dirs if d in DATE_DIR_WHITELIST]
    return all_dirs


def discover_span_dirs_for_date(date_dir_path):
    """
    Under RAW_BASE_DIR/date_dir, span_dirs look like HH_MM_SS--HH_MM_SS.
    We treat every subdir as a span_dir and sort lexicographically.
    """
    subdirs = [
        d for d in os.listdir(date_dir_path)
        if os.path.isdir(os.path.join(date_dir_path, d))
    ]
    subdirs.sort()
    return subdirs


def build_node_bag_map_for_span(span_path, node_ids):
    """
    span_path: RAW_BASE_DIR/date_dir/span_dir
    Returns node_bag_map for this slice: {node_id: [bag_files...]}
    Skips slices if any required node has no bags.
    """
    node_bag_map = {}
    for node_id in node_ids:
        node_folder = os.path.join(span_path, f"Node{node_id}")
        bag_files = sorted(glob.glob(os.path.join(node_folder, "*.bag")))
        if not bag_files:
            print(f"    WARNING: no bags for Node{node_id} in {span_path}")
            return {}  # treat as unusable slice
        node_bag_map[node_id] = bag_files
    return node_bag_map


def process_one_slice(
    date_dir_name,
    span_dir_name,
    slice_idx,
    raw_base_dir,
    coinfra_root,
    node_ids,
    topics,
):
    """
    Convert one span_dir into one CoInfra slice_{idx}.
    """
    date_dir_path = os.path.join(raw_base_dir, date_dir_name)
    span_path = os.path.join(date_dir_path, span_dir_name)

    slice_name = f"slice{slice_idx}"
    slice_output_dir = os.path.join(coinfra_root, date_dir_name, slice_name)

    print(f"\nProcessing {date_dir_name} / {span_dir_name} -> {slice_name}")
    print(f"  Raw span path:    {span_path}")
    print(f"  Output slice dir: {slice_output_dir}")

    node_bag_map = build_node_bag_map_for_span(span_path, node_ids)
    if not node_bag_map:
        print("  Skipping slice due to missing node bags.")
        return

    # 1) find global overlapping time range for this slice
    global_start, global_end = find_common_time_range_for_slice(node_bag_map)
    print(f"  Overlapping range: {global_start:.3f} -> {global_end:.3f}")

    # 2) build fusion timestamps at 10Hz
    fusion_ts = generate_fusion_timestamps(
        global_start, global_end, rate_hz=FUSION_RATE_HZ
    )
    print(f"  Generated {len(fusion_ts)} candidate fusion timestamps.")

    # 3) index messages for all nodes in this slice
    print("  Indexing messages for nodes...")
    all_cache = {
        node_id: index_messages_for_node(node_id, bag_files, topics)
        for node_id, bag_files in node_bag_map.items()
    }

    # 4) match timestamps
    print("  Matching timestamps...")
    matched = match_timestamps(fusion_ts, all_cache, threshold=MATCH_THRESHOLD_SEC)

    # 5) extract and save into CoInfra layout (PcImage)
    extract_and_save_coinfra_slice(
        matched,
        slice_output_dir,
        fusion_ts,
        expected_nodes=node_ids,
        topic_keys=list(topics.keys()),
        step=1.0 / FUSION_RATE_HZ,
    )

    print(f"  Finished slice {slice_name}.")


def main():
    date_dirs = discover_date_dirs(RAW_BASE_DIR)
    if not date_dirs:
        print("No date directories found under RAW_BASE_DIR.")
        return

    for date_dir_name in date_dirs:
        date_dir_path = os.path.join(RAW_BASE_DIR, date_dir_name)
        span_dirs = discover_span_dirs_for_date(date_dir_path)
        if not span_dirs:
            print(f"\n[SKIP] {date_dir_name}: no span dirs found.")
            continue

        print(f"\n=== Date: {date_dir_name} ({len(span_dirs)} spans) ===")
        for idx, span_dir_name in enumerate(span_dirs):
            process_one_slice(
                date_dir_name=date_dir_name,
                span_dir_name=span_dir_name,
                slice_idx=idx,
                raw_base_dir=RAW_BASE_DIR,
                coinfra_root=COINFRA_ROOT,
                node_ids=INTERESTED_NODE_LIST,
                topics=TOPICS,
            )


if __name__ == "__main__":
    main()
