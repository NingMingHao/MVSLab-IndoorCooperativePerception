#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate CoInfra-style Calibration folders for each slice.

For each scenario/slice in DATASET_ROOT:

  slicex/
    Calibration/
      lidar/
        {node_id}.yaml
      camera/
        {node_id}_left.yaml
        {node_id}_right.yaml

Each YAML contains:

- lidar/{node_id}.yaml:
    lidar_to_ground: 4x4 (lidar -> ground_aligned)
    ground_to_global: 4x4 (ground_aligned -> global/local)

- camera/{node_id}_{left|right}.yaml:
    intrinsic_matrix: 3x3 (from first 3x3 of projection_matrix in ost.yaml)
    camera_to_lidar: 4x4 (camera -> lidar)
"""

import os
import sys
import math
import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, List
from numbers import Number
import numpy as np
import yaml

# =========================
# CONFIGURATION
# =========================

# Root of processed CoInfra dataset (with scenario/slicex/PcImage already created)
DATASET_ROOT = Path("/media/minghao/Data2TB/IndoorData/TwoNodeSapCopeExpCoInfra")

# Nodes included in this dataset (should match your PcImage generation)
NODE_IDS = [1,3]

# Optional: restrict to a subset of scenarios (folder names under DATASET_ROOT)
SCENARIO_WHITELIST = None
# Example:
# SCENARIO_WHITELIST = ["2025_02_12_heavysnow", "2025_03_24_rainy"]
# SCENARIO_WHITELIST = ["2025_11_18_sunny"]

# Root of per-node calibration/TF configs (OutdoorNode side)
# Expect structure:
#   NODE_CFG_ROOT / f"node{node_id}" / "Calibration/Left/ost.yaml"
#   NODE_CFG_ROOT / f"node{node_id}" / "Calibration/Right/ost.yaml"
#   NODE_CFG_ROOT / f"node{node_id}" / "tf_info.launch"
NODE_CFG_ROOT = Path(
    "/home/minghao/Documents/Gits/IndoorSensorNodes/MVSLab-IndoorCooperativePerception/labeling/transformation_files/cfg"
)

# MEC tf launch (contains mec->local and mec->ground_aligned_N)
MEC_TF_LAUNCH = Path(
    "/home/minghao/Documents/Gits/IndoorSensorNodes/MVSLab-IndoorCooperativePerception/labeling/transformation_files/mec_tf_info.launch"
)

# =========================
# YAML flow rows (pretty matrices)
# =========================

MATRIX_KEYS = {"camera_to_lidar", "intrinsic_matrix", "ground_to_global", "lidar_to_ground"}


class _FlowSeq(list):
    """Marker type to tell PyYAML to use flow style for this sequence."""
    pass


def _flow_seq_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", list(data), flow_style=True)


yaml.SafeDumper.add_representer(_FlowSeq, _flow_seq_representer)


def _builtin_number(x):
    # ensure numpy-like types become Python built-ins
    if isinstance(x, Number):
        try:
            return float(x)
        except Exception:
            return float(x)
    return x


def write_yaml(path: Path, data: dict):
    out = copy.deepcopy(data)
    # Force matrix-like keys' rows to flow style: [a, b, c, d]
    for k in list(out.keys()):
        if k in MATRIX_KEYS and isinstance(out[k], list):
            rows = out[k]
            if all(isinstance(r, (list, tuple)) for r in rows):
                flow_rows = []
                for r in rows:
                    flow_rows.append(_FlowSeq([_builtin_number(v) for v in r]))
                out[k] = flow_rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(
            out,
            f,
            sort_keys=False,
            default_flow_style=False,
            width=4096,
            allow_unicode=True,
        )


def read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =========================
# Math utils
# =========================

def Rx(r):
    cr, sr = math.cos(r), math.sin(r)
    return [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]


def Ry(p):
    cp, sp = math.cos(p), math.sin(p)
    return [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]


def Rz(y):
    cy, sy = math.cos(y), math.sin(y)
    return [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]


def matmul3(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def euler_zyx_to_R(yaw, pitch, roll):
    # ROS static_transform_publisher: yaw (Z), pitch (Y), roll (X)
    return matmul3(Rz(yaw), matmul3(Ry(pitch), Rx(roll)))


def homog4(R: List[List[float]], t: List[float]) -> List[List[float]]:
    return [
        [R[0][0], R[0][1], R[0][2], t[0]],
        [R[1][0], R[1][1], R[1][2], t[1]],
        [R[2][0], R[2][1], R[2][2], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def inv_homog4(T: List[List[float]]) -> List[List[float]]:
    R = [row[:3] for row in T[:3]]
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    t = [T[i][3] for i in range(3)]
    t_new = [-sum(Rt[i][k] * t[k] for k in range(3)) for i in range(3)]
    return homog4(Rt, t_new)


def matmul4(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(A[i][k] * B[k][j] for k in range(4))
    return out


# =========================
# TF parsing helpers
# =========================

def _remove_xml_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _parse_args_to_T(args_str: str):
    """
    args="x y z yaw pitch roll parent child period"
    Robust to spaces inside parent (e.g., 'mec_$(arg mec_id)').
    Returns T_parent_to_child, (parent, child)
    """
    parts = args_str.strip().split()
    if len(parts) < 9:
        raise ValueError(f"Unexpected args length in: {args_str}")

    x, y, z = map(float, parts[0:3])
    yaw, pitch, roll = map(float, parts[3:6])

    child = parts[-2]
    parent_tokens = parts[6:-2]
    parent = " ".join(parent_tokens).strip()

    R = euler_zyx_to_R(yaw, pitch, roll)
    T = np.linalg.inv(homog4(R, [x, y, z]))
    return T, (parent, child)


def parse_mec_tf(launch_path: Path):
    """
    From MEC tf launch:
      - T_mec_to_local
      - T_mec_to_ground_by_node[node_id]
    """
    raw = launch_path.read_text()
    xml = _remove_xml_comments(raw)
    root = ET.fromstring(xml)

    T_mec_to_local = None
    T_mec_to_ground_by_node: Dict[int, List[List[float]]] = {}

    for node in root.findall(".//node"):
        args = node.attrib.get("args", "")
        if not args:
            continue
        try:
            T, (parent, child) = _parse_args_to_T(args)
            T = np.linalg.inv(T)
        except Exception as e:
            print(f"[MEC tf] Skip node with unparsable args: {args} ({e})")
            continue

        name = node.attrib.get("name", "")

        # mec -> local
        if name == "mec_map_tf_broadcaster" or child.strip() == "local":
            T_mec_to_local = T

        # mec -> ground_aligned_N
        m_ground = re.search(r"ground_aligned_(\d+)", child)
        if m_ground:
            node_id = int(m_ground.group(1))
            T_mec_to_ground_by_node[node_id] = T

    if T_mec_to_local is None:
        raise RuntimeError("Failed to find mec->local TF in MEC launch file.")

    return T_mec_to_local, T_mec_to_ground_by_node


def parse_node_tf(node_tf_path: Path, node_id: int):
    """
    Parse per-node tf_info.launch and extract:
      - T_ground_to_lidar  : ground_aligned_{id} -> rslidar_front_{id}
      - T_lidar_to_cam_by_side["left"] / ["right"]
    
    This version is simple:
      - identify rows purely by node name
      - ignore parent/child frame names entirely
      - extract transforms only from:
           ground_aligned_tf_broadcaster
           lidar_cam_tf_broadcaster_left
           lidar_cam_tf_broadcaster_right
    """
    raw = node_tf_path.read_text()
    xml = _remove_xml_comments(raw)
    root = ET.fromstring(xml)

    T_ground_to_lidar = None
    T_lidar_to_cam_by_side = {}

    for node in root.findall(".//node"):
        name = node.attrib.get("name", "")
        args = node.attrib.get("args", "")
        if not args:
            continue

        # only these 3 names are needed
        if name not in [
            "ground_aligned_tf_broadcaster",
            "lidar_cam_tf_broadcaster_left",
            "lidar_cam_tf_broadcaster_right",
        ]:
            continue

        try:
            T, (parent, child) = _parse_args_to_T(args)
        except Exception as e:
            print(f"[parse_node_tf] Skip invalid args in {node_tf_path}: {args} ({e})")
            continue

        if name == "ground_aligned_tf_broadcaster":
            # This is: ground_aligned_{id} -> rslidar_front_{id}
            T_ground_to_lidar = T

        elif name == "lidar_cam_tf_broadcaster_left":
            # lidar -> left_cam
            T_lidar_to_cam_by_side["left"] = T

        elif name == "lidar_cam_tf_broadcaster_right":
            # lidar -> right_cam
            T_lidar_to_cam_by_side["right"] = T

    # validations
    if T_ground_to_lidar is None:
        raise RuntimeError(
            f"Missing ground_aligned_tf_broadcaster in {node_tf_path}"
        )

    if "left" not in T_lidar_to_cam_by_side or "right" not in T_lidar_to_cam_by_side:
        print(
            f"[WARNING] Node {node_id}: Missing left/right lidar→cam TFs in {node_tf_path}. "
            f"Found keys={list(T_lidar_to_cam_by_side.keys())}"
        )

    return T_ground_to_lidar, T_lidar_to_cam_by_side



# =========================
# Intrinsic loading
# =========================

def load_intrinsic_from_ost(ost_path: Path) -> List[List[float]]:
    """
    Read the 'projection_matrix' from ost.yaml and take the first 3x3 as intrinsic_matrix.
    """
    data = read_yaml(ost_path)
    proj = data.get("projection_matrix", {})
    rows = proj.get("rows", 0)
    cols = proj.get("cols", 0)
    flat = proj.get("data", [])

    if rows != 3 or cols != 4 or len(flat) != 12:
        raise RuntimeError(f"Unexpected projection_matrix shape in {ost_path}: rows={rows}, cols={cols}, len={len(flat)}")

    # Row-major 3x4
    P = [
        flat[0:4],
        flat[4:8],
        flat[8:12],
    ]
    # Take first 3x3
    K = [row[0:3] for row in P]
    return K


# =========================
# Dataset discovery
# =========================

def discover_scenarios(dataset_root: Path) -> List[Path]:
    scenarios = []
    for d in sorted(dataset_root.iterdir()):
        if not d.is_dir():
            continue
        if SCENARIO_WHITELIST is not None and d.name not in SCENARIO_WHITELIST:
            continue
        scenarios.append(d)
    return scenarios


def discover_slices_for_scenario(scen_path: Path) -> List[Path]:
    slices = []
    for sl in sorted(scen_path.iterdir()):
        if sl.is_dir() and sl.name.startswith("slice"):
            slices.append(sl)
    return slices


# =========================
# Calibration generation
# =========================

def build_global_calib():
    """
    Build calibration info that is independent of slice:
      - For each node_id: lidar_to_ground, ground_to_global, camera intrinsics, camera_to_lidar.
    """
    if not DATASET_ROOT.exists():
        print(f"[ERR] DATASET_ROOT not found: {DATASET_ROOT}")
        sys.exit(1)

    if not NODE_CFG_ROOT.exists():
        print(f"[ERR] NODE_CFG_ROOT not found: {NODE_CFG_ROOT}")
        sys.exit(1)

    if not MEC_TF_LAUNCH.exists():
        print(f"[ERR] MEC_TF_LAUNCH not found: {MEC_TF_LAUNCH}")
        sys.exit(1)

    # Parse MEC TFs
    T_mec_to_local, T_mec_to_ground_by_node = parse_mec_tf(MEC_TF_LAUNCH)

    # Precompute inverse once
    T_mec_to_local_inv = inv_homog4(T_mec_to_local)

    calib_by_node = {}

    for node_id in NODE_IDS:
        node_cfg = NODE_CFG_ROOT / f"node{node_id}"
        node_tf = node_cfg / "tf_info.launch"
        if not node_tf.exists():
            raise RuntimeError(f"Node TF file missing: {node_tf}")

        # Node-side TFs
        T_ground_to_lidar, T_lidar_to_cam_by_side = parse_node_tf(node_tf, node_id)

        # lidar_to_ground is inverse of ground -> lidar
        T_lidar_to_ground = inv_homog4(T_ground_to_lidar)

        if node_id not in T_mec_to_ground_by_node:
            raise RuntimeError(f"No MEC->ground_aligned_{node_id} TF found in {MEC_TF_LAUNCH}")
        T_mec_to_ground = T_mec_to_ground_by_node[node_id]

        # ground_aligned -> global(local):
        #   ground -> local = (mec -> local)^-1 * (mec -> ground)
        T_ground_to_global = matmul4(T_mec_to_local_inv, T_mec_to_ground)

        # Camera intrinsics
        left_ost = node_cfg / "Calibration" / "Left" / "ost.yaml"
        right_ost = node_cfg / "Calibration" / "Right" / "ost.yaml"
        if not left_ost.exists():
            raise RuntimeError(f"Left ost.yaml missing for node{node_id}: {left_ost}")
        if not right_ost.exists():
            raise RuntimeError(f"Right ost.yaml missing for node{node_id}: {right_ost}")

        K_left = load_intrinsic_from_ost(left_ost)
        K_right = load_intrinsic_from_ost(right_ost)

        # camera_to_lidar: inverse of lidar->cam
        cam_to_lidar_left = inv_homog4(T_lidar_to_cam_by_side["left"])
        cam_to_lidar_right = inv_homog4(T_lidar_to_cam_by_side["right"])

        calib_by_node[node_id] = {
            "lidar_to_ground": T_lidar_to_ground,
            "ground_to_global": T_ground_to_global,
            "intrinsic_left": K_left,
            "intrinsic_right": K_right,
            "cam_to_lidar_left": cam_to_lidar_left,
            "cam_to_lidar_right": cam_to_lidar_right,
        }

        print(f"[calib] Prepared calibration for node{node_id}")

    return calib_by_node


def write_slice_calibration(slice_dir: Path, calib_by_node: Dict[int, dict]):
    calib_root = slice_dir / "Calibration"
    lidar_dir = calib_root / "lidar"
    cam_dir = calib_root / "camera"

    lidar_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    for node_id, c in calib_by_node.items():
        # Lidar YAML
        # lidar_yaml = {
        #     "node_id": int(node_id),
        #     "frame_id": f"rslidar_front_{node_id}",
        #     "ground_frame": f"ground_aligned_{node_id}",
        #     "global_frame": "local",  # as defined by MEC tf
        #     "lidar_to_ground": c["lidar_to_ground"],
        #     "ground_to_global": c["ground_to_global"],
        # }
        lidar_yaml = {
            "ground_to_global": c["ground_to_global"],
            "lidar_to_ground": c["lidar_to_ground"],
        }
        lidar_path = lidar_dir / f"{node_id}.yaml"
        write_yaml(lidar_path, lidar_yaml)

        # Camera YAMLs
        # cam_left_yaml = {
        #     "node_id": int(node_id),
        #     "side": "left",
        #     "camera_frame": f"left_cam_{node_id}",
        #     "lidar_frame": f"rslidar_front_{node_id}",
        #     "intrinsic_matrix": c["intrinsic_left"],
        #     "camera_to_lidar": c["cam_to_lidar_left"],
        # }
        cam_left_yaml = {
            "camera_to_lidar": c["cam_to_lidar_left"],
            "intrinsic_matrix": c["intrinsic_left"],
        }
        cam_left_path = cam_dir / f"{node_id}_left.yaml"
        write_yaml(cam_left_path, cam_left_yaml)

        # cam_right_yaml = {
        #     "node_id": int(node_id),
        #     "side": "right",
        #     "camera_frame": f"right_cam_{node_id}",
        #     "lidar_frame": f"rslidar_front_{node_id}",
        #     "intrinsic_matrix": c["intrinsic_right"],
        #     "camera_to_lidar": c["cam_to_lidar_right"],
        # }
        cam_right_yaml = {
            "camera_to_lidar": c["cam_to_lidar_right"],
            "intrinsic_matrix": c["intrinsic_right"],
        }
        cam_right_path = cam_dir / f"{node_id}_right.yaml"
        write_yaml(cam_right_path, cam_right_yaml)

    print(f"[slice] Calibration written under: {calib_root}")


def main():
    calib_by_node = build_global_calib()

    scenarios = discover_scenarios(DATASET_ROOT)
    if not scenarios:
        print(f"[WARN] No scenarios found under {DATASET_ROOT}")
        return

    print(f"[scan] Found {len(scenarios)} scenario folders")

    for scen in scenarios:
        slices = discover_slices_for_scenario(scen)
        if not slices:
            print(f"[scan] Scenario {scen.name}: no slice_* folders, skipping")
            continue

        print(f"[scenario] {scen.name}: {len(slices)} slice(s)")
        for sl in slices:
            print(f"  -> Generating Calibration for {sl.name}")
            write_slice_calibration(sl, calib_by_node)

    print("[done] All slices processed.")


if __name__ == "__main__":
    main()
