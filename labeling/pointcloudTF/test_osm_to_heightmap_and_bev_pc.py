#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import math
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from matplotlib import cm
from geodesy import utm
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from sklearn.neighbors import KDTree

# Optional Open3D
try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

# ============ CONFIG (edit here or via CLI) ============
DEFAULT_OSM_PATH = "/home/minghao/Documents/Gits/IndoorSensorNodes/MVSLab-IndoorCooperativePerception/labeling/pointcloudTF/lanelet2_maps.osm"
DEFAULT_PCD_PATH = "/home/minghao/Documents/Gits/IndoorSensorNodes/MVSLab-IndoorCooperativePerception/labeling/pointcloudTF/map.pcd"
DEFAULT_OUTPUT_FOLDER = "/home/minghao/Documents/Gits/IndoorSensorNodes/MVSLab-IndoorCooperativePerception/labeling/pointcloudTF/osm_bev_height_from_pcd"

# image / geo
DEFAULT_IMG_SIZE = 1200          # pixels (square)
DEFAULT_SCALE = 50.0             # pixels per meter
HALF_EXPANSION_SIZE = 1.0        # meters (beyond lanelet extents) for "non-white" expansion

DEFAULT_REF_EASTING = 0.0
DEFAULT_REF_NORTHING = 0.0
DEFAULT_REF_ELEVATION = 0.0

# image anchor (metric shift from reference point)
DEFAULT_ANCHOR_EAST_SHIFT  = -6.0
DEFAULT_ANCHOR_NORTH_SHIFT =  20.0

# PCD (MGRS local) reference that matches the above UTM ref
# Because 36545 + 500000 = 536545 and 13925 + 4800000 = 4813925,
# mapping to the same local metric image frame is:
# x_metric = pc_easting - 36545 ; y_metric = pc_northing - 13925
PCD_REF_EASTING_MGRS  = 0.0
PCD_REF_NORTHING_MGRS = 0.0

# lanelet colors (BGR)
LANELET_COLORS = {
    "road"       : (60, 60, 60),
    "crosswalk"  : (140, 140, 140),
    "sidewalk"   : (80, 80, 80),
    "bicyclelane": (110, 110, 110),
    "unknown"    : (200, 200, 200),
}

# ========= Helpers =========
class Transformer:
    def __init__(self, ref_easting: float, ref_northing: float):
        self.ref_easting = ref_easting
        self.ref_northing = ref_northing
    def transform(self, lon: float, lat: float) -> Tuple[float, float]:
        p = utm.fromLatLong(lat, lon)
        return p.easting - self.ref_easting, p.northing - self.ref_northing

def polyfill(img_shape, pts):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.asarray(pts, dtype=np.int32)], 255)
    return mask.astype(bool)

def metric_grids(img_h: int, img_w: int, min_x: float, max_y: float, scale: float):
    px = (np.arange(img_w) + 0.5) / scale
    py = (np.arange(img_h) + 0.5) / scale
    X = min_x + px[None, :]
    Y = max_y - py[:, None]
    return np.broadcast_to(X, (img_h, img_w)).copy(), np.broadcast_to(Y, (img_h, img_w)).copy()

def parse_osm(osm_path: str):
    """
    Parse a Lanelet2-style OSM.

    For your new map, node positions are already given in a local metric frame
    via tags:
        <tag k="local_x" v="..."/>
        <tag k="local_y" v="..."/>

    We ignore lat/lon and directly use (local_x, local_y).
    """
    tree = ET.parse(osm_path)
    root = tree.getroot()

    # ---- Nodes: use local_x / local_y (meters in local frame) ----
    raw_nodes = {}
    for n in root.findall("node"):
        nid = n.attrib["id"]
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in n.findall("tag")}

        if "local_x" in tags and "local_y" in tags:
            x = float(tags["local_x"])
            y = float(tags["local_y"])
            raw_nodes[nid] = (x, y)
        else:
            # If you still want to support old lat/lon lanelet2 maps, you can
            # optionally fall back here. For your new indoor map it's not needed.
            continue

    # ---- Ways: same as before ----
    ways = {}
    for w in root.findall("way"):
        wid = int(w.attrib["id"])
        nds = [nd.attrib["ref"] for nd in w.findall("nd") if nd.attrib["ref"] in raw_nodes]
        if len(nds) >= 2:
            ways[wid] = nds

    # ---- Lanelets: same as before ----
    lanelets = []
    for rel in root.findall("relation"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in rel.findall("tag")}
        if tags.get("type") == "lanelet":
            members = {m.attrib["role"]: int(m.attrib["ref"]) for m in rel.findall("member") if m.attrib["type"] == "way"}
            subtype = tags.get("subtype", "unknown")
            if tags.get("landuse") == "walkway":
                subtype = "sidewalk"
            if tags.get("landuse") == "bicycle_lane":
                subtype = "bicyclelane"
            lanelets.append({
                "id": int(rel.attrib["id"]),
                "left": members.get("left"),
                "right": members.get("right"),
                "subtype": subtype,
                "one_way": tags.get("one_way", "unknown"),
                "speed_limit": tags.get("speed_limit", "unknown")
            })
    return raw_nodes, ways, lanelets


def draw_vertical_colorbar(
    img: np.ndarray,
    vmin: float,
    vmax: float,
    cmap=cm.get_cmap("rainbow"),
    bar_h: int = 200,
    bar_w: int = 20,
    margin: int = 20,
    label_pad: int = 6,
    tick_count: int = 5,
    decimals: int = 1,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.45,
    thickness: int = 1
) -> None:
    h, w = img.shape[:2]
    label_space = 55
    x1 = w - margin
    x0 = x1 - bar_w
    x_bg0 = x0 - label_space - label_pad
    y0 = margin
    y1 = y0 + bar_h
    x_bg0 = max(0, x_bg0); y0 = max(0, y0); x1 = min(w - 1, x1); y1 = min(h - 1, y1)
    x0 = x1 - bar_w
    cv2.rectangle(img, (x_bg0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(img, (x_bg0, y0), (x1, y1), (0, 0, 0), 1)
    for i in range(bar_h):
        frac = 1.0 - (i / (bar_h - 1))
        color = (np.array(cmap(frac)[:3]) * 255.0).astype(np.uint8)
        y = y0 + i
        cv2.line(img, (x0, y), (x1, y), color.tolist(), 1)
    ticks = np.linspace(0.0, 1.0, tick_count)
    for t in ticks:
        y = int(round(y1 - t * (bar_h - 1)))
        cv2.line(img, (x0 - 6, y), (x0 - 1, y), (0, 0, 0), 1)
        val = vmin + t * (vmax - vmin)
        txt = f"{val:.{decimals}f} m"
        (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
        tx = x0 - 8 - tw
        ty = max(y + th // 2, y0 + th)
        cv2.putText(img, txt, (tx, ty), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

# -------- PCD loading & frame conversion --------
def load_pcd_xyz(pcd_path: str) -> np.ndarray:
    """
    Returns Nx3 array of [easting_mgrs, northing_mgrs, z] from PCD.
    Uses Open3D if available; otherwise parses ASCII PCD minimally.
    """
    if HAS_O3D:
        pc = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pc.points, dtype=np.float32)
        return pts  # assume pcd x=easting_mgrs, y=northing_mgrs, z=height
    # Fallback: minimal ASCII PCD reader (XYZ only)
    with open(pcd_path, "r") as f:
        header = True
        points = []
        for line in f:
            if header:
                if line.strip().lower().startswith("data"):
                    header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append((x, y, z))
                except Exception:
                    continue
    return np.asarray(points, dtype=np.float32)

def pcd_to_metric_xyzh(pts_mgrs_xyz: np.ndarray,
                       mgrs_ref_e: float,
                       mgrs_ref_n: float,
                       ref_elevation: float) -> np.ndarray:
    """
    Convert PCD MGRS-local (e,n,z) to the same local metric frame as the OSM image.
    Because DEFAULT_REF_E/N equals (mgrs_ref + 100km origin), this simplifies to:
      x_metric = e - mgrs_ref_e
      y_metric = n - mgrs_ref_n
      z_metric = z - ref_elevation
    Returns Nx3 [x_m, y_m, z_rel]
    """
    if pts_mgrs_xyz.size == 0:
        return pts_mgrs_xyz
    xm = pts_mgrs_xyz[:, 0] - mgrs_ref_e
    ym = pts_mgrs_xyz[:, 1] - mgrs_ref_n
    zm = pts_mgrs_xyz[:, 2] - ref_elevation
    return np.stack([xm, ym, zm], axis=1).astype(np.float32)

# -------- Height from PCD (robust local ground) --------
def height_from_pcd_for_pixels(xP: np.ndarray, yP: np.ndarray,
                               pc_xyzh: np.ndarray,
                               radius_m: float = 1.5,
                               k: int = 48,
                               lowest_frac: float = 0.10,
                               chunk_size: int = 200_000) -> np.ndarray:
    """
    For each (xP,yP) pixel center:
      - gather up to k nearest PCD neighbors
      - keep only those within radius_m
      - take mean of the lowest (10%) z values among those
      - if none within radius, fallback to nearest neighbor
    Vectorized in batches with a KD-tree.
    """
    assert pc_xyzh.shape[1] == 3
    if pc_xyzh.shape[0] == 0:
        return np.full(xP.shape[0], np.nan, dtype=np.float32)

    tree = KDTree(pc_xyzh[:, :2])  # on XY (metric)
    k_eff = min(k, pc_xyzh.shape[0])
    z_out = np.empty(xP.shape[0], dtype=np.float32)

    # number to keep from the lowest z within the radius
    keep = max(1, int(math.ceil(lowest_frac * k_eff)))

    for i0 in range(0, xP.shape[0], chunk_size):
        i1 = min(xP.shape[0], i0 + chunk_size)
        pts = np.column_stack((xP[i0:i1], yP[i0:i1]))
        dist, ind = tree.query(pts, k=k_eff, return_distance=True)  # (B,k)
        z_neighbors = pc_xyzh[ind, 2]                               # (B,k)
        # mask by radius
        within = dist <= float(radius_m)

        # If a row has no neighbors within radius, use nearest neighbor
        has_any = within.any(axis=1)
        # Set z to +inf when outside radius (so they won't be selected among "lowest")
        z_masked = np.where(within, z_neighbors, np.inf)

        # If there are rows with no neighbors, fill them from nearest neighbor directly:
        if not np.all(has_any):
            rows = np.where(~has_any)[0]
            z_out[i0:i1][rows] = z_neighbors[rows, 0].astype(np.float32)

        # For rows with any neighbors inside radius, take mean of lowest `keep`
        rows_ok = np.where(has_any)[0]
        if rows_ok.size > 0:
            z_ok = z_masked[rows_ok]  # (R,k)
            # partition to get the smallest `keep` per row
            kth = keep - 1
            part = np.partition(z_ok, kth=kth, axis=1)[:, :keep]  # (R,keep)
            # guard infinite (if fewer than keep were within radius)
            finite_mask = np.isfinite(part)
            count = finite_mask.sum(axis=1)
            # fallback where count==0 should already be handled; for safety:
            safe_count = np.where(count == 0, 1, count)
            sums = np.where(finite_mask, part, 0.0).sum(axis=1)
            z_est = (sums / safe_count).astype(np.float32)
            z_out[i0:i1][rows_ok] = z_est

    return z_out

def gaussian_smooth_masked(H: np.ndarray,
                           mask: np.ndarray,
                           sigma_m: float,
                           scale: float) -> np.ndarray:
    """
    NaN-aware Gaussian smoothing inside `mask`:
      - convert meters -> pixels for sigma
      - normalized convolution: blur(H*mask) / blur(mask)
      - preserves NaNs outside mask
    """
    if sigma_m <= 0:
        return H
    sigma_px = max(1e-3, float(sigma_m) * float(scale))
    ksize = int(2 * math.ceil(3.0 * sigma_px) + 1)  # ~±3σ, force odd
    # Build weighted blur
    Mf = mask.astype(np.float32)
    Hf = np.where(np.isfinite(H), H, 0.0).astype(np.float32)
    num = cv2.GaussianBlur(Hf * Mf, (ksize, ksize), sigma_px, borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(Mf,         (ksize, ksize), sigma_px, borderType=cv2.BORDER_REPLICATE)
    out = np.where(den > 1e-6, num / (den + 1e-12), np.nan).astype(np.float32)
    # keep values only inside mask; NaN elsewhere
    out[~mask] = np.nan
    return out


# ======== Main builders ========
def build_maps(osm_path: str,
               pcd_path: str,
               output_folder: str,
               img_size: int,
               scale: float,
               ref_easting: float,
               ref_northing: float,
               ref_elevation: float,
               anchor_east_shift: float,
               anchor_north_shift: float,
               neighbor_radius_m: float = 1.5,
               knn: int = 48,
               lowest_frac: float = 0.10,
               smooth_sigma_m: float = 0.5):

    os.makedirs(output_folder, exist_ok=True)

    # Parse OSM & build lanelet BEV
    raw_nodes, ways, lanelets = parse_osm(osm_path)

    # raw_nodes already stores local metric coordinates (local_x, local_y)
    nodes_xy = raw_nodes  # dict: node_id -> (x_m, y_m)


    # Image anchor
    min_x = anchor_east_shift
    max_y = anchor_north_shift

    bev = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    # Draw lanelets (BEV only; heights come from PCD later)
    for ll in lanelets:
        left_ids  = ways.get(ll["left"])
        right_ids = ways.get(ll["right"])
        if not left_ids or not right_ids:
            continue
        poly_pts = []
        for nid in left_ids:
            if nid in nodes_xy:
                x, y = nodes_xy[nid]
                ix = int((x - min_x) * scale)
                iy = int((max_y - y) * scale)
                poly_pts.append((ix, iy))
        for nid in reversed(right_ids):
            if nid in nodes_xy:
                x, y = nodes_xy[nid]
                ix = int((x - min_x) * scale)
                iy = int((max_y - y) * scale)
                poly_pts.append((ix, iy))
        if len(poly_pts) < 3:
            continue
        color = LANELET_COLORS.get(ll["subtype"], LANELET_COLORS["unknown"])
        cv2.fillPoly(bev, [np.asarray(poly_pts, dtype=np.int32)], color)

    # Expand the non-white region of BEV to include a small buffer
    expansion_grid = int(2 * HALF_EXPANSION_SIZE * scale) + 1
    kernel = np.ones((expansion_grid, expansion_grid), np.uint8)
    mask = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    mask[mask == 255] = 0
    mask[mask != 0] = 255
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    diff_mask = cv2.absdiff(mask, expanded_mask)
    diff_mask = cv2.bitwise_not(diff_mask)
    diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)
    bev = cv2.bitwise_and(bev, diff_mask)

    # Lane region (non-white)
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    lane_mask_all = gray < 255

    # Prepare pixel centers for lane region
    ys_all, xs_all = np.where(lane_mask_all)
    xP = (min_x + (xs_all + 0.5) / scale).astype(np.float32)
    yP = (max_y - (ys_all + 0.5) / scale).astype(np.float32)

    # Load PCD & convert to metric frame (match the image)
    pts_mgrs = load_pcd_xyz(pcd_path)  # columns: e_mgrs, n_mgrs, z
    if pts_mgrs.size == 0:
        raise RuntimeError(f"No points loaded from {pcd_path}")
    pc_xyzh = pcd_to_metric_xyzh(
        pts_mgrs,
        mgrs_ref_e=PCD_REF_EASTING_MGRS,
        mgrs_ref_n=PCD_REF_NORTHING_MGRS,
        ref_elevation=ref_elevation
    )  # columns: x_metric, y_metric, z_rel

    # Remove points whose z is too large
    z_threshold = 0.3  # meters
    pc_xyzh = pc_xyzh[np.abs(pc_xyzh[:, 2]) <= z_threshold]

    # Compute height from PCD for all lane pixels (robust ground estimate)
    z_vals = height_from_pcd_for_pixels(
        xP, yP, pc_xyzh,
        radius_m=neighbor_radius_m,
        k=knn,
        lowest_frac=lowest_frac,
        chunk_size=200_000
    )


    # Scatter into the image grid
    H = np.full((img_size, img_size), np.nan, dtype=np.float32)
    H[ys_all, xs_all] = z_vals

    # Optional: small residual fill using neighbors in raster space (fast IDW)
    # Build known/target masks
    known_mask  = np.isfinite(H) & lane_mask_all
    target_mask = lane_mask_all & ~np.isfinite(H)
    if np.any(target_mask):
        # Reuse KD-tree + vectorized IDW from earlier implementation
        idw_fill_from_known_pixels(H, known_mask, target_mask, radius_px=35, k=24, power=2.0)

    # Optional Gaussian smoothing (meters -> pixels internally)
    if smooth_sigma_m > 0:
        H = gaussian_smooth_masked(H, lane_mask_all, smooth_sigma_m, scale)

    # ---------- Visualization & save ----------
    H_valid = H[lane_mask_all]
    if H_valid.size == 0:
        raise RuntimeError("No pixels inside lane regions—check anchor/scale/extents.")
    hmin, hmax = np.nanpercentile(H_valid, [1, 99])
    if not np.isfinite(hmin) or not np.isfinite(hmax) or abs(hmax - hmin) < 1e-6:
        hmin, hmax = float(np.nanmin(H_valid)), float(np.nanmax(H_valid) + 1e-3)

    H_vis = np.full_like(bev, 255, dtype=np.uint8)
    H_norm = np.clip((H - hmin) / (hmax - hmin + 1e-12), 0.0, 1.0)
    rgb = (cm.get_cmap("rainbow")(H_norm)[..., :3] * 255.0).astype(np.uint8)
    H_vis[lane_mask_all] = rgb[lane_mask_all]

    draw_vertical_colorbar(
        H_vis, vmin=hmin, vmax=hmax, cmap=cm.get_cmap("rainbow"),
        bar_h=200, bar_w=20, margin=20, tick_count=5, decimals=1, font_scale=0.5
    )

    # Save outputs
    os.makedirs(output_folder, exist_ok=True)
    bev_path = os.path.join(output_folder, "bev_hdmap.png")
    height_color_path = os.path.join(output_folder, "height_color_from_pcd.png")
    height_float_path = os.path.join(output_folder, "height_from_pcd_float32.npy")
    height_16u_path = os.path.join(output_folder, "height_from_pcd_16u.png")

    cv2.imwrite(bev_path, bev)
    cv2.imwrite(height_color_path, H_vis)
    np.save(height_float_path, H.astype(np.float32))

    H_clip = np.clip(H, hmin, hmax)
    H_16u = np.round((H_clip - hmin) / (hmax - hmin + 1e-12) * 65535.0).astype(np.uint16)
    H_16u[~lane_mask_all] = 0
    cv2.imwrite(height_16u_path, H_16u)

    print(f"[OK] BEV saved:               {bev_path}")
    print(f"[OK] Height (color, PCD):     {height_color_path}")
    print(f"[OK] Height (float32, PCD):   {height_float_path}")
    print(f"[OK] Height (16-bit, PCD):    {height_16u_path}")

# ----- Fast IDW from known pixels (KD-tree batches) -----
def idw_fill_from_known_pixels(H: np.ndarray,
                               known_mask: np.ndarray,
                               target_mask: np.ndarray,
                               radius_px: int = 35,
                               k: int = 24,
                               eps: float = 1e-6,
                               power: float = 2.0,
                               chunk_size: int = 200_000):
    ys_k, xs_k = np.where(known_mask)
    if xs_k.size == 0:
        return
    z_k = H[ys_k, xs_k].astype(np.float32)
    ys_t, xs_t = np.where(target_mask)
    if xs_t.size == 0:
        return
    pts_k = np.column_stack((xs_k.astype(np.float32), ys_k.astype(np.float32)))
    tree = KDTree(pts_k)

    k_eff = int(min(k, xs_k.size))
    for i0 in range(0, xs_t.size, chunk_size):
        i1 = min(xs_t.size, i0 + chunk_size)
        xt = xs_t[i0:i1].astype(np.float32)
        yt = ys_t[i0:i1].astype(np.float32)
        pts_t = np.column_stack((xt, yt))
        dist, ind = tree.query(pts_t, k=k_eff, return_distance=True)

        within = dist <= float(radius_px)
        has_any = within.any(axis=1)
        if not np.all(has_any):
            within[np.where(~has_any)[0], 0] = True

        with np.errstate(divide='ignore', invalid='ignore'):
            w = 1.0 / np.maximum(dist, eps) ** power
        w[~within] = 0.0

        wsum = w.sum(axis=1, keepdims=True) + 1e-12
        w_norm = w / wsum

        z_neighbors = z_k[ind]
        z_est = np.einsum("ij,ij->i", w_norm, z_neighbors).astype(np.float32)

        exact_hit = dist <= eps
        if exact_hit.any():
            first_exact = exact_hit.argmax(axis=1)
            has_exact = exact_hit.any(axis=1)
            z_exact = z_neighbors[np.arange(z_neighbors.shape[0]), first_exact]
            z_est = np.where(has_exact, z_exact, z_est).astype(np.float32)

        H[yt.astype(int), xt.astype(int)] = z_est

def main():
    ap = argparse.ArgumentParser(description="BEV HD map from OSM + height map from PCD (robust ground).")
    ap.add_argument("--osm", default=DEFAULT_OSM_PATH)
    ap.add_argument("--pcd", default=DEFAULT_PCD_PATH)
    ap.add_argument("--out", default=DEFAULT_OUTPUT_FOLDER)
    ap.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    ap.add_argument("--ref_easting", type=float, default=DEFAULT_REF_EASTING)
    ap.add_argument("--ref_northing", type=float, default=DEFAULT_REF_NORTHING)
    ap.add_argument("--ref_elevation", type=float, default=DEFAULT_REF_ELEVATION)
    ap.add_argument("--anchor_shift_east", type=float, default=DEFAULT_ANCHOR_EAST_SHIFT)
    ap.add_argument("--anchor_shift_north", type=float, default=DEFAULT_ANCHOR_NORTH_SHIFT)
    ap.add_argument("--radius", type=float, default=0.3, help="Neighborhood radius (m) for ground estimate")
    ap.add_argument("--knn", type=int, default=200, help="k-NN for PCD search")
    ap.add_argument("--lowest_frac", type=float, default=0.01, help="Use mean of lowest fraction of z within radius")
    ap.add_argument("--smooth_sigma_m", type=float, default=1.0, help="Gaussian smoothing sigma (m); 0=none")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    build_maps(args.osm, args.pcd, args.out, args.img_size, args.scale,
               args.ref_easting, args.ref_northing, args.ref_elevation,
               args.anchor_shift_east, args.anchor_shift_north,
               neighbor_radius_m=args.radius, knn=args.knn, lowest_frac=args.lowest_frac, smooth_sigma_m=args.smooth_sigma_m)

if __name__ == "__main__":
    main()
