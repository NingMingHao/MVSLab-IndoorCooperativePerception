#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import cv2
import math
import yaml
import argparse
import numpy as np
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================= Math & Color =========================

def invert_se3(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti

def _gamma_decode_srgb(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _gamma_encode_srgb(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0.0, 1.0)
    a = 0.055
    return (np.where(y <= 0.0031308, 12.92 * y, (1 + a) * (y ** (1/2.4)) - a) * 255.0).astype(np.uint8)

def colorize_scalar_to_bgr(vals: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    vals_clipped = np.clip(vals, vmin, vmax)
    vmax = max(vmax, vmin + 1e-6)
    norm = ((vals_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)  # Nx1x3 or Nx3
    if bgr.ndim == 3:
        bgr = bgr.reshape(-1,3)
    return bgr

# ========================= Calib & Discovery =====================

IMG_NAME_RE = re.compile(r"^(?P<node>\d+)_(?P<side>left|right)_(?P<ts>[-\d\.]+)\.(jpg|jpeg|png)$", re.IGNORECASE)

def list_images_in_group(cam_dir: Path) -> Dict[str, Dict[str, List[Tuple[str, Path]]]]:
    out: Dict[str, Dict[str, List[Tuple[str, Path]]]] = {}
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        for p in cam_dir.glob(ext):
            m = IMG_NAME_RE.match(p.name)
            if not m:
                continue
            node = m.group("node")
            side = m.group("side").lower()
            ts   = m.group("ts")
            out.setdefault(node, {}).setdefault(side, []).append((ts, p))
    for node in out.keys():
        for side in out[node].keys():
            out[node][side].sort(key=lambda x: x[0])
    return out

def find_pcimage_groups(camera_root: Path) -> List[str]:
    groups = []
    if not camera_root.exists():
        return groups
    for g in sorted(camera_root.iterdir()):
        cam_dir = g / "camera"
        if cam_dir.is_dir() and any(list(cam_dir.glob("*.jpg")) + list(cam_dir.glob("*.png"))):
            groups.append(g.name)
    return groups

def _parse_ts(ts: str) -> float:
    try:
        return float(ts)
    except ValueError:
        m = re.search(r"[-+]?\d*\.\d+|\d+", ts)
        return float(m.group(0)) if m else 0.0

def align_lr_pairs(pair_list_L: List[Tuple[str, Path]],
                   pair_list_R: List[Tuple[str, Path]],
                   max_dt: float = 0.050) -> List[Tuple[str, Path, str, Path]]:
    L = sorted([( _parse_ts(ts), ts, p) for ts, p in pair_list_L])
    R = sorted([( _parse_ts(ts), ts, p) for ts, p in pair_list_R])
    i = j = 0
    matches: List[Tuple[str, Path, str, Path]] = []
    while i < len(L) and j < len(R):
        tL, tsL, pL = L[i]
        tR, tsR, pR = R[j]
        dt = tL - tR
        if abs(dt) <= max_dt:
            matches.append((tsL, pL, tsR, pR))
            i += 1; j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1
    if not matches and len(L) == len(R):
        for k in range(len(L)):
            _, tsL, pL = L[k]; _, tsR, pR = R[k]
            matches.append((tsL, pL, tsR, pR))
    return matches

def load_camera_calib(cam_yaml: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(cam_yaml, "r") as f:
        data = yaml.safe_load(f) or {}
    K = np.array(data["intrinsic_matrix"], dtype=np.float64)
    T_l_c = np.array(data["camera_to_lidar"], dtype=np.float64)  # ^L T_C
    T_c_l = invert_se3(T_l_c)                                     # ^C T_L
    return K, T_c_l

def load_lidar_and_global(lidar_yaml: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(lidar_yaml, "r") as f:
        data = yaml.safe_load(f) or {}
    T_g_l = np.array(data["lidar_to_ground"], dtype=np.float64)  # ^G T_L
    T_l_g = invert_se3(T_g_l)                                    # ^L T_G
    if "ground_to_global" in data:
        T_w_g = np.array(data["ground_to_global"], dtype=np.float64)  # ^W T_G
    else:
        T_w_g = np.eye(4, dtype=np.float64)
    return T_l_g, T_w_g

# ========================= World Canvas & DEM ======================

def world_canvas_spec(xmin, xmax, ymin, ymax, res) -> Tuple[int, int]:
    W = int(math.ceil((xmax - xmin) / res))
    H = int(math.ceil((ymax - ymin) / res))
    return W, H

def world_grid_from_canvas(W: int, H: int, xmin: float, ymax: float, res: float) -> Tuple[np.ndarray, np.ndarray]:
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    X = xmin + uu * res
    Y = ymax - vv * res
    return X, Y

# ========================= DEM-aware Projection ====================

def build_uv_and_weights_from_dem(K: np.ndarray,
                                  T_c_w: np.ndarray,
                                  Xw: np.ndarray, Yw: np.ndarray, Zw: np.ndarray,
                                  valid_dem: np.ndarray,
                                  delta_m: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Hh, Ww = Xw.shape
    Pw = np.stack([Xw, Yw, Zw, np.ones_like(Xw)], axis=-1).reshape(-1, 4).T  # 4xN
    Pc = T_c_w @ Pw
    xc, yc, zc = Pc[0, :], Pc[1, :], Pc[2, :]
    valid = (zc > 1e-6) & valid_dem.reshape(-1)
    inv_z = np.zeros_like(zc)
    inv_z[valid] = 1.0 / zc[valid]
    uv = (K @ np.vstack([xc, yc, zc]))
    u = (uv[0, :] * inv_z).astype(np.float32)
    v = (uv[1, :] * inv_z).astype(np.float32)

    rng = np.sqrt(xc*xc + yc*yc + zc*zc, dtype=np.float64)
    w = np.zeros_like(rng, dtype=np.float32)
    w[valid] = (1.0 / (rng[valid].astype(np.float32) + float(delta_m)))**3

    u_map = u.reshape(Hh, Ww)
    v_map = v.reshape(Hh, Ww)
    w_map = w.reshape(Hh, Ww)
    w_map[~valid_dem] = 0.0
    return u_map, v_map, w_map

def apply_black_mask_weight(src_img: np.ndarray,
                            u_map: np.ndarray, v_map: np.ndarray,
                            w_base: np.ndarray,
                            black_thresh: int) -> np.ndarray:
    if black_thresh < 0:
        return w_base
    if src_img.ndim == 3:
        nonblack = (src_img[...,0] > black_thresh) | \
                   (src_img[...,1] > black_thresh) | \
                   (src_img[...,2] > black_thresh)
    else:
        nonblack = (src_img > black_thresh)
    src_mask = (nonblack.astype(np.uint8) * 255)
    mask_canvas = cv2.remap(src_mask, u_map, v_map,
                            interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return w_base * (mask_canvas > 0).astype(np.float32)

def warp_image_with_uv(img: np.ndarray,
                       u_map: np.ndarray, v_map: np.ndarray,
                       w_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    warped = cv2.remap(img, u_map, v_map, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    inmask = cv2.remap(ones, u_map, v_map, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = (inmask > 0).astype(np.float32)
    w = (w_map * m).astype(np.float32)
    I_lin = _gamma_decode_srgb(warped) * w[..., None]
    return I_lin, w

def cosine_gate_world(Xw: np.ndarray, Yw: np.ndarray,
                      cx: float, cy: float,
                      R: float, falloff: float) -> np.ndarray:
    D = np.sqrt((Xw - cx)**2 + (Yw - cy)**2, dtype=np.float32)
    if falloff <= 1e-6:
        return (D <= R).astype(np.float32)
    t = (D - R) / float(falloff)
    t = np.clip(t, 0.0, 1.0)
    G = 0.5 * (1.0 + np.cos(np.pi * t)).astype(np.float32)
    G[D <= R] = 1.0
    G[D >= (R + falloff)] = 0.0
    return G

# ========================= LiDAR → Canvas ==========================

def points_world_to_canvas_uv(
    xyz_w: np.ndarray,
    xmin: float, ymax: float, res: float,
    Ww: int, Hw: int,
    pc_hmin: float, pc_hmax: float,
    H_dem: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = xyz_w.shape[0]
    if N == 0:
        return (np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.zeros((0,), dtype=bool),
                np.empty((0,), dtype=np.float32))

    x = xyz_w[:,0].astype(np.float32)
    y = xyz_w[:,1].astype(np.float32)
    z = xyz_w[:,2].astype(np.float32)

    uf = (x - xmin) / res
    vf = (ymax - y) / res

    u0 = np.floor(uf).astype(np.int32)
    v0 = np.floor(vf).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    Hh, Ww2 = H_dem.shape
    in_bounds = (u0 >= 0) & (v0 >= 0) & (u1 < Ww2) & (v1 < Hh)
    if not np.any(in_bounds):
        return (np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.zeros((N,), dtype=bool),
                np.empty((0,), dtype=np.float32))

    du = (uf - u0).astype(np.float32)
    dv = (vf - v0).astype(np.float32)

    z_dem = np.full(N, np.nan, dtype=np.float32)
    idx = np.where(in_bounds)[0]
    u0i = u0[idx]; v0i = v0[idx]
    u1i = u1[idx]; v1i = v1[idx]
    dui = du[idx]; dvi = dv[idx]

    z00 = H_dem[v0i, u0i]
    z10 = H_dem[v0i, u1i]
    z01 = H_dem[v1i, u0i]
    z11 = H_dem[v1i, u1i]

    w00 = (1.0 - dui) * (1.0 - dvi)
    w10 = (       dui) * (1.0 - dvi)
    w01 = (1.0 - dui) * (       dvi)
    w11 = (       dui) * (       dvi)

    z_interp = z00 * w00 + z10 * w10 + z01 * w01 + z11 * w11
    z_dem[idx] = z_interp.astype(np.float32)

    dem_valid = np.isfinite(z_dem)
    h_res = z - z_dem
    keep = in_bounds & dem_valid & (h_res >= pc_hmin) & (h_res <= pc_hmax)
    if not np.any(keep):
        return (np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                keep,
                np.empty((0,), dtype=np.float32))

    u_int = uf.astype(np.int32)
    v_int = vf.astype(np.int32)
    return u_int[keep], v_int[keep], keep, h_res[keep].astype(np.float32)

def draw_points_on_canvas_fast(
    canvas_bgr: np.ndarray,
    u: np.ndarray, v: np.ndarray,
    colors_bgr: np.ndarray,
    radius: int = 2,
    voxel_px: int = 6
) -> None:
    if u.size == 0:
        return
    uq = (u // voxel_px).astype(np.int32)
    vq = (v // voxel_px).astype(np.int32)
    keys = (vq.astype(np.int64) << 32) + uq.astype(np.int64)
    _, keep = np.unique(keys, return_index=True)

    u_ds = u[keep]; v_ds = v[keep]
    col_ds = colors_bgr[keep]
    Hh, Ww = canvas_bgr.shape[:2]
    for x, y, c in zip(u_ds, v_ds, col_ds):
        if 0 <= x < Ww and 0 <= y < Hh:
            cv2.circle(canvas_bgr, (int(x), int(y)), radius, tuple(int(t) for t in c.tolist()), -1)

# ========================= Slice-level Cache ========================

class SliceCache:
    """Cache per-slice items to reuse for every timestamp in that slice."""
    def __init__(self):
        self.K: Dict[Tuple[int,str], np.ndarray] = {}
        self.T_c_w: Dict[Tuple[int,str], np.ndarray] = {}
        self.cam_xy: Dict[int, Tuple[float,float]] = {}
        self.uv_base: Dict[Tuple[int,str], Tuple[np.ndarray,np.ndarray,np.ndarray]] = {}
        self.gate: Dict[int, np.ndarray] = {}
        self.R_w_l: Dict[int, np.ndarray] = {}
        self.t_w_l: Dict[int, np.ndarray] = {}

# ========================= Per-group Worker =========================

def process_single_group(
    group: str,
    slice_in_dir: Path,
    slice_out_dir: Path,
    nodes: List[int],
    cache: SliceCache,
    Zw: np.ndarray,
    hdmap_canvas: np.ndarray,
    world_res: float,
    world_xmin: float, world_ymax: float,
    black_thresh: int,
    pc_voxel_px: int, pc_radius_px: int,
    pc_hmin: float, pc_hmax: float
) -> Tuple[Path, Path]:
    """
    Processes one timestamp group:
      - builds context mosaic using cached UV/weights + per-image black mask
      - projects LiDAR onto HD map
      - saves <out>/<group>.jpg and related_images/<group>_jpg/context_image_for_<group>.jpg
    """
    Ww, Hw = Zw.shape[1], Zw.shape[0]
    cam_dir = slice_in_dir / "PcImage" / group / "camera"
    imgs_by_node = list_images_in_group(cam_dir)

    mosaic_sum_lin = np.zeros((Hw, Ww, 3), dtype=np.float32)
    mosaic_weight  = np.zeros((Hw, Ww), dtype=np.float32)

    # Build context mosaic
    for n in nodes:
        node_str = str(n)
        if node_str not in imgs_by_node or "left" not in imgs_by_node[node_str] or "right" not in imgs_by_node[node_str]:
            continue
        pairs = align_lr_pairs(imgs_by_node[node_str]["left"], imgs_by_node[node_str]["right"], max_dt=0.050)
        if not pairs:
            continue
        _tsL, imgL_path, _tsR, imgR_path = pairs[0]
        imgL = cv2.imread(str(imgL_path), cv2.IMREAD_COLOR)
        imgR = cv2.imread(str(imgR_path), cv2.IMREAD_COLOR)
        if imgL is None or imgR is None:
            continue
        if (n,"left") not in cache.uv_base or (n,"right") not in cache.uv_base:
            continue
        uL, vL, wL_base = cache.uv_base[(n,"left")]
        uR, vR, wR_base = cache.uv_base[(n,"right")]

        wL = apply_black_mask_weight(imgL, uL, vL, wL_base, black_thresh)
        wR = apply_black_mask_weight(imgR, uR, vR, wR_base, black_thresh)

        # Optional feather
        for w in (wL, wR):
            vis = (w > 0).astype(np.uint8)*255
            if vis.any():
                d = cv2.distanceTransform((vis>0).astype(np.uint8), cv2.DIST_L2, 5)
                d = cv2.GaussianBlur(d, (0,0), sigmaX=1.2)
                d = (d / (d.max() if d.max()>0 else 1.0)).astype(np.float32)
                w *= d

        L_lin, wL_out = warp_image_with_uv(imgL, uL, vL, wL)
        R_lin, wR_out = warp_image_with_uv(imgR, uR, vR, wR)
        mosaic_sum_lin += (L_lin + R_lin)
        mosaic_weight  += (wL_out + wR_out)

    if mosaic_weight.max() > 0:
        denom = np.maximum(mosaic_weight[...,None], 1e-6)
        bev_context = _gamma_encode_srgb(mosaic_sum_lin / denom)
    else:
        bev_context = np.full_like(hdmap_canvas, 127)

    # LiDAR → HD map
    pcd_on_hd = hdmap_canvas.copy()
    lidar_dir = slice_in_dir / "PcImage" / group / "lidar"
    if lidar_dir.is_dir():
        all_u = []
        all_v = []
        all_h = []
        for n in nodes:
            R = cache.R_w_l.get(n, None)
            t = cache.t_w_l.get(n, None)
            if R is None or t is None:
                continue
            node_npys = sorted(glob(str(lidar_dir / f"{n}_*.npy")))
            if not node_npys:
                continue
            Rt = R.T
            for npy_path in node_npys:
                pts = np.load(npy_path)
                if pts.ndim != 2 or pts.shape[1] < 3:
                    continue
                Xl = pts[:, :3].astype(np.float32)
                Xw = Xl @ Rt + t.reshape(1,3).astype(np.float32)
                u_all, v_all, keep, h_sel = points_world_to_canvas_uv(
                    Xw, world_xmin, world_ymax, world_res,
                    Ww, Hw, pc_hmin, pc_hmax, Zw
                )
                if u_all.size:
                    all_u.append(u_all); all_v.append(v_all); all_h.append(h_sel)
        if all_u:
            u = np.concatenate(all_u); v = np.concatenate(all_v); h = np.concatenate(all_h)
            colors_bgr = colorize_scalar_to_bgr(h, pc_hmin, pc_hmax)
            draw_points_on_canvas_fast(
                pcd_on_hd, u, v, colors_bgr,
                radius=pc_radius_px, voxel_px=pc_voxel_px
            )

    # Save in CVAT layout under OUTPUT slice dir
    slice_out_dir.mkdir(parents=True, exist_ok=True)
    related_root = slice_out_dir / "related_images" / f"{group}_jpg"
    related_root.mkdir(parents=True, exist_ok=True)

    main_img_path = slice_out_dir / f"{group}.jpg"
    # ctx_img_path  = related_root / f"context_image_for_{group}.jpg"
    ctx_img_path  = related_root / "bev.jpg"

    cv2.imwrite(str(main_img_path), pcd_on_hd)
    cv2.imwrite(str(ctx_img_path),  bev_context)

    return main_img_path, ctx_img_path

# ========================= Core Processing (Slice) =========================

def process_slice(
    slice_in_dir: Path,
    slice_out_dir: Path,
    nodes: List[int],
    world_res: float,
    world_xmin: float, world_xmax: float,
    world_ymin: float, world_ymax: float,
    Zw: np.ndarray,                  # DEM already loaded & resampled to canvas
    hdmap_canvas: np.ndarray,        # HD map already resampled to canvas
    black_thresh: int,
    near_bias_m: float,
    gate_radius_m: float, gate_falloff_m: float,
    pc_voxel_px: int, pc_radius_px: int,
    pc_hmin: float, pc_hmax: float,
    workers: int
):
    camera_root = slice_in_dir / "PcImage"
    if not camera_root.exists():
        print(f"[warn] Missing PcImage in {slice_in_dir}")
        return
    groups = find_pcimage_groups(camera_root)
    if not groups:
        print(f"[warn] No timestamps under {slice_in_dir}")
        return

    # World canvas & grid (fixed globally)
    Ww, Hw = Zw.shape[1], Zw.shape[0]
    Xw, Yw = world_grid_from_canvas(Ww, Hw, world_xmin, world_ymax, world_res)
    dem_valid = np.isfinite(Zw)

    # Calib dirs (INPUT slice)
    calib_dir = slice_in_dir / "Calibration"
    lidar_calib_dir = calib_dir / "lidar"
    cam_calib_dir   = calib_dir / "camera"

    # Per-slice cache (shared, read-only in worker threads)
    cache = SliceCache()

    # Load lidar->ground and ground->global ONCE; build camera transforms; precompute UV/weights
    for n in nodes:
        lid_yaml = lidar_calib_dir / f"{n}.yaml"
        if not lid_yaml.exists():
            continue
        try:
            T_l_g, T_w_g = load_lidar_and_global(lid_yaml)
        except Exception as e:
            print(f"[{slice_in_dir.name}] Node{n} lidar/global load fail: {e}")
            continue
        T_g_w = invert_se3(T_w_g)
        # Cache ^W T_L = ^W T_G * ^G T_L
        T_w_l = T_w_g @ invert_se3(T_l_g)
        cache.R_w_l[n] = T_w_l[:3,:3].copy()
        cache.t_w_l[n] = T_w_l[:3, 3].copy()

        for side in ("left", "right"):
            cam_yaml = cam_calib_dir / f"{n}_{side}.yaml"
            if not cam_yaml.exists():
                continue
            try:
                K, T_c_l = load_camera_calib(cam_yaml)
            except Exception as e:
                print(f"[{slice_in_dir.name}] Node{n} {side} calib fail: {e}")
                continue
            T_c_w = T_c_l @ T_l_g @ T_g_w
            cache.K[(n, side)] = K
            cache.T_c_w[(n, side)] = T_c_w

        if (n,"left") in cache.T_c_w:
            T_w_c_L = invert_se3(cache.T_c_w[(n,"left")])
            cam_world = T_w_c_L @ np.array([0.0,0.0,0.0,1.0], dtype=np.float64)
            cache.cam_xy[n] = (float(cam_world[0]), float(cam_world[1]))
            cache.gate[n] = cosine_gate_world(Xw, Yw, cache.cam_xy[n][0], cache.cam_xy[n][1],
                                              gate_radius_m, gate_falloff_m)

    # Precompute UV + base weights (DEM-aware) per node/side for the slice
    for (n, side), T_c_w in cache.T_c_w.items():
        K = cache.K[(n, side)]
        u_map, v_map, w_base = build_uv_and_weights_from_dem(
            K, T_c_w, Xw, Yw, Zw, dem_valid, delta_m=near_bias_m
        )
        # Apply static world gate ONCE here
        if n in cache.gate:
            w_base = (w_base * cache.gate[n]).astype(np.float32)
        cache.uv_base[(n, side)] = (u_map, v_map, w_base)

    # Ensure output slice dir exists
    slice_out_dir.mkdir(parents=True, exist_ok=True)

    # --------------- Parallelize every timestamp group ---------------
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for group in groups:
            futures.append(
                ex.submit(
                    process_single_group,
                    group,
                    slice_in_dir,
                    slice_out_dir,
                    nodes,
                    cache,
                    Zw,
                    hdmap_canvas,
                    world_res,
                    world_xmin, world_ymax,
                    black_thresh,
                    pc_voxel_px, pc_radius_px,
                    pc_hmin, pc_hmax
                )
            )
        for fut in as_completed(futures):
            try:
                main_img_path, ctx_img_path = fut.result()
                print(f"[ok] {main_img_path} (points on HD map)")
                print(f"[ok] {ctx_img_path}  (context BEV)")
            except Exception as e:
                print(f"[error] group failed: {e}")

# ========================= Entry Point ============================

def main():
    ap = argparse.ArgumentParser(description="Batch scenarios/slices/timestamps → (HD-map pointcloud, BEV context) with caching + multithreading.")
    ap.add_argument("--root", default="/media/minghao/Data2TB/IndoorData/TwoNodeSapCopeExpCoInfra", help="Root containing scenario folders.")
    ap.add_argument("--out_root", default="/media/minghao/Data2TB/IndoorData/TwoNodeSapCopeExpCVAT", help="Output root for CVAT-formatted results.")
    ap.add_argument("--scenarios", nargs="*", default=[
        "NoBed",
        "WithBed",
    ])
    ap.add_argument("--nodes", nargs="*", type=int, default=[1,3])

    # Global world canvas
    ap.add_argument("--world_res",  type=float, default=0.02)
    ap.add_argument("--world_xmin", type=float, default=-6)
    ap.add_argument("--world_xmax", type=float, default=18)
    ap.add_argument("--world_ymin", type=float, default=-4)
    ap.add_argument("--world_ymax", type=float, default= 20)

    # Shared HD map & DEM
    ap.add_argument("--global_map_dir", default="/home/minghao/Documents/Gits/IndoorSensorNodes/MVSLab-IndoorCooperativePerception/labeling/pointcloudTF/osm_bev_height_from_pcd",
                    help="Directory that contains BOTH bev_hdmap.png and height_from_pcd_float32.npy")
    ap.add_argument("--hdmap_name", default="bev_hdmap.png")
    ap.add_argument("--height_npy_name", default="height_from_pcd_float32.npy")
    ap.add_argument("--height_offset_m", type=float, default=0.0)

    # Image weighting/masking
    ap.add_argument("--black_thresh", type=int, default=1)
    ap.add_argument("--near_bias_m", type=float, default=0.0)

    # Gating
    ap.add_argument("--gate_radius_m",   type=float, default=15.0)
    ap.add_argument("--gate_falloff_m",  type=float, default=5.0)

    # LiDAR overlay (on HD map)
    ap.add_argument("--pc_voxel_px", type=int, default=3)
    ap.add_argument("--pc_radius_px", type=int, default=3)
    ap.add_argument("--pc_hmin", type=float, default=0.2)
    ap.add_argument("--pc_hmax", type=float, default=2.1)

    # Parallelism
    default_workers = min(12, os.cpu_count() or 4)
    ap.add_argument("--workers", type=int, default=default_workers, help="Threads for processing timestamps per slice.")

    args = ap.parse_args()

    # ---------- Global: load DEM & HD map ONCE ----------
    Ww, Hw = world_canvas_spec(args.world_xmin, args.world_xmax,
                               args.world_ymin, args.world_ymax,
                               args.world_res)

    # DEM
    dem_path = Path(args.global_map_dir) / args.height_npy_name
    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    H_dem = np.load(str(dem_path)).astype(np.float32) + float(args.height_offset_m)
    if H_dem.shape != (Hw, Ww):
        H_dem = cv2.resize(H_dem, (Ww, Hw), interpolation=cv2.INTER_AREA)

    # HD map
    hdmap_path = Path(args.global_map_dir) / args.hdmap_name
    if not hdmap_path.is_file():
        raise FileNotFoundError(f"HD map not found: {hdmap_path}")
    hdmap_img = cv2.imread(str(hdmap_path), cv2.IMREAD_COLOR)
    if hdmap_img is None:
        raise RuntimeError(f"Failed to read HD map: {hdmap_path}")
    if hdmap_img.shape[:2] != (Hw, Ww):
        hdmap_img = cv2.resize(hdmap_img, (Ww, Hw), interpolation=cv2.INTER_AREA)

    # ---------- Batch all scenarios / slices ----------
    root_in  = Path(args.root).expanduser().resolve()
    root_out = Path(args.out_root).expanduser().resolve()
    root_out.mkdir(parents=True, exist_ok=True)

    for scen in args.scenarios:
        scen_in  = root_in  / scen
        scen_out = root_out / scen
        if not scen_in.is_dir():
            print(f"[warn] Scenario not found: {scen_in}")
            continue
        scen_out.mkdir(parents=True, exist_ok=True)

        for slice_in_dir in sorted(scen_in.iterdir()):
            if not slice_in_dir.is_dir():
                continue
            if not (slice_in_dir / "PcImage").exists():
                continue
            slice_out_dir = scen_out / slice_in_dir.name
            # if slice_in_dir.name != "slice1":
            #     continue  # For testing, only process slice0
            print(f"\n=== Processing slice: {slice_in_dir} → {slice_out_dir} ===")
            process_slice(
                slice_in_dir=slice_in_dir,
                slice_out_dir=slice_out_dir,
                nodes=args.nodes,
                world_res=args.world_res,
                world_xmin=args.world_xmin, world_xmax=args.world_xmax,
                world_ymin=args.world_ymin, world_ymax=args.world_ymax,
                Zw=H_dem,                      # shared DEM
                hdmap_canvas=hdmap_img,        # shared HD map
                black_thresh=args.black_thresh,
                near_bias_m=args.near_bias_m,
                gate_radius_m=args.gate_radius_m,
                gate_falloff_m=args.gate_falloff_m,
                pc_voxel_px=args.pc_voxel_px,
                pc_radius_px=args.pc_radius_px,
                pc_hmin=args.pc_hmin, pc_hmax=args.pc_hmax,
                workers=max(1, int(args.workers))
            )

if __name__ == "__main__":
    main()
