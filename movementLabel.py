#!/usr/bin/env python3
"""
Compute robot motion description from trajectory CSV, allowing
simultaneous turning + translation.

Assumes CSV has columns:
  time_s, frame, prim_path, tx, ty, tz, roll, pitch, yaw
where yaw is in radians.
"""

import os
import math
import pandas as pd
import numpy as np

# ------------------ CONFIG ------------------
CSV_PATH = "/home/deepak/WarehouseResultrobotnewmult2/seed_13765_numseed_935590915_np/trajectories/robots/World_Robots_Nova_Carter_trajectory.csv"
OUT_PER_FRAME = CSV_PATH.replace(".csv", "_with_motion_new.csv")
OUT_SEGMENTS  = CSV_PATH.replace(".csv", "_motion_segments_new.csv")

WINDOW_RADIUS = 0              # +/- 5 frames for smoothing
WINDOW_SIZE   = 2 * WINDOW_RADIUS + 1

# Thresholds (tune for your robot)
V_EPS          = 0.02             # m/s, below this ~ no translation
V_MOVE         = 0.05             # m/s, consider as meaningful motion
V_LAT_MOVE     = 0.05             # m/s, consider as lateral motion
W_EPS          = math.radians(3)  # rad/s, below this ~ no rotation
W_TURN         = math.radians(8)  # rad/s, above this ~ meaningful turning

# -------------------------------------------------

def wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def classify_action(v_fwd: float, v_lat: float, w: float) -> str:
    """
    Classify motion using forward velocity in body frame (v_fwd, signed)
    and yaw rate w (rad/s).

    Returns one of:
      - stationary
      - in-place-turn-left / in-place-turn-right / in-place-rotation-jitter
      - translation-jitter / translation-jitter-turn-left / ...
      - forward-straight / reverse-straight
      - forward-turn-left / forward-turn-right
      - forward-rotation-jitter
      (You can collapse these later if you want simpler labels.)
    """
    v_abs = abs(v_fwd)
    v_lat_asb = abs(v_lat)
    w_abs = abs(w)

    # 1) Almost no translation
    if v_abs < V_EPS and v_lat_asb < V_EPS:
        # 1a) No real rotation either -> stationary
        if w_abs < W_EPS:
            return "stationary"
        # 1b) Strong rotation -> in-place turn
        if w_abs > W_TURN:
            turn_side = "left" if w > 0 else "right"
            return f"in-place-turn-{turn_side}"
        # 1c) Weak/uncertain rotation
        return "in-place-rotation-jitter"
    
    if v_abs < V_EPS and v_lat_asb > V_LAT_MOVE:
        side = "left" if v_lat > 0 else "right"
        if w_abs < W_EPS:
            return f"side-slip-{side}"
        if w_abs > W_TURN:
            turn_side = "left" if w > 0 else "right"
            return f"side-slip-{side}-turn-{turn_side}"
        else:
            return f"side-slip-{side}-rotation-jitter"
        
    # 2) Small translation (jitter region)
    if V_EPS <= v_abs < V_MOVE:
        if w_abs < W_EPS:
            return "translation-jitter"
        if w_abs > W_TURN:
            turn_side = "left" if w > 0 else "right"
            return f"translation-jitter-turn-{turn_side}"
        return "translation-jitter-rotation-jitter"

    # 3) Meaningful translation (forward or reverse)
    if v_abs >= V_MOVE:
        moving_forward = v_fwd > 0
        direction_prefix = "forward" if moving_forward else "reverse"

        if w_abs < W_EPS:
            return f"{direction_prefix}-straight"

        if w_abs > W_TURN:
            turn_side = "left" if w > 0 else "right"
            return f"{direction_prefix}-turn-{turn_side}"

        return f"{direction_prefix}-rotation-jitter"

    return "unclassified"

def angle_bin(total_deg: float) -> str:
    """Bin net turn angle into slight / moderate / sharp / unclear."""
    a = abs(total_deg)
    if a < 5:
        return "jitter (0–5°)"
    elif 5 <= a < 20:
        return "slight (5–20°)"
    elif 20 <= a < 60:
        return "moderate (20–60°)"
    elif 60 <= a <= 120:
        return "sharp (60–120°)"
    elif 120 <= a <= 180:
        return "half-turn (120–180°)"
    else:
        return "unclear"
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    required= {"time_s", "frame", "tx", "ty", "tz", "yaw"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns missing from {CSV_PATH}")
    
    df = df.sort_values("frame").reset_index(drop=True)

    tx = df["tx"].values
    ty = df["ty"].values
    yaw = df["yaw"].values
    t = df["time_s"].values

    #per-frame difference
    dx = np.diff(tx, prepend=tx[0])
    dy = np.diff(ty, prepend=ty[0])
    dt = np.diff(t, prepend=t[0])

    dt = np.where(dt <= 0, np.nan, dt)
    median_dt = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0/30.0
    dt = np.where((dt<=0)|np.isnan(dt), median_dt, dt)

    
    # global linear speed (m/s
    dist = np.sqrt(dx**2 + dy**2)
    v_global = dist / dt

    #yaw rate (rad/s), with wrapping
    dyaw = wrap_to_pi(np.diff(yaw, prepend=yaw[0]))
    w = dyaw / dt

    heading_x = np.cos(yaw)
    heading_y = np.sin(yaw)

    heading_x_prev = np.roll(heading_x, 1)
    heading_y_prev = np.roll(heading_y, 1)
    heading_x_prev[0] = heading_x[0]
    heading_y_prev[0] = heading_y[0]

    proj = dx* heading_x_prev + dy * heading_y_prev
    v_fwd = proj / dt

    lat = -dx*heading_y_prev + dy*heading_x_prev
    v_lat = lat / dt

    # --------- classify per frame ---------
    df["v_forward_med"] = pd.Series(v_global).rolling(
        window=WINDOW_SIZE, center=True, min_periods=1
    ).median()
    df["yaw_rate_med"] = pd.Series(w).rolling(
        window=WINDOW_SIZE, center=True, min_periods=1
    ).median()
    df["v_lateral_med"] = pd.Series(v_lat).rolling(
        window=WINDOW_SIZE, center=True, min_periods=1
    ).median()    # --------- classify per frame ---------
    actions = []
    for vf, vl,wm in zip(df["v_forward_med"], df["v_lateral_med"], df["yaw_rate_med"]):
        actions.append(classify_action(vf, vl, wm))
    df["action"] = actions

    # --------- group into segments ---------
    segments = []
    if len(df) > 0:
        cur_action = df.loc[0, "action"]
        seg_start_idx = 0

        for i in range(1, len(df)):
            if df.loc[i, "action"] != cur_action:
                seg = df.iloc[seg_start_idx:i]
                segments.append(seg)
                seg_start_idx = i
                cur_action = df.loc[i, "action"]
        segments.append(df.iloc[seg_start_idx:])

    seg_rows = []
    for seg in segments:
        a = seg["action"].iloc[0]
        start_frame = int(seg["frame"].iloc[0])
        end_frame   = int(seg["frame"].iloc[-1])
        start_t     = float(seg["time_s"].iloc[0])
        end_t       = float(seg["time_s"].iloc[-1])
        duration    = end_t - start_t

        yaw_seg = seg["yaw"].values
        total_dyaw = wrap_to_pi(yaw_seg[-1] - yaw_seg[0])
        total_deg  = math.degrees(total_dyaw)
        turn_bin   = angle_bin(total_deg)

        # simple description
        if a == "stationary":
            desc = f"Stay stationary for {duration:.2f}s."
        elif a.startswith("in-place-turn"):
            side = "left" if "left" in a else "right"
            desc = f"Turn in place {side} by about {total_deg:+.1f}° ({turn_bin}) over {duration:.2f}s."
        elif "straight" in a:
            direction = "forward" if "forward" in a else "reverse"
            desc = f"Move {direction} straight for {duration:.2f}s."
        elif "-turn-" in a:
            direction = "forward" if "forward" in a else "reverse"
            side = "left" if "left" in a else "right"
            desc = f"Move {direction} while turning {side} by about {total_deg:+.1f}° ({turn_bin}) over {duration:.2f}s."
        else:
            desc = f"{a} for {duration:.2f}s."

        seg_rows.append({
            "action": a,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": start_t,
            "end_time_s": end_t,
            "duration_s": duration,
            "net_yaw_deg": total_deg,
            "turn_angle_bin": turn_bin,
            "description": desc,
        })

    seg_df = pd.DataFrame(seg_rows)

    # --------- save & print summary ---------
    df.to_csv(OUT_PER_FRAME, index=False)
    seg_df.to_csv(OUT_SEGMENTS, index=False)

    print(f"Per-frame motion written to: {OUT_PER_FRAME}")
    print(f"Segment summary written to: {OUT_SEGMENTS}")
    print("\nTop motion segments:")
    with pd.option_context("display.max_rows", 20, "display.width", 160):
        print(seg_df.head(20))

if __name__ == "__main__":
    main()