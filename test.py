# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Generate offline synthetic dataset (single Jackal-mounted camera, no camera randomization)
"""

import argparse
import json
import math
import os
import random
import asyncio

import yaml
from isaacsim import SimulationApp
#import isaacsim.replicator.agent.core as agent_core  # ensure package available

# ---- CONFIG -----------------------------------------------------------------
config = {
    "launch_config": {
        "renderer": "RaytracedLighting",
        "headless": False,
    },
    "resolution": [512, 512],
    "rt_subframes": 16,
    "num_frames": 20,
    "fps": 30.0,  # used to advance time realistically
    "env_url": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "writer": "BasicWriter",
    "writer_config": {
        "output_dir": "_out_scene_based_sdg",
        "rgb": True,
        "bounding_box_2d_tight": True,
        "semantic_segmentation": True,
        "distance_to_image_plane": True,
        "bounding_box_3d": True,
        "occlusion": True,
    },
    "clear_previous_semantics": True,
    "forklift": {
        "url": "/Isaac/Props/Forklift/forklift.usd",
        "class": "forklift",
    },
    "cone": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
        "class": "traffic_cone",
    },
    "pallet": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
        "class": "pallet",
    },
    "cardbox": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
        "class": "cardbox",
    },
    "close_app_after_run": True,
    # --- People Randomizer params ---
    "people": {
        "max_agents": 10,           # how many People to control (cap)
        "use_navmesh": False,       # set True if you have a baked navmesh/area
        "custom_transition_map": "" # optional path to your JSON; empty = use default from extension
    },
    # --- Jackal motion (example) ---
    "jackal_motion": {
        "enable": True,
        "speed_mps": 1.0,           # forward speed along +X (m/s)
    },
}

import carb

# Robot asset (relative to Nucleus root)
ROBOT_USD_REL = "/Isaac/Robots/Clearpath/Jackal/jackal.usd"

# ---- ARGS -------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    print("File exist")
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            carb.log_warn(f"File {args.config} is not json or yaml, will use default config")
else:
    carb.log_warn(f"File {args.config} does not exist, will use default config")

# Merge config (if input provides writer_config, clear defaults first)
if "writer_config" in args_config:
    config["writer_config"].clear()
config.update(args_config)

# ---- APP --------------------------------------------------------------------
simulation_app = SimulationApp(launch_config=config["launch_config"])

# ---- LATE IMPORTS (require SimulationApp) -----------------------------------
import omni.kit.app
import omni.replicator.core as rep
import omni.usd

# Custom util functions for this sample
import scene_based_sdg_utils
from isaacsim.core.utils import prims
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_current_stage, open_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf

# ---- OPEN STAGE -------------------------------------------------------------
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not get nucleus server path, closing application..")
    simulation_app.close()

print(f"[scene_based_sdg] Loading Stage {config['env_url']}")
if not open_stage(assets_root_path + config["env_url"]):
    carb.log_error(f"Could not open stage {config['env_url']}, closing application..")
    simulation_app.close()

# Disable capture on play (we will step manually)
rep.orchestrator.set_capture_on_play(False)

# Clear any previous semantic data in the loaded stage
if config["clear_previous_semantics"]:
    stage = get_current_stage()
    scene_based_sdg_utils.remove_previous_semantics(stage)

# ---- SCENE CONTENT ----------------------------------------------------------
# 1) Forklift at random pose
forklift_prim = prims.create_prim(
    prim_path="/World/Forklift",
    position=(random.uniform(-20, -2), random.uniform(-1, 3), 0),
    orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
    usd_path=assets_root_path + config["forklift"]["url"],
    semantic_label=config["forklift"]["class"],
)

# 2) Pallet in front of forklift (random offset along forklift's forward)
forklift_tf = omni.usd.get_world_transform_matrix(forklift_prim)
pallet_offset_tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, random.uniform(-1.2, -1.8), 0))
pallet_pos_gf = (pallet_offset_tf * forklift_tf).ExtractTranslation()
forklift_quat_gf = forklift_tf.ExtractRotationQuat()
forklift_quat_xyzw = (forklift_quat_gf.GetReal(), *forklift_quat_gf.GetImaginary())

pallet_prim = prims.create_prim(
    prim_path="/World/Pallet",
    position=pallet_pos_gf,
    orientation=forklift_quat_xyzw,
    usd_path=assets_root_path + config["pallet"]["url"],
    semantic_label=config["pallet"]["class"],
)

# 3) Register randomization graphs for props/lights (unchanged)
scene_based_sdg_utils.register_scatter_boxes(pallet_prim, assets_root_path, config)
scene_based_sdg_utils.register_cone_placement(forklift_prim, assets_root_path, config)
scene_based_sdg_utils.register_lights_placement(forklift_prim, pallet_prim)

# 4) Jackal robot
jackal_prim = prims.create_prim(
    prim_path="/World/Robots/Jackal",
    position=(pallet_pos_gf[0] + 2.0, pallet_pos_gf[1] + 1.0, 0.0),
    orientation=(0, 0, 0, 1),
    usd_path=assets_root_path + ROBOT_USD_REL,
    semantic_label="jackal",
)

# 5) SINGLE CAMERA: mounted on the Jackal (follows robot automatically)
jackal_cam = rep.create.camera(
    name="JackalCam",
    parent="/World/Robots/Jackal",  # parented -> follows robot motion
    position=(0.0, 0.0, 0.6),       # relative to robot body
    rotation=(0.0, 0.0, 0.0),       # no randomization
    clipping_range=(0.1, 100000.0),
    focal_length=16.0,
    focus_distance=400.0,
)

# 6) Render product for the single camera
resolution = tuple(config.get("resolution", (512, 512)))
jackal_rp = rep.create.render_product(jackal_cam, resolution, name="JackalView")
rps = [jackal_rp]

# Disable updates until we start SDG
for rp in rps:
    rp.hydra_texture.set_updates_enabled(False)

# ---- OUTPUT + WRITER --------------------------------------------------------
# Ensure absolute output dir
if not os.path.isabs(config["writer_config"]["output_dir"]):
    config["writer_config"]["output_dir"] = os.path.join(os.getcwd(), config["writer_config"]["output_dir"])
os.makedirs(config["writer_config"]["output_dir"], exist_ok=True)
print(f"[scene_based_sdg] Output directory={config['writer_config']['output_dir']}")

# Writer
writer_type = config.get("writer", "BasicWriter")
if writer_type not in rep.WriterRegistry.get_writers():
    carb.log_error(f"Writer type {writer_type} not found, closing application..")
    simulation_app.close()

writer = rep.WriterRegistry.get(writer_type)
writer_kwargs = config["writer_config"]
print(f"[scene_based_sdg] Initializing {writer_type} with: {writer_kwargs}")
writer.initialize(**writer_kwargs)
writer.attach(rps)

# ---- REPLICATOR RANDOMIZERS (NO CAMERA RANDOM JITTER) -----------------------
with rep.trigger.on_frame():
    rep.randomizer.scatter_boxes()
    rep.randomizer.randomize_lights()

# Optional: on-demand cones
with rep.trigger.on_custom_event("randomize_cones"):
    rep.randomizer.place_cones()

# ---- PEOPLE CHARACTER RANDOMIZER -------------------------------------------
# Ensure required extensions are enabled
em = omni.kit.app.get_app().get_extension_manager()
for ext in ("omni.anim.people", "isaacsim.replicator.agent", "isaacsim.replicator.agent.core"):
    try:
        em.set_extension_enabled_immediate(ext, True)
    except Exception:
        pass

# Import CharacterRandomizer
try:
    from isaacsim.replicator.agent.core.character_randomizer import CharacterRandomizer
except ImportError:
    from isaacsim.replicator.agent.core import CharacterRandomizer  # type: ignore

# Create & configure
rand = CharacterRandomizer(global_seed=1234)

custom_map = (config.get("people") or {}).get("custom_transition_map", "")
if custom_map:
    if os.path.isfile(custom_map):
        rand.load_command_transition_map(os.path.abspath(custom_map))
        carb.log_info(f"[people] Loaded custom transition map: {custom_map}")
    else:
        carb.log_warn(f"[people] custom_transition_map not found: {custom_map}; using default")

# Plan duration based on frames/fps
num_frames = int(config.get("num_frames", 0))
fps = float(config.get("fps", 30.0))
active_seconds = float(num_frames) / fps

# Build character plan (requires People characters present in the stage)
plan = asyncio.get_event_loop().run_until_complete(
    rand.generate_character_commands(
        global_seed=4321,
        duration=active_seconds,
        agent_count=int(config.get("people", {}).get("max_agents", 10)),
        navigation_area=None if not config.get("people", {}).get("use_navmesh", False) else None  # plug your area if any
    )
)

# ---- OPTIONAL: SIMPLE JACKAL FORWARD MOTION EACH FRAME ----------------------
# This keeps the camera view interesting but the camera stays rigidly attached to the robot.
_j_world_tf = omni.usd.get_world_transform_matrix(jackal_prim)
_j_init = _j_world_tf.ExtractTranslation()  # Gf.Vec3d
jackal_pos = [float(_j_init[0]), float(_j_init[1]), float(_j_init[2])]
jackal_speed_mps = float(config.get("jackal_motion", {}).get("speed_mps", 1.0))
enable_jackal_motion = bool(config.get("jackal_motion", {}).get("enable", True))

def move_jackal_step(dt_seconds: float):
    if not enable_jackal_motion:
        return
    jackal_pos[0] += jackal_speed_mps * dt_seconds  # advance along +X in world
    prims.set_world_pose(
        prim_path="/World/Robots/Jackal",
        position=tuple(jackal_pos),
        orientation=(0, 0, 0, 1)
    )

# ---- RUN SOME PHYSICS BEFORE CAPTURE (OPTIONAL) -----------------------------
scene_based_sdg_utils.simulate_falling_objects(forklift_prim, assets_root_path, config)

# ---- ENABLE RPs AND START SDG ----------------------------------------------
rt_subframes = config.get("rt_subframes", -1)
for rp in rps:
    rp.hydra_texture.set_updates_enabled(True)

print(f"[scene_based_sdg] Running SDG for {num_frames} frames @ {fps} FPS (dt={1.0/fps:.4f}s)")
dt = 1.0 / fps
headless = bool(config["launch_config"].get("headless", False))

for i in range(num_frames):
    print(f"[scene_based_sdg] \t Capturing frame {i}")

    # optional scene change
    if i % 2 == 0:
        rep.utils.send_og_event(event_name="randomize_cones")

    # >>> advance sim time (enables character actions / physics to progress)
    rep.orchestrator.step(delta_time=dt, rt_subframes=rt_subframes)

    # Move Jackal deterministically (camera follows because it's parented)
    move_jackal_step(dt)

    if not headless:
        simulation_app.update()

# ---- WRAP UP ----------------------------------------------------------------
rep.orchestrator.wait_until_complete()

writer.detach()
for rp in rps:
    rp.destroy()

close_app_after_run = bool(config.get("close_app_after_run", True))
if headless:
    if not close_app_after_run:
        print("[scene_based_sdg] 'close_app_after_run' is ignored when running headless. The application will be closed.")
elif not close_app_after_run:
    print("[scene_based_sdg] The application will not be closed after the run. Make sure to close it manually.")
    while simulation_app.is_running():
        simulation_app.update()

simulation_app.close()