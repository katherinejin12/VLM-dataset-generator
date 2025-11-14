import argparse
import json
import math
import os
import random
import asyncio
import yaml

import carb
from isaacsim import SimulationApp

# --- Step 1: Load configuration file ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, default="config.yaml", help="Path to config file (YAML or JSON)")
args, unknown = parser.parse_known_args()

if os.path.isfile(args.config):
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            config = json.load(f)
        elif args.config.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported config file format (use .yaml or .json)")
else:
    raise FileNotFoundError(f"Config file {args.config} not found.")

# --- Step 2: Create simulation app ---
simulation_app = SimulationApp(launch_config=config["launch_config"])

# --- Step 3: Runtime imports after app creation ---
import omni.replicator.core as rep
import omni.usd
import omni.kit.app
from isaacsim.core.utils import prims
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_current_stage, open_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf
import scene_based_sdg_utils

# --- Step 4: Open stage ---
assets_root_path = get_assets_root_path()
if not assets_root_path:
    carb.log_error("Could not get nucleus server path.")
    simulation_app.close()

print(f"[scene_based_sdg] Loading Stage {config['env_url']}")
if not open_stage(assets_root_path + config["env_url"]):
    carb.log_error(f"Could not open stage {config['env_url']}.")
    simulation_app.close()

# --- Enable required extensions ---
em = omni.kit.app.get_app().get_extension_manager()
for ext in ("isaacsim.replicator.agent.core", "isaacsim.replicator.agent", "omni.anim.people"):
    try:
        em.set_extension_enabled_immediate(ext, True)
    except Exception:
        pass

# --- Stage setup ---
rep.orchestrator.set_capture_on_play(False)
if config.get("clear_previous_semantics", False):
    stage = get_current_stage()
    scene_based_sdg_utils.remove_previous_semantics(stage)

# --- Spawn forklift and pallet ---
forklift_cfg = config["forklift"]
pallet_cfg = config["pallet"]

forklift_prim = prims.create_prim(
    prim_path="/World/Forklift",
    position=(random.uniform(-20, -2), random.uniform(-1, 3), 0),
    orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
    usd_path=assets_root_path + forklift_cfg["url"],
    semantic_label=forklift_cfg["class"],
)

forklift_tf = omni.usd.get_world_transform_matrix(forklift_prim)
pallet_offset_tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, random.uniform(-1.2, -1.8), 0))
pallet_pos_gf = (pallet_offset_tf * forklift_tf).ExtractTranslation()
forklift_quat_xyzw = (forklift_tf.ExtractRotationQuat().GetReal(), *forklift_tf.ExtractRotationQuat().GetImaginary())

pallet_prim = prims.create_prim(
    prim_path="/World/Pallet",
    position=pallet_pos_gf,
    orientation=forklift_quat_xyzw,
    usd_path=assets_root_path + pallet_cfg["url"],
    semantic_label=pallet_cfg["class"],
)

# --- Spawn humans from config ---
people_root = "/World/People"
prims.create_prim(people_root)

for human in config.get("humans", []):
    offset = human.get("offset", [0, 0, 0])
    pos = (pallet_pos_gf[0] + offset[0], pallet_pos_gf[1] + offset[1], offset[2])
    prims.create_prim(
        prim_path=f"{people_root}/{human['name']}",
        position=pos,
        orientation=(0, 0, 0, 1),
        usd_path=human["usd_path"],
        semantic_label=human.get("class", "person"),
    )

# --- Randomization utilities ---
scene_based_sdg_utils.register_scatter_boxes(pallet_prim, assets_root_path, config)
scene_based_sdg_utils.register_cone_placement(forklift_prim, assets_root_path, config)
scene_based_sdg_utils.register_lights_placement(forklift_prim, pallet_prim)

# --- Jackal robot + camera ---
jackal_cfg = config["robots"]["jackal"]
jackal_prim = prims.create_prim(
    prim_path="/World/Robots/Jackal",
    position=(pallet_pos_gf[0] + 2.0, pallet_pos_gf[1] + 1.0, 0.0),
    orientation=(0, 0, 0, 1),
    usd_path=assets_root_path + jackal_cfg["url"],
    semantic_label=jackal_cfg["class"],
)

jackal_cam = rep.create.camera(
    name="JackalCam",
    parent="/World/Robots/Jackal",
    position=(0.0, 0.0, 0.6),
    rotation=(0.0, 0.0, 0.0),
    clipping_range=(0.1, 100000.0),
    focal_length=16.0,
    focus_distance=400.0,
)

resolution = config.get("resolution", [512, 512])
jackal_rp = rep.create.render_product(jackal_cam, resolution, name="JackalView")

# --- Writer setup ---
writer_type = config.get("writer", "BasicWriter")
writer = rep.WriterRegistry.get(writer_type)
writer.initialize(**config["writer_config"])
writer.attach([jackal_rp])

# --- Character Randomizer ---
try:
    from isaacsim.replicator.agent.core.character_randomizer import CharacterRandomizer
except ImportError:
    from isaacsim.replicator.agent.core import CharacterRandomizer

rand = CharacterRandomizer(global_seed=1234)
fps = 30.0
active_seconds = float(config.get("num_frames", 0)) / fps
asyncio.get_event_loop().run_until_complete(
    rand.generate_character_commands(
        global_seed=4321, duration=active_seconds, agent_count=len(config["humans"]), navigation_area=None
    )
)

# --- Simulation + capture loop ---
rt_subframes = config.get("rt_subframes", -1)
num_frames = config.get("num_frames", 0)
dt = 1.0 / fps
print(f"[scene_based_sdg] Capturing {num_frames} frames (dt={dt})")

for i in range(num_frames):
    print(f"  Frame {i}")
    rep.orchestrator.step(delta_time=dt, rt_subframes=rt_subframes)

rep.orchestrator.wait_until_complete()
writer.detach()
jackal_rp.destroy()
simulation_app.close()