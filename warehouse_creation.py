# warehouse_creation.py
# Isaac Sim 5.x — minimal scene creation with robust create_prim calls

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os, random
from typing import Dict, Tuple, List

import omni.usd
from omni.isaac.core import World

# Import create_prim from 5.0 path, fall back to older
try:
    from isaacsim.core.utils.prims import create_prim  # Isaac Sim 5.0+
except ImportError:
    from omni.isaac.core.utils.prims import create_prim  # 4.x fallback

from pxr import UsdGeom

# ---------- Your asset roots ----------
ROOT = "/home/deepak/Downloads/isaac_sim_assets_4_5/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5/Isaac"
WAREHOUSE_USD_PATH = f"{ROOT}/Environments/Simple_Warehouse/full_warehouse.usd"
ROBOT_USD_PATH     = f"{ROOT}/Robots/Clearpath/Jackal/jackal.usd"
DRONE_USD_PATH     = f"{ROOT}/Robots/Iris/iris.usd"
HUMAN_USD_PATH     = f"{ROOT}/People/Characters/original_male_adult_construction_03/male_adult_construction_03.usd"
HUMAN1_USD_PATH    = f"{ROOT}/People/Characters/female_adult_police_02/female_adult_police_02.usd"
HUMAN2_USD_PATH    = f"{ROOT}/People/Characters/original_female_adult_police_02/female_adult_police_02.usd"
FORKLIFT_USD_PATH  = f"{ROOT}/Robots/Forklift/forklift_b.usd"

USD_BY_TYPE: Dict[str, str] = {
    "human":    HUMAN_USD_PATH,
    "human1":   HUMAN1_USD_PATH,
    "human2":   HUMAN2_USD_PATH,
    "forklift": FORKLIFT_USD_PATH,
    "robot":    ROBOT_USD_PATH,
    "drone":    DRONE_USD_PATH,
}

# ---------- Stage / World ----------
world = World()
ctx = omni.usd.get_context()
if ctx.get_stage() is None:
    ctx.new_stage()
stage = ctx.get_stage()

UsdGeom.Xform.Define(stage, "/World")
world.scene.add_default_ground_plane()

# ---------- Helpers ----------
def _safe_create_prim(*, prim_path: str, prim_type: str, usd_path: str, position: Tuple[float,float,float]):
    """
    Call create_prim with keyword args. Some Isaac Sim builds expect 'translation'
    instead of 'position'. We try 'position' first, then fall back to 'translation'.
    """
    try:
        return create_prim(
            prim_path=prim_path,
            prim_type=prim_type,
            usd_path=usd_path,       # <<< KEYWORDED (fixes your error)
            position=position,       # first attempt
        )
    except TypeError:
        # Retry with 'translation' keyword (older signatures)
        return create_prim(
            prim_path=prim_path,
            prim_type=prim_type,
            usd_path=usd_path,
            translation=position,    # fallback
        )

# Random placement bounds (meters)
BOUNDS_X = (-15.0, 15.0)
BOUNDS_Y = (-15.0, 15.0)

def _rand(lo: float, hi: float) -> float:
    import random
    return random.uniform(lo, hi)

def sample_position(kind: str) -> Tuple[float, float, float]:
    x = _rand(*BOUNDS_X)
    y = _rand(*BOUNDS_Y)
    z = _rand(2.0, 3.5) if kind == "drone" else 0.0
    return (x, y, z)

# ---------- Spawners ----------
def create_env(prim_name: str = "Warehouse"):
    print(f"[ENV] Spawning: {prim_name}")
    _safe_create_prim(
        prim_path=f"/World/{prim_name}",
        prim_type="Xform",
        usd_path=WAREHOUSE_USD_PATH,
        position=(0.0, 0.0, 0.0),
    )

def create_objects(obj_type: str, base_name: str, number: int) -> List[str]:
    usd = USD_BY_TYPE.get(obj_type.lower())
    if not usd:
        raise ValueError(f"Unsupported obj_type: {obj_type}")
    spawned = []
    for i in range(number):
        name = f"{base_name}_{i:03d}" if number > 1 else base_name
        pos = sample_position(obj_type.lower())
        prim_path = f"/World/{name}"
        print(f"[SPAWN] {obj_type:<8} -> {prim_path} @ {tuple(round(v,2) for v in pos)}  usd={os.path.basename(usd)}")
        _safe_create_prim(
            prim_path=prim_path,
            prim_type="Xform",
            usd_path=usd,
            position=pos,
        )
        spawned.append(prim_path)
    return spawned

# ---------- Build scene ----------
create_env("warehouse")
create_objects("human",    "HumanA",   number=2)
create_objects("human1",   "HumanB",   number=2)
create_objects("human2",   "HumanC",   number=1)
create_objects("drone",    "Drone",    number=1)
create_objects("forklift", "Forklift", number=1)
create_objects("robot",    "Robot",    number=1)

# ---------- Run ----------
world.reset()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()