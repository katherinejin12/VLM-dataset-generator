# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Standalone script to run an actor sdg job

Example:
    ./python.sh tools/actor_sdg/sdg_scheduler.py -c tools/actor_sdg/default_config.yaml

Optional params:
    --sensor_placment_file sensor_placements.json
    --crash_report_path /home/xxx
    --no_random_commands
    --debug_print
    --save_usd
"""

import argparse
import asyncio
import os
import sys

import numpy as np
from isaacsim import SimulationApp

BASE_EXP_PATH = os.path.join(os.environ["EXP_PATH"], "isaacsim.exp.action_and_event_data_generation.base.kit")
APP_CONFIG = {"renderer": "RayTracedLighting", "headless": True, "width": 1920, "height": 1080}




class ActorSDG:
    def __init__(
        self, sim_app, config_file_path, camera_file_path, crash_report_path, no_random_commands, debug_print, save_usd
    ):
        self._sim_app = sim_app
        # Inputs
        self.config_file_path = config_file_path
        self.camera_file_path = camera_file_path
        self.crash_report_path = crash_report_path
        self.no_random_commands = no_random_commands
        self.debug_print = debug_print
        self.save_usd = save_usd

        self.output_path = None
        self.camera_placements_json = None
        self._sim_manager = None
        self._setup_sim_sub = None
        self._setup_sim_succeed = False
        self._dg_sub = None
        self._settings = None
        self._traj_running = False  # <-- needed for the logger loop

    async def run(self):
        # Enable all required extensions
        self._enable_extensions()
        await self._sim_app.app.next_update_async()
        # Set up global settings
        self._set_simulation_settings()
        await self._sim_app.app.next_update_async()

        # Init SimulatonManager
        from isaacsim.replicator.agent.core.simulation import SimulationManager

        self._sim_manager = SimulationManager()

        try:
            can_load_config = self._sim_manager.load_config_file(self.config_file_path)
            if not can_load_config:
                return False

            seed_prop = self._sim_manager.get_config_file_property("global", "seed")
            seed_val = None

            if seed_prop:
                try:
                    seed_val = seed_prop.get_value()
                except Exception:
                    seed_val = None

            if seed_val is None:
                import time, random
                random.seed(time.time())
                seed_val = random.randrange(0, 2147483647)
                if seed_prop:
                    seed_prop.set_value(int(seed_val))

            # ====Character count randomization macro =====
            import random, time
            CHAR_MIN = 5
            CHAR_MAX = 15
            random.seed(time.time())
            #char_num_rand = random.randrange(0, 2147483647)
            char_num_rand = 935590915
            random.seed(char_num_rand)
            RANDOM_CHARACTER_COUNT = random.randint(CHAR_MIN, CHAR_MAX)
            print(f"[characterMArco] Randomized character count for this run: {RANDOM_CHARACTER_COUNT}")

            # Override character count in config
            prop = self._sim_manager.get_config_file_property("character", "num")
            if prop:
                prop.set_value(RANDOM_CHARACTER_COUNT)
                print(f"[characterMarco] set character_num = {RANDOM_CHARACTER_COUNT}")
            else:
                print(f"[characterMarco] No character_num")

            writer_selection = self._sim_manager.get_config_file_property_group("replicator", "writer_selection")
            params = writer_selection.content_prop.get_value()
            base_out = params.get("output_dir") or os.getcwd()
            seed_out = os.path.join(base_out, f"seed_{int(seed_val)}_numseed_{int(char_num_rand)}_v")
            params["output_dir"] = seed_out
            self.output_path = params["output_dir"]
            os.makedirs(self.output_path, exist_ok=True)
            randData_path = os.path.join(self.output_path, "randData.txt")
            with open(randData_path, "w", encoding="utf-8") as f:
                f.write(f"seed: {seed_val}\n")
                f.write(f"character_num: {RANDOM_CHARACTER_COUNT}\n")
                f.write(f"character_num_seed: {char_num_rand}\n")
            print(f"[randData] saved randData to {randData_path}")
            

            # ---create per-run blank command files and set them in the config---
            robot_cmd_path = os.path.join(self.output_path, "robot_command.txt")
            char_cmd_path = os.path.join(self.output_path, "character_command.txt")

            open(char_cmd_path, "w").close()   # <-- add ()
            open(robot_cmd_path, "w").close()  # <-- add ()

            char_prop = self._sim_manager.get_config_file_property("character", "command_file")
            if char_prop:
                char_prop.set_value(char_cmd_path)
                print(f"[configpath] Set character.command_file = {char_cmd_path}")

            robot_prop = self._sim_manager.get_config_file_property("robot", "command_file")
            if robot_prop:
                robot_prop.set_value(robot_cmd_path)
                print(f"[configpath] Set robot.command_file = {robot_cmd_path}")

            # Set up sim
            await self._setup_sim()

            import omni.usd
            from pxr import UsdGeom, Sdf, Usd

            stage = omni.usd.get_context().get_stage()

            # [Optional] Camera placement
            if self.camera_file_path:
                self._do_camera_placement()

            # [Optional] Generate random commands
            if not self.no_random_commands:
                await self._gen_random_commands()

            # ---- start per-frame position logging ----
            self._traj_running = True
            traj_task = asyncio.create_task(self._record_trajectories(self.output_path))

            # Wait for data generation callback
            await self._sim_manager.run_data_generation_async(will_wait_until_complete=True)

            # ---- stop logging ----
            self._traj_running = False
            await traj_task
            print(f"[Traj] wrote {self.output_path}")
            return True

        except Exception as e:
            import carb
            carb.log_error(f"Failed to load config file {e}")
            return False

    def _enable_extensions(self):
        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()

        ext_manager.set_extension_enabled_immediate("omni.kit.viewport.window", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.manipulator.prim", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.property.usd", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.scripting", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.timeline", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.graph.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.retarget.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.navigation.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.navigation.meshtools", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.people", True)
        ext_manager.set_extension_enabled_immediate("isaacsim.replicator.agent.core", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.mesh.raycast", True)
        ext_manager.set_extension_enabled_immediate("omni.physx.graph", True)  # For Conveyor Belt

    def _set_simulation_settings(self):
        import carb
        import omni.replicator.core as rep

        rep.settings.carb_settings("/omni/replicator/backend/writeThreads", 16)
        self._settings = carb.settings.get_settings()
        self._settings.set("/rtx/rtxsensor/coordinateFrameQuaternion", "0.5,-0.5,-0.5,-0.5")
        self._settings.set("/app/scripting/ignoreWarningDialog", True)
        self._settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
        self._settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/aim_camera_to_character", True)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/min_camera_distance", 6.5)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/max_camera_distance", 14.5)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/max_camera_look_down_angle", 60)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/min_camera_look_down_angle", 0)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/min_camera_height", 2)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/max_camera_height", 3)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/character_focus_height", 0.7)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/frame_write_interval", 1)
        self._settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)
        self._settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)
        self._settings.set("/log/level", "info")
        self._settings.set("/log/channels/omni.replicator.core", "info")
        self._settings.set("/log/channels/isaacsim.replicator.character.core", "info")
        self._settings.set("/log/channels/omni.usd", "error")
        self._settings.set("/log/channels/omni.hydra", "error")
        self._settings.set("/log/channels/omni.kit.menu.*", "error")
        self._settings.set("/log/channels/omni.kit.property.*", "error")
        self._settings.set("/log/channels/omni.anim.graph.*", "error")
        self._settings.set("/exts/isaacsim.replicator.agent/debug_print", self.debug_print)
        self._settings.set("/crashreporter/enabled", True)
        if self.crash_report_path:
            self._settings.set("/crashreporter/dumpDir", self.crash_report_path)
            path = self._settings.get("/crashreporter/dumpDir")
            print(f"Using carsh reporter path: {path}")
    '''
    async def _record_trajectories(self, out_dir: str):
        import csv, os, hashlib, omni.usd, omni.timeline
        from pxr import Usd, UsdGeom, Gf, UsdPhysics, Sdf
        import math
        import numpy as np
        from isaacsim.core.prims import XFormPrim, Articulation, RigidPrim
        from isaacsim.core.utils.numpy.rotations import quats_to_euler_angles

        stage = omni.usd.get_context().get_stage()

        try:
            from isaacsim.replicator.agent.core.stage_util import CharacterUtil
            char_prims = CharacterUtil.get_characters_in_stage()
        except Exception:
            # fallback: grab characters under this root if CharacterUtil isn't available
            char_prims = [p for p in stage.Traverse() if p.GetPath().pathString.startswith("/World/People")]

        try:
            from isaacsim.replicator.agent.core.stage_util import RobotUtil
            robot_prims = RobotUtil.get_robots_in_stage()
        except Exception:
            robot_prims = []
            robot_root = stage.GetPrimAtPath("/World/Robots")
            if robot_root and robot_root.IsValid():
                robot_prims = list(robot_root.GetChildren())

        targets = list(dict.fromkeys(char_prims + robot_prims))
        target_paths = [p.GetPath().pathString for p in targets]
        num_targets = len(targets)
        index_of = {p: i for i, p in enumerate(target_paths)}

        # robot-only maps
        robot_paths = [p.GetPath().pathString for p in robot_prims]

        def path_to_filename(pstr: str) -> str:
            raw = pstr.lstrip('/')
            safe = raw.replace('/', '_')
            if len(safe) > 180:
                h = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]
                safe = safe[:160] + '_' + h
            return safe

        base_traj = os.path.join(out_dir, "trajectories")
        char_dir = os.path.join(base_traj, "characters")
        robot_dir = os.path.join(base_traj, "robots")
        os.makedirs(char_dir, exist_ok=True)
        os.makedirs(robot_dir, exist_ok=True)

        # Pose view (Fabric mode)
        pose_view = XFormPrim(prim_paths_expr=target_paths, reset_xform_properties=False, usd=True)
        print("pose view class =", type(pose_view))
        def list_rigid_children(root_path: str):
            """Return list of child paths that have RigidBodyAPI."""
            prim = stage.GetPrimAtPath(root_path)
            if not (prim and prim.IsValid()):
                return []
            out = []
            q = list(prim.GetChildren())
            while q:
                p = q.pop(0)
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    out.append(p.GetPath().pathString)
                q.extend(p.GetChildren())
            return out

        def pick_primary_rigid(rigid_paths: list[str]) -> str | None:
            """Heuristic: prefer base_link / chassis; else first."""
            if not rigid_paths:
                return None
            names = [rp.split('/')[-1].lower() for rp in rigid_paths]
            for key in ("base_link", "chassis", "base"):
                for i, nm in enumerate(names):
                    if key in nm:
                        return rigid_paths[i]
            return rigid_paths[0]
        # Velocity views for robot
        def find_first_articulation_child(root_path: str) -> str | None:
            prim = stage.GetPrimAtPath(root_path)
            if not (prim and prim.IsValid()):
                return None
            q = list(prim.GetChildren())
            while q:
                p = q.pop(0)
                api = UsdPhysics.ArticulationRootAPI.Get(stage, p.GetPath())
                if api and api.IsValid():
                    return api.GetPrim().GetPath().pathString
                q.extend(p.GetChildren())
            return None

    # --- Build articulation maps (store both API and root path)
        art_api_by_original: dict[str, UsdPhysics.ArticulationRootAPI] = {}
        art_root_by_original: dict[str, str] = {}

        robot_paths = [p.GetPath().pathString for p in robot_prims]
        for rp in robot_paths:
            prim = stage.GetPrimAtPath(rp)
            api = UsdPhysics.ArticulationRootAPI.Get(stage, prim.GetPath()) if (prim and prim.IsValid()) else None
            if api and api.IsValid():
                ar_path = api.GetPrim().GetPath().pathString
                art_api_by_original[rp] = api
                art_root_by_original[rp] = ar_path
                print(f"[DEBUG] Registered articulation: original={rp} root={ar_path}")
            else:
                ar = find_first_articulation_child(rp)
                if ar:
                    api2 = UsdPhysics.ArticulationRootAPI.Get(stage, Usd.Prim.GetStage(stage).GetPrimAtPath(ar).GetPath())
                    art_api_by_original[rp] = api2  # may be None if not valid, but ar implies valid
                    art_root_by_original[rp] = ar
                    print(f"[DEBUG] Registered child articulation: original={rp} root={ar}")
                else:
                    print(f"[DEBUG] No articulation under: {rp}")

    # Dedup articulation roots (values) for the Articulation view
        seen = set()
        art_paths = []
        for ar in art_root_by_original.values():
            if ar not in seen:
                seen.add(ar)
                art_paths.append(ar)

        # After you build articulation maps (art_root_by_original)
        rigid_primary_by_original: dict[str, str] = {}   # original robot root -> chosen rigid body path
        rigid_all_paths: list[str] = []                  # flat list for RigidPrim view (dedup later)

        for rp in robot_paths:
            if rp in art_root_by_original:
                continue  # articulation will provide velocities
            rbs = list_rigid_children(rp)
            if rbs:
                primary = pick_primary_rigid(rbs)
                rigid_primary_by_original[rp] = primary
                rigid_all_paths.extend(rbs)
                print(f"[DEBUG] Rigid fallback for {rp}: primary={primary}, total_links={len(rbs)}")
            else:
                print(f"[DEBUG] No rigid bodies under: {rp} (will use finite-difference fallback)")
        seen_rig = set()
        rigid_paths = []
        for rp in rigid_all_paths:
            if rp not in seen_rig:
                seen_rig.add(rp)
                rigid_paths.append(rp)

        bots_rig = RigidPrim(
            prim_paths_expr=rigid_paths,
            reset_xform_properties=False,
            usd=False
        )if rigid_paths else None
        #def _has_rigid_api(path: str) -> bool:
            #prim = stage.GetPrimAtPath(path)
            #return bool(prim and prim.IsValid() and UsdPhysics.RigidBodyAPI(prim))

        #art_paths   = [p for p in robot_paths if _has_art_root_api(p)]
        #rigid_paths = [p for p in robot_paths if _has_rigid_api(p)]

        bots_art = Articulation(prim_paths_expr=art_paths,   reset_xform_properties=False, usd=False) if art_paths   else None
        #bots_rig = RigidPrim(    prim_paths_expr=rigid_paths, reset_xform_properties=False, usd=False) if rigid_paths else None

        # Writers (append velocity columns at the end)
        header = [
            "time_s", "frame", "prim_path",
            "tx", "ty", "tz", "roll", "pitch", "yaw",
            "tx_usd", "ty_usd", "tz_usd", "roll_usd", "pitch_usd", "yaw_usd",
            "lin_art_x", "lin_art_y", "lin_art_z", "ang_art_x", "ang_art_y", "ang_art_z",
            #"lin_rig_x", "lin_rig_y", "lin_rig_z", "ang_rig_x", "ang_rig_y", "ang_rig_z",
        ]

        writers = {}
        files = {}
        for prim in char_prims:
            pstr = prim.GetPath().pathString
            base = path_to_filename(pstr)
            csv_path = os.path.join(char_dir, f"{base}_trajectory.csv")
            f = open(csv_path, "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(header)
            writers[pstr] = w
            files[pstr] = f

        for prim in robot_prims:
            pstr = prim.GetPath().pathString
            base = path_to_filename(pstr)
            csv_path = os.path.join(robot_dir, f"{base}_trajectory.csv")
            f = open(csv_path, "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(header)
            writers[pstr] = w
            files[pstr] = f

        tps = stage.GetTimeCodesPerSecond() or 60.0
        tl = omni.timeline.get_timeline_interface()

        frame = 0
        try:
            sim_len_prop = self._sim_manager.get_config_file_property("global", "simulation_length")
            simulation_length = int(sim_len_prop.get_value())
        except Exception:
            simulation_length = 1800
        max_valid_frame = simulation_length - 1

        # helpful maps for velocity indexing
        idx_art_of = {p: i for i, p in enumerate(art_paths)}
        #idx_rig_of = {p: i for i, p in enumerate(rigid_paths)}

        while getattr(self, "_traj_running", False) and not self._sim_app.is_exiting():
            await self._sim_app.app.next_update_async()
            if frame > max_valid_frame:
                break

            t = float(tl.get_current_time())
            tc = Usd.TimeCode(t)
            xcache = UsdGeom.XformCache(tc)

            # Poses from Fabric (your conversion stays the same)
            positions, orientations = pose_view.get_world_poses(usd=True)
            pos_np = positions.numpy() if hasattr(positions, "numpy") else positions
            orn_np = orientations.numpy() if hasattr(orientations, "numpy") else orientations

            # EXACTLY your conversion (do not change)
            eulers = quats_to_euler_angles(orn_np, degrees=True, extrinsic=False)

           
            # Velocities (live / Fabric-backed)
            art_vel_by_root = {}
            if bots_art is not None and art_paths:
                arr = bots_art.get_velocities()  # (N,6)
                arr = arr.numpy() if hasattr(arr, "numpy") else arr
                for i, root in enumerate(art_paths):
                    art_vel_by_root[root] = tuple(map(float, arr[i]))
            
            rig_vel_by_path = {}
            if bots_rig is not None and rigid_paths:
                rv = bots_rig.get_velocities()  # (N_rig, 6)
                rv = rv.numpy() if hasattr(rv, "numpy") else rv
                for i, path in enumerate(rigid_paths):
                    rig_vel_by_path[path] = tuple(map(float, rv[i]))

            #art_lin_ang = None
            #if bots_art is not None:
                #art_lin_ang = bots_art.get_velocities()  # (N_art, 6): [lin(3), ang(3)]
                #if hasattr(art_lin_ang, "numpy"):
                    #art_lin_ang = art_lin_ang.numpy()

            #rig_lin_ang = None
            #if bots_rig is not None:
                #rig_lin_ang = bots_rig.get_velocities()  # (N_rig, 6): [lin(3), ang(3)]
                #if hasattr(rig_lin_ang, "numpy"):
                    #rig_lin_ang = rig_lin_ang.numpy()

            # Write rows
            for prim in targets:
                if not prim.IsValid():
                    continue
                pstr = prim.GetPath().pathString
                i = index_of[pstr]

                tx, ty, tz = [float(v) for v in pos_np[i]]
                roll, pitch, yaw = [float(v) for v in eulers[i]]

                M = xcache.GetLocalToWorldTransform(prim)
                tr = M.ExtractTranslation()
                #euler rotation
            
                rot = M.ExtractRotation()
                #decompose into XYZ euler angles
                roll_deg, pitch_deg, yaw_deg = rot.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())


                # default zeros for velocities
                avx = avy = avz = awx = awy = awz = 0.0
                #rvx = rvy = rvz = rwx = rwy = rwz = 0.0
                art_root = art_root_by_original.get(pstr)
                if art_root:
                    vel = art_vel_by_root.get(art_root)
                    if vel:
                        avx, avy, avz, awx, awy, awz = vel
                else:
                    primary_rigid = rigid_primary_by_original.get(pstr)
                    if primary_rigid:
                        vel = rig_vel_by_path.get(primary_rigid)
                        if vel:
                            avx, avy, avz, awx, awy, awz = vel
                # articulation velocities (if applicable)
                #if pstr in idx_art_of and art_lin_ang is not None:
                    #j = idx_art_of[pstr]
                    #avx, avy, avz, awx, awy, awz = map(float, art_lin_ang[j])

                # rigid velocities (if applicable)
                #if pstr in idx_rig_of and rig_lin_ang is not None:
                    #k = idx_rig_of[pstr]
                    #rvx, rvy, rvz, rwx, rwy, rwz = map(float, rig_lin_ang[k])

                writers[pstr].writerow([
                    f"{t:.6f}", frame, pstr,
                    f"{tx:.6f}", f"{ty:.6f}", f"{tz:.6f}",
                    f"{roll:.3f}", f"{pitch:.3f}", f"{yaw:.3f}",
                    f"{tr[0]:.6f}", f"{tr[1]:.6f}", f"{tr[2]:.6f}",
                    f"{roll_deg:.3f}", f"{pitch_deg:.3f}", f"{yaw_deg:.3f}",
                    f"{avx:.6f}", f"{avy:.6f}", f"{avz:.6f}",
                    f"{awx:.6f}", f"{awy:.6f}", f"{awz:.6f}",
                    #f"{rvx:.6f}", f"{rvy:.6f}", f"{rvz:.6f}",
                    #f"{rwx:.6f}", f"{rwy:.6f}", f"{rwz:.6f}",
                ])

            frame += 1

        # Close all files
        for f in files.values():
            f.close()
    '''
    async def _record_trajectories(self, out_dir: str):
        import csv, os, hashlib, omni.usd, omni.timeline
        from pxr import Usd, UsdGeom, Gf, UsdPhysics, Sdf
        import math
        import numpy as np
        from isaacsim.core.prims import XFormPrim, Articulation, RigidPrim
        from isaacsim.core.utils.numpy.rotations import quats_to_euler_angles

        stage = omni.usd.get_context().get_stage()

        try:
            from isaacsim.replicator.agent.core.stage_util import CharacterUtil
            char_prims = CharacterUtil.get_characters_in_stage()
        except Exception:
            # fallback: grab characters under this root if CharacterUtil isn't available
            char_prims = [p for p in stage.Traverse() if p.GetPath().pathString.startswith("/World/People")]

        try:
            from isaacsim.replicator.agent.core.stage_util import RobotUtil
            robot_prims = RobotUtil.get_robots_in_stage()
        except Exception:
            robot_prims = []
            robot_root = stage.GetPrimAtPath("/World/Robots")
            if robot_root and robot_root.IsValid():
                robot_prims = list(robot_root.GetChildren())

        targets = list(dict.fromkeys(char_prims + robot_prims))
        target_paths = [p.GetPath().pathString for p in targets]
        num_targets = len(targets)
        index_of = {p: i for i, p in enumerate(target_paths)}

        # robot-only maps
        robot_paths = [p.GetPath().pathString for p in robot_prims]

        def path_to_filename(pstr: str) -> str:
            raw = pstr.lstrip('/')
            safe = raw.replace('/', '_')
            if len(safe) > 180:
                h = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]
                safe = safe[:160] + '_' + h
            return safe

        base_traj = os.path.join(out_dir, "trajectories")
        char_dir = os.path.join(base_traj, "characters")
        robot_dir = os.path.join(base_traj, "robots")
        os.makedirs(char_dir, exist_ok=True)
        os.makedirs(robot_dir, exist_ok=True)

        # Pose view (Fabric mode)
        pose_view = XFormPrim(prim_paths_expr=target_paths, reset_xform_properties=False, usd=True)
        print("pose view class =", type(pose_view))
        def list_rigid_children(root_path: str):
            """Return list of child paths that have RigidBodyAPI."""
            prim = stage.GetPrimAtPath(root_path)
            if not (prim and prim.IsValid()):
                return []
            out = []
            q = list(prim.GetChildren())
            while q:
                p = q.pop(0)
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    out.append(p.GetPath().pathString)
                q.extend(p.GetChildren())
            return out

        def pick_primary_rigid(rigid_paths: list[str]) -> str | None:
            """Heuristic: prefer base_link / chassis; else first."""
            if not rigid_paths:
                return None
            names = [rp.split('/')[-1].lower() for rp in rigid_paths]
            for key in ("base_link", "chassis", "base"):
                for i, nm in enumerate(names):
                    if key in nm:
                        return rigid_paths[i]
            return rigid_paths[0]
        # Velocity views for robot
        def find_first_articulation_child(root_path: str) -> str | None:
            prim = stage.GetPrimAtPath(root_path)
            if not (prim and prim.IsValid()):
                return None
            q = list(prim.GetChildren())
            while q:
                p = q.pop(0)
                api = UsdPhysics.ArticulationRootAPI.Get(stage, p.GetPath())
                if api and api.IsValid():
                    return api.GetPrim().GetPath().pathString
                q.extend(p.GetChildren())
            return None

    # --- Build articulation maps (store both API and root path)
        art_api_by_original: dict[str, UsdPhysics.ArticulationRootAPI] = {}
        art_root_by_original: dict[str, str] = {}

        robot_paths = [p.GetPath().pathString for p in robot_prims]
        for rp in robot_paths:
            prim = stage.GetPrimAtPath(rp)
            api = UsdPhysics.ArticulationRootAPI.Get(stage, prim.GetPath()) if (prim and prim.IsValid()) else None
            if api and api.IsValid():
                ar_path = api.GetPrim().GetPath().pathString
                art_api_by_original[rp] = api
                art_root_by_original[rp] = ar_path
                print(f"[DEBUG] Registered articulation: original={rp} root={ar_path}")
            else:
                ar = find_first_articulation_child(rp)
                if ar:
                    api2 = UsdPhysics.ArticulationRootAPI.Get(stage, Usd.Prim.GetStage(stage).GetPrimAtPath(ar).GetPath())
                    art_api_by_original[rp] = api2  # may be None if not valid, but ar implies valid
                    art_root_by_original[rp] = ar
                    print(f"[DEBUG] Registered child articulation: original={rp} root={ar}")
                else:
                    print(f"[DEBUG] No articulation under: {rp}")

    # Dedup articulation roots (values) for the Articulation view
        seen = set()
        art_paths = []
        for ar in art_root_by_original.values():
            if ar not in seen:
                seen.add(ar)
                art_paths.append(ar)

        # After you build articulation maps (art_root_by_original)
        rigid_primary_by_original: dict[str, str] = {}  # original robot root -> chosen rigid body path
        rigid_all_paths: list[str] = []                # flat list for RigidPrim view (dedup later)

        for rp in robot_paths:
            if rp in art_root_by_original:
                continue  # articulation will provide velocities
            rbs = list_rigid_children(rp)
            if rbs:
                primary = pick_primary_rigid(rbs)
                rigid_primary_by_original[rp] = primary
                rigid_all_paths.extend(rbs)
                print(f"[DEBUG] Rigid fallback for {rp}: primary={primary}, total_links={len(rbs)}")
            else:
                print(f"[DEBUG] No rigid bodies under: {rp} (will use finite-difference fallback)")
        seen_rig = set()
        rigid_paths = []
        for rp in rigid_all_paths:
            if rp not in seen_rig:
                seen_rig.add(rp)
                rigid_paths.append(rp)

        bots_rig = RigidPrim(
            prim_paths_expr=rigid_paths,
            reset_xform_properties=False,
            usd=False
        )if rigid_paths else None

        bots_art = Articulation(prim_paths_expr=art_paths, reset_xform_properties=False, usd=False) if art_paths else None
    
        # Writers (append velocity columns at the end)
        header = [
        "time_s", "frame", "prim_path",
        "tx", "ty", "tz", "roll", "pitch", "yaw",
        "tx_usd", "ty_usd", "tz_usd", "roll_usd", "pitch_usd", "yaw_usd",
        "lin_art_x", "lin_art_y", "lin_art_z", "ang_art_x", "ang_art_y", "ang_art_z",
        #"lin_rig_x", "lin_rig_y", "lin_rig_z", "ang_rig_x", "ang_rig_y", "ang_rig_z",
        ]

        writers = {}
        files = {}
        for prim in char_prims:
            pstr = prim.GetPath().pathString
            base = path_to_filename(pstr)
            csv_path = os.path.join(char_dir, f"{base}_trajectory.csv")
            f = open(csv_path, "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(header)
            writers[pstr] = w
            files[pstr] = f

        for prim in robot_prims:
            pstr = prim.GetPath().pathString
            base = path_to_filename(pstr)
            csv_path = os.path.join(robot_dir, f"{base}_trajectory.csv")
            f = open(csv_path, "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(header)
            writers[pstr] = w
            files[pstr] = f

        tps = stage.GetTimeCodesPerSecond() or 60.0
        tl = omni.timeline.get_timeline_interface()

        frame = 0
        try:
            sim_len_prop = self._sim_manager.get_config_file_property("global", "simulation_length")
            simulation_length = int(sim_len_prop.get_value())
        except Exception:
            simulation_length = 1800
        max_valid_frame = simulation_length - 1

        # helpful maps for velocity indexing
        idx_art_of = {p: i for i, p in enumerate(art_paths)}
    #idx_rig_of = {p: i for i, p in enumerate(rigid_paths)}

        while getattr(self, "_traj_running", False) and not self._sim_app.is_exiting():
            await self._sim_app.app.next_update_async()
            if frame > max_valid_frame:
                break

        # 🚀 THE FIX: Force Articulation and Rigid Prim buffers to update with new physics data
            if bots_art is not None:
            # Articulation update is necessary for Articulation.get_velocities() to work
                bots_art.update_buffers(0) 
            if bots_rig is not None:
            # RigidPrim update is necessary for RigidPrim.get_velocities() to work
                bots_rig.update_buffers(0)

            t = float(tl.get_current_time())
            tc = Usd.TimeCode(t)
            xcache = UsdGeom.XformCache(tc)

        # Poses from Fabric (your conversion stays the same)
            positions, orientations = pose_view.get_world_poses(usd=True)
            pos_np = positions.numpy() if hasattr(positions, "numpy") else positions
            orn_np = orientations.numpy() if hasattr(orientations, "numpy") else orientations

        # EXACTLY your conversion (do not change)
            eulers = quats_to_euler_angles(orn_np, degrees=True, extrinsic=False)

        
        # Velocities (live / Fabric-backed)
            art_vel_by_root = {}
            if bots_art is not None and art_paths:
                arr = bots_art.get_velocities()  # (N,6) - NOW CORRECTLY POPULATED
                arr = arr.numpy() if hasattr(arr, "numpy") else arr
                for i, root in enumerate(art_paths):
                    art_vel_by_root[root] = tuple(map(float, arr[i]))
            
            rig_vel_by_path = {}
            if bots_rig is not None and rigid_paths:
                rv = bots_rig.get_velocities()  # (N_rig, 6) - NOW CORRECTLY POPULATED
                rv = rv.numpy() if hasattr(rv, "numpy") else rv
                for i, path in enumerate(rigid_paths):
                    rig_vel_by_path[path] = tuple(map(float, rv[i]))

        # Write rows
            for prim in targets:
                if not prim.IsValid():
                    continue
                pstr = prim.GetPath().pathString
                i = index_of[pstr]

                tx, ty, tz = [float(v) for v in pos_np[i]]
                roll, pitch, yaw = [float(v) for v in eulers[i]]

                M = xcache.GetLocalToWorldTransform(prim)
                tr = M.ExtractTranslation()
            #euler rotation
            
                rot = M.ExtractRotation()
            #decompose into XYZ euler angles
                roll_deg, pitch_deg, yaw_deg = rot.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())


            # default zeros for velocities
                avx = avy = avz = awx = awy = awz = 0.0
            
                art_root = art_root_by_original.get(pstr)
                if art_root:
                    vel = art_vel_by_root.get(art_root)
                    if vel:
                        avx, avy, avz, awx, awy, awz = vel
                else:
                    primary_rigid = rigid_primary_by_original.get(pstr)
                    if primary_rigid:
                        vel = rig_vel_by_path.get(primary_rigid)
                        if vel:
                            avx, avy, avz, awx, awy, awz = vel
            
                writers[pstr].writerow([
                f"{t:.6f}", frame, pstr,
                f"{tx:.6f}", f"{ty:.6f}", f"{tz:.6f}",
                f"{roll:.3f}", f"{pitch:.3f}", f"{yaw:.3f}",
                f"{tr[0]:.6f}", f"{tr[1]:.6f}", f"{tr[2]:.6f}",
                f"{roll_deg:.3f}", f"{pitch_deg:.3f}", f"{yaw_deg:.3f}",
                f"{avx:.6f}", f"{avy:.6f}", f"{avz:.6f}",
                f"{awx:.6f}", f"{awy:.6f}", f"{awz:.6f}",
                ])

            frame += 1

    # Close all files
        for f in files.values():
            f.close()

    async def _setup_sim(self):
        def done_callback(e):
            self._setup_sim_succeed = True
            self._setup_sim_sub = None

        # Set up simulation and start data generation
        self._setup_sim_sub = self._sim_manager.register_set_up_simulation_done_callback(done_callback)
        self._sim_manager.set_up_simulation_from_config_file()

        while self._setup_sim_sub and not self._sim_app.is_exiting():
            await self._sim_app.app.next_update_async()

    async def _gen_random_commands(self):
        if self._sim_manager.get_config_file_valid_value("character", "command_file"):
            task = asyncio.create_task(self._sim_manager.generate_random_commands())
            await task
            commands = task.result()
            self._sim_manager.save_commands(commands)
        if self._sim_manager.get_config_file_valid_value("robot", "command_file"):
            task = asyncio.create_task(self._sim_manager.generate_random_robot_commands())
            await task
            commands = task.result()
            self._sim_manager.save_robot_commands(commands)

    # ===== Camera placement related =====

    def _do_camera_placement(self):
        self._read_camera_json()
        if not self.camera_placements_json:
            return
        print(f"camera_placements_json = {self.camera_placements_json}")
        prop = self._sim_manager.get_config_file_property("sensor", "camera_num")
        prop.set_value(len(self.camera_placements_json))
        # Load camera
        self._sim_manager.load_camera_from_config_file()
        self._place_cameras()

    def _read_camera_json(self):
        import json

        import carb
        import omni.client

        # Read json file
        result, version, context = omni.client.read_file(self.camera_file_path)
        if result != omni.client.Result.OK:
            carb.log_error(f"Cannot load camera file path: {self.camera_file_path}. Skip camera placement.")
            return
        json_str = memoryview(context).tobytes().decode("utf-8")
        self.camera_placements_json = json.loads(json_str)

    def _place_cameras(self):
        import carb

        # Perform placement
        from isaacsim.replicator.agent.core.stage_util import CameraUtil

        camera_prims = CameraUtil.get_cameras_in_stage()
        count = 0
        for camera_dict in self.camera_placements_json:
            if count >= len(camera_prims):
                carb.log_warn("No enough cameras in the scene to set placement. Will skip the rest placement data.")
                break
            self._place_one_camera(camera_dict, camera_prims[count])
            count += 1

        print(f"Place total {count} cameras.")

    def _place_one_camera(self, camera_dict, camera_prim):
        from isaacsim.core.utils.rotations import euler_to_rot_matrix
        from isaacsim.replicator.agent.core.stage_util import CameraUtil
        from pxr import Gf

        # Extract focal length
        # - In OV, the default pixel size is 20.955/1920=0.0109140625mm
        ov_focal_length = camera_dict["focal_length"] * 0.0109140625
        # Extract transformation
        ov_pos = Gf.Vec3d(camera_dict["x"], camera_dict["y"], camera_dict["height"])
        yaw = camera_dict["yaw"]
        pitch = camera_dict["pitch"]
        np_mat_yaw = euler_to_rot_matrix(np.array([0, yaw, 0]), degrees=True, extrinsic=False)
        np_mat_pitch = euler_to_rot_matrix(np.array([-pitch, 0, 0]), degrees=True, extrinsic=False)
        np_mat_default = euler_to_rot_matrix(
            np.array([90, -90, 0]), degrees=True, extrinsic=False
        )  # To make sure when yaw=0, the camera in IsaacSim points to X positive
        rot_matrix = (
            Gf.Matrix3d(np_mat_pitch.T.tolist())
            * Gf.Matrix3d(np_mat_yaw.T.tolist())
            * Gf.Matrix3d(np_mat_default.T.tolist())
        )
        # ov_rot_euler = rot_matrix.ExtractRotation().Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
        ov_rot = rot_matrix.ExtractRotation().GetQuat()
        CameraUtil.set_camera(camera_prim, ov_pos, ov_rot, ov_focal_length)


def get_args():
    parser = argparse.ArgumentParser("Actor SDG")
    parser.add_argument("-c", "--config_file", required=True, help="Path to a IRA config file")
    # Optional config
    parser.add_argument(
        "--sensor_placment_file", required=False, help="Path to camera placement json file. Default is none."
    )
    parser.add_argument(
        "--crash_report_path",
        required=False,
        default=None,
        help="Path to store the crash report. Default is the current working directory.",
    )
    parser.add_argument(
        "--no_random_commands",
        required=False,
        default=False,
        action="store_true",
        help="Do not generate random commands.",
    )
    parser.add_argument(
        "--debug_print", required=False, default=False, action="store_true", help="Enable IRA debug print."
    )
    parser.add_argument(
        "--save_usd",
        action="store_true",
        default=False,
        help="Save the simulated scene when data generation finishes.",
    )
    args, _ = parser.parse_known_args()
    return args


# ===== Save USD =====
async def _save_usd(save_as_path):
    print("Saving USD...")
    try:
        import omni.usd

        await omni.usd.get_context().save_as_stage_async(save_as_path)
        print("Save scene to: " + str(save_as_path))
        await omni.usd.get_context().close_stage_async()
    except Exception as e:
        print("Caught exception. Unable to save USD. " + str(e), file=sys.stderr)


def main():
    # Read command line arguments
    args = get_args()
    config_file_path = args.config_file
    camera_file_path = args.sensor_placment_file
    crash_report_path = args.crash_report_path
    no_random_commands = args.no_random_commands
    debug_print = args.debug_print
    save_usd = args.save_usd

    print("Config file path: {}".format(config_file_path))
    print("Sensor placement file path: {}".format(camera_file_path))
    print("Crash Report Path: {}".format(crash_report_path))
    print("Don't random commands: {}".format(no_random_commands))
    print("Debug Print: {}".format(debug_print))
    print("Save USD: {}".format(save_usd))

    # Check files exist
    if not os.path.isfile(config_file_path):
        print("Invalid config file path. Exit.", file=sys.stderr)
        return
    if camera_file_path and not os.path.isfile(camera_file_path):
        print("Invalid camera placement path. Exit.", file=sys.stderr)
        return

    # Start SimApp
    sim_app = SimulationApp(launch_config=APP_CONFIG, experience=BASE_EXP_PATH)

    # Start SDG
    sdg = ActorSDG(
        sim_app,
        os.path.abspath(config_file_path),
        camera_file_path,
        crash_report_path,
        no_random_commands,
        debug_print,
        save_usd,
    )

    from omni.kit.async_engine import run_coroutine

    task = run_coroutine(sdg.run())
    try:
        while not task.done():
            sim_app.update()

        if not task.result():
            print("Failed to run SDG")

        # [Optional] Save USD to
        if save_usd:
            import omni.client

            save_as_path = omni.client.combine_urls("{}/".format(sdg.output_path), "scene.usd")
            save_usd_task = asyncio.ensure_future(_save_usd(save_as_path))
            while not save_usd_task.done():
                sim_app.update()

    # Close app
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
