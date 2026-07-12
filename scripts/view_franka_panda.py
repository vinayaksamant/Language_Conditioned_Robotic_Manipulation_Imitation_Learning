from __future__ import annotations

import time
from importlib import resources

import mujoco
import mujoco.viewer

from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment


def main() -> None:
    model_path = resources.files("robot_manipulation_pi0.sim").joinpath(
        "assets",
        PickPlaceConfig(
            robot_name="franka_panda",
            object_names=("cube",),
            workspace_size=0.5,
        ).model_file,
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    environment = PickPlaceEnvironment(
        PickPlaceConfig(robot_name="franka_panda", object_names=("cube",), workspace_size=0.5)
    )
    observation = environment.reset(seed=0)
    data.qpos[:] = environment.data.qpos
    data.ctrl[:] = environment.data.ctrl
    mujoco.mj_forward(model, data)

    print("Loaded:", model_path)
    print("End effector:", observation["ee_pos"])
    print("Object:", observation["object_pos"])
    print("Target:", observation["target_pos"])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat[:] = (0.45, 0.0, 0.35)

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
