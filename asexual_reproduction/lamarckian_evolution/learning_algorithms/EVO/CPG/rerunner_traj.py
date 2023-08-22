"""Rerun(watch) a modular robot in Mujoco."""

from pyrr import Quaternion, Vector3
from revolve2.core.modular_robot import ModularRobot
from .runner_mujoco import LocalRunner
from .environment_steering_controller import EnvironmentActorController
from revolve2.core.physics.running import Batch, Environment, PosedActor
from revolve2.core.physics import Terrain
from revolve2.core.physics.running import RecordSettings
import numpy as np
import math
from random import Random
from typing import Optional, Tuple, List

class ModularRobotRerunner:
    """Rerunner for a single robot that uses Mujoco."""

    async def rerun(self, 
                    robot: ModularRobot, 
                    control_frequency: float,
                    terrain: Terrain,
                    targets: List[Tuple[float]],
                    record_dir: Optional[str], 
                    record: bool = False) -> None:
        """
        Rerun a single robot.

        :param robot: The robot the simulate.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        """

        batch = Batch(
            simulation_time=60,
            sampling_frequency=5,
            control_frequency=control_frequency,
        )

        actor, self._controller = robot.make_actor_and_controller()

        env = Environment(EnvironmentActorController(self._controller, targets, steer=True))
        bounding_box = actor.calc_aabb()
        env.actors.append(
            PosedActor(
                actor,
                Vector3([0.0, 0.0, bounding_box.size.z / 2.0 - bounding_box.offset.z]),
                Quaternion(),
                [0.0 for _ in self._controller.get_dof_targets()],
            )
        )
        env.static_geometries.extend(terrain.static_geometry)
        batch.environments.append(env)

        runner = LocalRunner(headless=True, target_points=targets)
        rs = None
        if record:
            rs = RecordSettings(record_dir)
        res = await runner.run_batch(batch, rs)
        traj_x = [env_state.actor_states[0].position[0] for env_state in res.environment_results[0].environment_states]
        traj_y = [env_state.actor_states[0].position[1] for env_state in res.environment_results[0].environment_states]
        traj_z = [env_state.actor_states[0].position[2] for env_state in res.environment_results[0].environment_states]
        return traj_x, traj_y, traj_z




if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )

def generate_targets(num_targets: int, rng: Random, starting_point: Tuple[int] = (0,0), between_dist: float = 1.) -> List[Tuple[int]]:

    targets = []

    x, y = starting_point
    for _ in range(num_targets):
        dx = (rng.random() * (2 * between_dist)) - (1 * between_dist)
        x = x + dx
        dy = math.sqrt((between_dist ** 2) - (dx ** 2))
        y = y - dy
        targets.append((x, y))

    return targets