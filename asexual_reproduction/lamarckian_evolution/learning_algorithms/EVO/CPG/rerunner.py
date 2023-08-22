"""Rerun(watch) a modular robot in Mujoco."""

import math
from pyrr import Quaternion, Vector3
from revolve2.core.modular_robot import ModularRobot
from .runner_mujoco import LocalRunner
from .environment_steering_controller import EnvironmentActorController
from revolve2.core.physics.running import Batch, Environment, PosedActor
from revolve2.core.physics import Terrain
from revolve2.core.physics.running import RecordSettings
from typing import Optional, List, Tuple
import numpy as np
from matplotlib import pyplot as plt

class ModularRobotRerunner:
    """Rerunner for a single robot that uses Mujoco."""

    async def rerun(self, 
                    robot: ModularRobot, 
                    control_frequency: float, 
                    terrain: Terrain,
                    targets: List[Tuple[int]],
                    record_dir: Optional[str], 
                    record: bool = False) -> None:
        """
        Rerun a single robot.

        :param robot: The robot the simulate.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        """
        batch = Batch(
            simulation_time=90,
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
        batch_results = await runner.run_batch(batch, rs)
        fitness = self._calculate_point_navigation(batch_results.environment_results[0], targets)
        return fitness

    @staticmethod
    def _calculate_point_navigation(results, targets) -> float:
        trajectory = [(0.0, 0.0)] + targets
        distances = [compute_distance(trajectory[i], trajectory[i-1]) for i in range(1, len(trajectory))]
        target_range = 0.1
        reached_target_counter = 0
        coordinates = [env_state.actor_states[0].position[:2] for env_state in results.environment_states]
        lengths = [compute_distance(coordinates[i-1], coordinates[i]) for i in range(1,len(coordinates))]
        starting_idx = 0
        for idx, state in enumerate(coordinates):
            if reached_target_counter < len(targets) and check_target(state, targets[reached_target_counter], target_range):
                reached_target_counter += 1
                starting_idx = idx
        
        fitness = 0
        if reached_target_counter > 0:
            path_len = sum(lengths[:starting_idx])
            fitness = sum(distances[:reached_target_counter]) - 0.1*path_len
        if reached_target_counter == len(targets):
            return fitness
        else:
            if reached_target_counter == 0:
                last_target = (0.0, 0.0)
            else:
                last_target = trajectory[reached_target_counter]
            last_coord = coordinates[-1]
            distance = compute_distance(targets[reached_target_counter], last_target)
            distance -= compute_distance(targets[reached_target_counter], last_coord)
            new_path_len = sum(lengths[:]) - sum(lengths[:starting_idx])
            return fitness + (distance - 0.1*new_path_len)

def check_target(coord, target, target_range):
    if abs(coord[0]-target[0]) < target_range and abs(coord[1]-target[1]) < target_range:
        return True
    else:
        return False

def compute_distance(point_a, point_b):
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2
    )

if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )