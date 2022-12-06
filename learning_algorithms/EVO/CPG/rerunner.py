"""Rerun(watch) a modular robot in Mujoco."""

from pyrr import Quaternion, Vector3
from revolve2.core.modular_robot import ModularRobot
from runner_mujoco import LocalRunner
#from environment_steering_controller import EnvironmentActorController
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.core.physics.running import Batch, Environment, PosedActor
import math
from revolve2.core.physics.running import RecordSettings

class ModularRobotRerunner:
    """Rerunner for a single robot that uses Mujoco."""

    async def rerun(self, robot: ModularRobot, control_frequency: float) -> None:
        """
        Rerun a single robot.

        :param robot: The robot the simulate.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        """
        batch = Batch(
            simulation_time=30,
            sampling_frequency=5,
            control_frequency=control_frequency,
        )

        actor, self._controller = robot.make_actor_and_controller()
        bounding_box = actor.calc_aabb()
        env = Environment(EnvironmentActorController(self._controller))
        env.actors.append(
            PosedActor(
                actor,
                Vector3([0.0, 0.0, bounding_box.size.z / 2.0 - bounding_box.offset.z]),
                Quaternion(),
                [0.0 for _ in self._controller.get_dof_targets()],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner(headless=False)
        results = await runner.run_batch(batch,)
        print(ModularRobotRerunner._calculate_panoramic_rotation(results.environment_results[0]))


    @staticmethod
    def _calculate_panoramic_rotation(results, vertical_angle_limi = math.pi/4) -> float:
        total_angle = 0.0

        orientations = [env_state.actor_states[0].orientation for env_state in results.environment_states[1:]]
        orientations = [ModularRobotRerunner._from_quaternion_to_axisangle(o)[0] for o in orientations]

        total_angle = abs(sum([o for o in orientations]))

        return total_angle

    @staticmethod
    def _from_quaternion_to_axisangle(rotation: Quaternion):
        theta = 2 * math.acos(rotation.w)
        x = rotation.x / math.sin(theta/2)
        y = rotation.y / math.sin(theta/2)
        z = rotation.z / math.sin(theta/2)
        return x,y,z


if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )
