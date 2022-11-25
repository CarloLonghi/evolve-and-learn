"""Rerun(watch) a modular robot in Mujoco."""

from pyrr import Quaternion, Vector3
from revolve2.core.modular_robot import ModularRobot
from .runner_mujoco import LocalRunner
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.core.physics.running import Batch, Environment, PosedActor


class ModularRobotRerunner:
    """Rerunner for a single robot that uses Mujoco."""

    async def rerun(self, robot: ModularRobot, control_frequency: float) -> None:
        """
        Rerun a single robot.

        :param robot: The robot the simulate.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        """
        batch = Batch(
            simulation_time=1000000,
            sampling_frequency=0.0001,
            control_frequency=control_frequency,
        )

        actor, self._controller = robot.make_actor_and_controller()

        env = Environment(EnvironmentActorController(self._controller))
        env.actors.append(
            PosedActor(
                actor,
                Vector3([0.0, 0.0, 0.1]),
                Quaternion(),
                [0.0 for _ in self._controller.get_dof_targets()],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner(headless=False)
        await runner.run_batch(batch)

if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )