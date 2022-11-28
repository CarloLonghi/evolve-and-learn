
import math
from random import Random
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from pyrr import Quaternion, Vector3
from .revde_optimizer import RevDEOptimizer
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.modular_robot import Body
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic, make_cpg_network_structure_neighbour)
from revolve2.core.optimization import DbId
from revolve2.core.physics.actor import Actor
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.core.physics.running import (ActorState, Batch,
                                           Environment, PosedActor, Runner)
from .runner_mujoco import LocalRunner
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession


class Optimizer(RevDEOptimizer):
    """
    Optimizer for the problem.

    Uses the generic EA optimizer as a base.
    """

    _body: Body
    _actor: Actor
    _dof_ids: List[int]
    _cpg_network_structure: CpgNetworkStructure

    _runner: Runner

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int
    _target_point: Tuple[float]

    async def ainit_new(  # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        rng: Random,
        population_size: int,
        robot_body: Body,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
        scaling: float,
        cross_prob: float,
    ) -> None:
        """
        Initialize this class async.

        Called when creating an instance using `new`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when saving data to the database during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param rng: Random number generator.
        :param population_size: Population size for the OpenAI ES algorithm.
        :param sigma: Standard deviation for the OpenAI ES algorithm.
        :param learning_rate: Directional vector gain for OpenAI ES algorithm.
        :param robot_body: The body to optimize the brain for.
        :param simulation_time: Time in second to simulate the robots for.
        :param sampling_frequency: Sampling frequency for the simulation. See `Batch` class from physics running.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        :param num_generations: Number of generation to run the optimizer for.
        """
        self._body = robot_body
        self._init_actor_and_cpg_network_structure()

        nprng = np.random.Generator(
            np.random.PCG64(rng.randint(0, 2**63))
        )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.

        nprng = np.random.Generator(
            np.random.PCG64(rng.randint(0, 2**63))
        )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.
        initial_population = nprng.standard_normal((population_size, self._cpg_network_structure.num_connections))

        await super().ainit_new(
            database=database,
            session=session,
            db_id=db_id,
            rng=rng,
            population_size=population_size,
            initial_population=initial_population,
            scaling=scaling,
            cross_prob=cross_prob,
        )

        self._runner = self._init_runner()

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations
        self._target_point = np.random.uniform(low=1, high=10, size=2)

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        rng: Random,
        robot_body: Body,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
    ) -> bool:
        """
        Try to initialize this class async from a database.

        Called when creating an instance using `from_database`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when loading and saving data to the database during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param rng: Random number generator.
        :param robot_body: The body to optimize the brain for.
        :param simulation_time: Time in second to simulate the robots for.
        :param sampling_frequency: Sampling frequency for the simulation. See `Batch` class from physics running.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        :param num_generations: Number of generation to run the optimizer for.
        :returns: True if this complete object could be deserialized from the database.
        """
        if not await super().ainit_from_database(
            database=database,
            session=session,
            db_id=db_id,
            rng=rng,
        ):
            return False

        self._body = robot_body
        self._init_actor_and_cpg_network_structure()

        self._runner = self._init_runner()

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations

        return True

    def _init_actor_and_cpg_network_structure(self) -> None:
        self._actor, self._dof_ids = self._body.to_actor()
        active_hinges_unsorted = self._body.find_active_hinges()
        active_hinge_map = {
            active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
        }
        active_hinges = [active_hinge_map[id] for id in self._dof_ids]

        self._cpg_network_structure = make_cpg_network_structure_neighbour(
            active_hinges
        )

    def _init_runner(self, num_simulators: int = 1) -> None:
        return LocalRunner(headless=True, num_simulators=num_simulators)

    async def _evaluate_population(
        self,
        database: AsyncEngine,
        db_id: DbId,
        population: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
        )

        self._runner = self._init_runner(population.shape[0])

        for params in population:
            initial_state = self._cpg_network_structure.make_uniform_state(
                0.5 * math.pi / 2.0
            )
            weight_matrix = (
                self._cpg_network_structure.make_connection_weights_matrix_from_params(
                    params
                )
            )
            dof_ranges = self._cpg_network_structure.make_uniform_dof_ranges(1.0)
            brain = BrainCpgNetworkStatic(
                initial_state,
                self._cpg_network_structure.num_cpgs,
                weight_matrix,
                dof_ranges,
            )
            controller = brain.make_controller(self._body, self._dof_ids)

            bounding_box = self._actor.calc_aabb()
            env = Environment(EnvironmentActorController(controller))
            env.actors.append(
                PosedActor(
                    self._actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                )
            )
            batch.environments.append(env)

        batch_results = await self._runner.run_batch(batch)

        return np.array(
            [
                self._calculate_fitness(
                    environment_result.environment_states[0].actor_states[0],
                    environment_result.environment_states[-1].actor_states[0],
                    self._target_point
                )
                for environment_result in batch_results.environment_results
            ]
        )

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState, target_point) -> float:
        # TODO simulation can continue slightly passed the defined sim time.

        # distance traveled on the xy plane
        initial_d = math.sqrt(
            (begin_state.position[0] - target_point[0]) ** 2
            + ((begin_state.position[1] - target_point[1]) ** 2)
        )

        final_d = math.sqrt(
            (end_state.position[0] - target_point[0]) ** 2
            + ((end_state.position[1] - target_point[1]) ** 2)
        )

        return initial_d - final_d

    def _must_do_next_gen(self) -> bool:
        return self.generation_number != self._num_generations
