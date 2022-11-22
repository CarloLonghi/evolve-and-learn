"""Setup and running of the openai es optimization program."""

import argparse
import logging
from random import Random

from .optimizer import Optimizer
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen
from revolve2.standard_resources import modular_robots
from .revde_optimizer import DbRevDEOptimizerIndividual
from revolve2.core.database.serializers import Ndarray1xnSerializer
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic, make_cpg_network_structure_neighbour)
import math

from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select


async def main(body, gen, num) -> None:
    """Run the optimization process."""

    POPULATION_SIZE = 2
    NUM_GENERATIONS = 2
    SCALING = 0.5
    CROSS_PROB = 0.9

    SIMULATION_TIME = 10
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 5

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    # random number generator
    rng = Random()
    rng.seed(42)

    # database
    database = open_async_database_sqlite('database/morph/gen_' + str(gen) + '/database_' + str(num))

    # process id generator
    process_id_gen = ProcessIdGen()
    process_id = process_id_gen.gen()

    maybe_optimizer = await Optimizer.from_database(
        database=database,
        process_id=process_id,
        process_id_gen=process_id_gen,
        rng=rng,
        robot_body=body,
        simulation_time=SIMULATION_TIME,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        num_generations=NUM_GENERATIONS,
    )
    if maybe_optimizer is not None:
        logging.info(
            f"Recovered. Last finished generation: {maybe_optimizer.generation_number - 1}."
        )
        optimizer = maybe_optimizer
    else:
        logging.info("No recovery data found. Starting at generation 0.")
        optimizer = await Optimizer.new(
            database,
            process_id,
            process_id_gen,
            rng,
            POPULATION_SIZE,
            body,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,
            scaling=SCALING,
            cross_prob=CROSS_PROB,
        )

    logging.info("Starting controller optimization process..")

    await optimizer.run()

    logging.info("Finished optimizing controller.")

    async with AsyncSession(database) as session:
        best_individual = (
            (
                await session.execute(
                    select(DbRevDEOptimizerIndividual).order_by(
                        DbRevDEOptimizerIndividual.fitness.desc()
                    )
                )
            )
            .scalars()
            .all()[0]
        )

        params = [
            p
            for p in (
                await Ndarray1xnSerializer.from_database(
                    session, [best_individual.individual]
                )
            )[0]
        ]

        actor, dof_ids = body.to_actor()
        active_hinges_unsorted = body.find_active_hinges()
        active_hinge_map = {
            active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
        }
        active_hinges = [active_hinge_map[id] for id in dof_ids]

        cpg_network_structure = make_cpg_network_structure_neighbour(active_hinges)

        initial_state = cpg_network_structure.make_uniform_state(0.5 * math.pi / 2.0)
        weight_matrix = (
            cpg_network_structure.make_connection_weights_matrix_from_params(params)
        )
        dof_ranges = cpg_network_structure.make_uniform_dof_ranges(1.0)
        brain = BrainCpgNetworkStatic(
            initial_state,
            cpg_network_structure.num_cpgs,
            weight_matrix,
            dof_ranges,
        )

        return brain, best_individual.fitness


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
