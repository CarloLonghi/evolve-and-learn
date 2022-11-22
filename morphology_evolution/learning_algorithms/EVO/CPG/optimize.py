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
        optimizer = maybe_optimizer
    else:
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

        return best_individual.fitness


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
