"""Visualize and simulate the best robot from the optimization process."""

import math
import numpy as np

from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea._database import (
    DbBase,
    DbEAOptimizerIndividual,
)
from revolve2.core.database.serializers import FloatSerializer
from array_genotype.array_genotype import ArrayGenotypeSerializer as BrainSerializer, develop as brain_develop
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1 as body_develop
from revolve2.genotypes.cppnwin._genotype import GenotypeSerializer as BodySerializer
from matplotlib import pyplot as plt


async def main() -> None:

    """Run the script."""
    db = open_async_database_sqlite('database/')
    async with AsyncSession(db) as session:
        individuals = (
            (
                await session.execute(
                    select(DbEAOptimizerIndividual.genotype_id, DbEAOptimizerIndividual.fitness_id)
                )
            )
            .all()
        )

        fitnesses_ids = [ind[1] for ind in individuals]
        fitnesses = np.array([(await FloatSerializer.from_database(session, [id]))[0] for id in fitnesses_ids])

    plt.plot(fitnesses)
    plt.show()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())