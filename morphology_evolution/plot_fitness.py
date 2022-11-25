"""Visualize and simulate the best robot from the optimization process."""

import argparse
import math
import numpy as np
import pandas

from revolve2.core.database import open_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea._database import (
    DbBase,
    DbEAOptimizer,
    DbEAOptimizerIndividual,
    DbEAOptimizerGeneration
)
from revolve2.core.database.serializers import FloatSerializer, DbFloat
from matplotlib import pyplot as plt
from revolve2.core.optimization import DbId



def plot(database: str, db_id: DbId) -> None:

    """Run the script."""
    db = open_database_sqlite(database)
    df = pandas.read_sql(
        select(
            DbEAOptimizer,
            DbEAOptimizerGeneration,
            DbEAOptimizerIndividual,
            DbFloat,
        ).filter(
            (DbEAOptimizer.db_id == db_id.fullname)
            & (DbEAOptimizerGeneration.ea_optimizer_id == DbEAOptimizer.id)
            & (DbEAOptimizerIndividual.ea_optimizer_id == DbEAOptimizer.id)
            & (DbEAOptimizerIndividual.fitness_id == DbFloat.id)
            & (
                DbEAOptimizerGeneration.individual_id
                == DbEAOptimizerIndividual.individual_id
            )
        ),
        db,
    )

    fitnesses = df[['generation_index','individual_index','individual_id', 'value']].sort_values(by=["generation_index", "individual_index"])
    plt.plot(np.linspace(start=1, stop=fitnesses.shape[0], num=fitnesses.shape[0]), fitnesses['value'])
    plt.show()



def main() -> None:
    """Run this file as a command line tool."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=str,
        help="The database to plot.",
    )
    parser.add_argument("db_id", type=str, help="The id of the ea optimizer to plot.")
    args = parser.parse_args()

    plot(args.database, DbId(args.db_id))

if __name__ == "__main__":
    main()