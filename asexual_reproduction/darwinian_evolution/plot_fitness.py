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
)
from _optimizer import DbEAOptimizerIndividual, DbEAOptimizerGeneration
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
        ).filter(
            (DbEAOptimizer.db_id == db_id.fullname)
            & (DbEAOptimizerGeneration.ea_optimizer_id == DbEAOptimizer.id)
        ),
        db,
    )
    describe = (
        df[["generation_index", "fitness"]]
        .groupby(by="generation_index")
        .describe()
    )

    df.to_csv('change_env.csv')

    a = df[["generation_index", "individual_id", "fitness"]].groupby(by="generation_index")['fitness'].transform(max) == df['fitness']
    b = df[a]
    print(b[['individual_id', 'fitness', 'generation_index']])

    c = df[["generation_index", "fitness"]].groupby(by="generation_index").max('fitness')
    plt.plot(c)
    plt.show()

    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()

    # plot max min mean, std
    describe[["max", "mean", "min"]].plot()
    plt.fill_between(range(len(mean)), mean - std, mean + std)
    plt.title("Darwinian Evolution")
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