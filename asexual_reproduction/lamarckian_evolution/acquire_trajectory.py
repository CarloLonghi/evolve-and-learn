"""Visualize and simulate the best robot from the optimization process."""

import math
import numpy as np

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic, make_cpg_network_structure_neighbour)
from learning_algorithms.EVO.CPG.rerunner_traj import ModularRobotRerunner
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from genotype import DbGenotype, GenotypeSerializer
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1 as body_develop
from matplotlib import pyplot as plt
import asyncio
import pandas as pd
from _optimizer import DbEAOptimizerIndividual
from revolve2.core.database.serializers import FloatSerializer
import learning_algorithms.EVO.CPG.terrain as terrains
import matplotlib.lines as mlines
from random import Random
from typing import List, Tuple


async def main() -> None:

    trajectories = np.zeros((3, 301))

    rng = Random()

    targets = generate_targets(num_targets=2, rng=rng, between_dist=0.8)
    print(f"Targets: {targets}")

    traj_x, traj_y, traj_z = await run_robot(targets)
    trajectories[0] = traj_x
    trajectories[1] = traj_y
    trajectories[2] = traj_z

    fig, ax = plt.subplots()    
    ax.scatter(trajectories[0,0], trajectories[1,0], marker=",", zorder=3, color='#450053', s=30)
    ax.scatter(trajectories[0,-1], trajectories[1,-1], marker=",", zorder=3, color='#2F728F', s=30)
    target1 = plt.Circle(targets[0], 0.1, color='#FDE723')
    target2 = plt.Circle(targets[1], 0.1, color='#FDE723')
    ax.add_patch(target1)
    ax.add_patch(target2)


    ax.plot(trajectories[0], trajectories[1], linewidth=1, color='#2F728F') #firebrick #6C8EBF","#9673A6"

    purple_square = mlines.Line2D([], [], color='#450053', marker='s', linestyle='None',
                            markersize=10, label='Start point')
    blue_square = mlines.Line2D([], [], color='#2F728F', marker='s', linestyle='None',
                            markersize=10, label='End point_Asexual')
    green_square = mlines.Line2D([], [], color='#62C762', marker='s', linestyle='None',
                                 markersize=10, label='End point_Sexual')
    yellow_circle = mlines.Line2D([], [], color='#FDE723', marker='o', linestyle='None',
                            markersize=10, label='Target point')
    ax.legend(handles=[purple_square, blue_square, green_square, yellow_circle])
    plt.title('Lamarckian Evolution')
    plt.show()

async def run_robot(targets):
    db = open_async_database_sqlite('lamarc_asex_database') # database
    async with AsyncSession(db) as session:
        individuals = (
            (
                await session.execute(
                    select(DbEAOptimizerIndividual.genotype_id, DbEAOptimizerIndividual.final_fitness_id,
                    DbEAOptimizerIndividual.absolute_size, DbEAOptimizerIndividual.proportion, DbEAOptimizerIndividual.num_bricks,
                    DbEAOptimizerIndividual.rel_num_limbs, DbEAOptimizerIndividual.symmetry, DbEAOptimizerIndividual.branching
                    )
                )
            )
            .all()
        )

        fitnesses_ids = [ind.final_fitness_id for ind in individuals]
        fitnesses = np.array([(await FloatSerializer.from_database(session, [id]))[0] for id in fitnesses_ids])
        max_id = np.argsort(fitnesses)[-1]

        genotype_id = individuals[max_id][0]
        genotype_db = (
            (
                await session.execute(
                    select(DbGenotype).filter(
                        DbGenotype.id == genotype_id
                    )
                )
            )
            .all()[0]
        )[0]
        genotype = (await GenotypeSerializer.from_database(session, [genotype_db.id]))[0]

    body = body_develop(genotype.body)

    actor, dof_ids = body.to_actor()
    active_hinges_unsorted = body.find_active_hinges()
    active_hinge_map = {
        active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
    }
    active_hinges = [active_hinge_map[id] for id in dof_ids]

    cpg_network_structure = make_cpg_network_structure_neighbour(
        active_hinges
    )

    brain_genotype = genotype.brain
    grid_size = 22
    num_potential_joints = ((grid_size**2)-1)
    brain_params = []
    for hinge in active_hinges:
        pos = body.grid_position(hinge)
        cpg_idx = int(pos[0] + pos[1] * grid_size + grid_size**2 / 2)
        brain_params.append(brain_genotype.params_array[
            cpg_idx*14
        ])

    for connection in cpg_network_structure.connections:
        hinge1 = connection.cpg_index_highest.index
        pos1 = body.grid_position(active_hinges[hinge1])
        cpg_idx1 = int(pos1[0] + pos1[1] * grid_size + grid_size**2 / 2)
        hinge2 = connection.cpg_index_lowest.index
        pos2 = body.grid_position(active_hinges[hinge2])
        cpg_idx2 = int(pos2[0] + pos2[1] * grid_size + grid_size**2 / 2)
        rel_pos = relative_pos(pos1[:2], pos2[:2])
        idx = max(cpg_idx1, cpg_idx2)
        brain_params.append(brain_genotype.params_array[
            idx*14 + rel_pos
        ])

    initial_state = cpg_network_structure.make_uniform_state(0.5 * math.pi / 2.0)
    weight_matrix = (
        cpg_network_structure.make_connection_weights_matrix_from_params(brain_params)
    )
    dof_ranges = cpg_network_structure.make_uniform_dof_ranges(1.0)
    brain = BrainCpgNetworkStatic(
        initial_state,
        cpg_network_structure.num_cpgs,
        weight_matrix,
        dof_ranges,
    )

    bot = ModularRobot(body, brain)

    rerunner = ModularRobotRerunner()
    traj_x, traj_y, traj_z = await rerunner.rerun(bot, 5, terrains.flat_plane(), targets, record_dir='prova', record=False)
    return traj_x, traj_y, traj_z



def relative_pos(pos1, pos2):
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]

    mapping = {(1,0):1, (1,1):2, (0,1):3, (-1,0):4, (-1,-1):5, (0,-1):6,
                (-1,1):7, (1,-1):8, (2,0):9, (0,2):10, (-2,0):11, (0,-2):12, (0,0):13}
    
    return mapping[(dx,dy)]

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

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())