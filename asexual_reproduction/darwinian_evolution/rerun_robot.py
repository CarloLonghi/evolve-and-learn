import math
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic, make_cpg_network_structure_neighbour)
from learning_algorithms.EVO.CPG.rerunner import ModularRobotRerunner
from revolve2.core.optimization.ea.generic_ea._database import (
    DbBase,
)
import learning_algorithms.EVO.CPG.terrain as terrains

async def main(body, brain_params, terrain) -> float:

    actor, dof_ids = body.to_actor()
    active_hinges_unsorted = body.find_active_hinges()
    active_hinge_map = {
        active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
    }
    active_hinges = [active_hinge_map[id] for id in dof_ids]

    cpg_network_structure = make_cpg_network_structure_neighbour(
        active_hinges
    )

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
    fitness = await rerunner.rerun(bot, 5, terrain, True, None, False)
    return fitness