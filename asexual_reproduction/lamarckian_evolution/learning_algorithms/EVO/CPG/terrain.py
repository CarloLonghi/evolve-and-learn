from typing import Tuple
from revolve2.core.physics import Terrain
from revolve2.standard_resources.terrains import rugged_heightmap, flat
from revolve2.core.physics.running import geometry
from pyrr import Quaternion, Vector3
import numpy as np
import numpy.typing as npt
from noise import pnoise2


def flat_plane(size: Vector3 = Vector3([20.0, 20.0, 0.0])) -> Terrain:
    return flat(size)

def rugged_plane(
    size: Tuple[float, float] = Vector3([4.0, 4.0, 0.0]),
    ruggedness: float = 0.06,
    granularity_multiplier: float = 0.2,
) -> Terrain:
    
    NUM_EDGES = 100  # arbitrary constant to get a nice number of edges

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = real_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )

    max_height = ruggedness
    if max_height == 0.0:
        heightmap = np.zeros(num_edges)
        max_height = 1.0
    else:
        heightmap = rugged
        heightmap /= heightmap.max() # normalize heightmap to get correct max height

    return Terrain(
        static_geometry=[
            geometry.Heightmap(
                position=Vector3([0, 0, 0]),
                orientation=Quaternion(),
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1 + ruggedness,
                heights=heightmap,
            ),
        ]
    )

def real_heightmap(
    size: Tuple[float, float],
    num_edges: Tuple[int, int],
    density: float = 1.0,
) -> npt.NDArray[np.float_]:
    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    obstacle_freq = 3
    dim_obstacle = 2

    return np.fromfunction(
        np.vectorize(
            lambda y, x: abs(pnoise2(
                x / num_edges[0] * C1 * size[0] * density,
                y / num_edges[1] * C1 * size[1] * density,
                OCTAVE,
            )) if (x % obstacle_freq) < dim_obstacle and (y % obstacle_freq) < dim_obstacle else
            0.,
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )


def mixed_track(
) -> Terrain:
    
    size = (10,10)
    ruggedness = 0.1
    granularity_multiplier = 1.

    NUM_EDGES = 100  # arbitrary constant to get a nice number of edges
    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = mixed_track_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )
    max_height = ruggedness
    if max_height == 0.0:
        heightmap = np.zeros(num_edges)
        max_height = 1.0
    else:
        heightmap = (ruggedness * rugged) / ruggedness
        
    return Terrain(
        static_geometry=[
            geometry.Heightmap(
                position=Vector3(),
                orientation=Quaternion(),
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1 + ruggedness,
                heights=heightmap,
            ),
        ]
    )

def mixed_track_heightmap(
    size: Tuple[float, float],
    num_edges: Tuple[int, int],
    density: float = 1.0,
) -> npt.NDArray[np.float_]:
    
    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    return np.fromfunction(
        np.vectorize(
            lambda y, x: mixed_track_map(y,x, size, num_edges, density),
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )

def mixed_track_map(y, x, size, num_edges, density):

    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    X = x / num_edges[0] * 2.0 - 1.0
    Y = y / num_edges[1] * 2.0 - 1.0

    h = 0.

    starting_x = -0.03
    starting_y = -0.08
    path_width = 0.06
    path_len = 0.08
    square_side = path_len + path_width
    

    if ((X >= starting_x and X < starting_x + path_len and Y >= starting_y and Y < starting_y + path_width)) or \
        ((X >= starting_x + path_width and X < starting_x + square_side and Y >= starting_y + path_len and Y < starting_y + square_side)):
        h = 0.0
    elif (X > starting_x + path_width and X < starting_x + path_len and Y > starting_y + path_width and Y < starting_y + path_len) or \
        (X < starting_x or X > starting_x + square_side or Y < starting_y or Y > starting_y + square_side):
        h = 5.0
    else:
        h = pnoise2(
        x / num_edges[0] * C1 * size[0] * density,
        y / num_edges[1] * C1 * size[1] * density,
        OCTAVE)
    
    return h

def plane_track(
) -> Terrain:
    
    size = (10,10)
    ruggedness = 0.1
    granularity_multiplier = 1.

    NUM_EDGES = 100  # arbitrary constant to get a nice number of edges
    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = plane_track_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )
    max_height = ruggedness
    if max_height == 0.0:
        heightmap = np.zeros(num_edges)
        max_height = 1.0
    else:
        heightmap = (ruggedness * rugged) / ruggedness
        
    return Terrain(
        static_geometry=[
            geometry.Heightmap(
                position=Vector3(),
                orientation=Quaternion(),
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1 + ruggedness,
                heights=heightmap,
            ),
        ]
    )

def plane_track_heightmap(
    size: Tuple[float, float],
    num_edges: Tuple[int, int],
    density: float = 1.0,
) -> npt.NDArray[np.float_]:
    
    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    return np.fromfunction(
        np.vectorize(
            lambda y, x: plane_track_map(y,x, size, num_edges, density),
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )

def plane_track_map(y, x, size, num_edges, density):

    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    X = x / num_edges[0] * 2.0 - 1.0
    Y = y / num_edges[1] * 2.0 - 1.0

    h = 0.

    starting_x = -0.03
    starting_y = -0.08
    path_width = 0.06
    path_len = 0.08
    square_side = path_len + path_width
    

    if (X > starting_x + path_width and X < starting_x + path_len and Y > starting_y + path_width and Y < starting_y + path_len) or \
        (X < starting_x or X > starting_x + square_side or Y < starting_y or Y > starting_y + square_side):
        h = 5.0
    else:
        h = 0.0
    
    return h

def rugged_track(
) -> Terrain:
    
    size = (10,10)
    ruggedness = 0.1
    granularity_multiplier = 1.

    NUM_EDGES = 100  # arbitrary constant to get a nice number of edges
    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = rugged_track_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )
    max_height = ruggedness
    if max_height == 0.0:
        heightmap = np.zeros(num_edges)
        max_height = 1.0
    else:
        heightmap = (ruggedness * rugged) / ruggedness
        
    return Terrain(
        static_geometry=[
            geometry.Heightmap(
                position=Vector3(),
                orientation=Quaternion(),
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1 + ruggedness,
                heights=heightmap,
            ),
        ]
    )

def rugged_track_heightmap(
    size: Tuple[float, float],
    num_edges: Tuple[int, int],
    density: float = 1.0,
) -> npt.NDArray[np.float_]:
    
    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    return np.fromfunction(
        np.vectorize(
            lambda y, x: rugged_track_map(y,x, size, num_edges, density),
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )

def rugged_track_map(y, x, size, num_edges, density):

    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    X = x / num_edges[0] * 2.0 - 1.0
    Y = y / num_edges[1] * 2.0 - 1.0

    h = 0.

    starting_x = -0.03
    starting_y = -0.08
    path_width = 0.06
    path_len = 0.08
    square_side = path_len + path_width
    

    if (X > starting_x + path_width and X < starting_x + path_len and Y > starting_y + path_width and Y < starting_y + path_len) or \
        (X < starting_x or X > starting_x + square_side or Y < starting_y or Y > starting_y + square_side):
        h = 5.0
    else:
        h = pnoise2(
        x / num_edges[0] * C1 * size[0] * density,
        y / num_edges[1] * C1 * size[1] * density,
        OCTAVE)
    
    return h