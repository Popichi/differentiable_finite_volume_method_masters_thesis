import phi
phi.verify()
from phi.torch.flow import *
from phi.geom import *

import numpy as np
from typing import Dict, List, Sequence, Union, Optional

math.set_global_precision(64)

def print_mesh_data(mesh_data: Dict[str, Union[np.ndarray, List[Sequence], Dict[str, List[Sequence]]]], x_size: int, y_size: int) -> None:
    # Calculate the number of points per row (x_size + 1)
    points_per_row = x_size + 1
    polygons_per_row = x_size
    boundaries_per_row = (x_size + 1) // 2

    # Print points
    points = mesh_data['points']
    print("points = [")
    for i, point in enumerate(points):
        end_char = ", " if (i + 1) % points_per_row != 0 else ",\n"
        print(f"    {tuple(point)}", end=end_char)
    print("]")

    # Print polygons
    polygons = mesh_data['polygons']
    print("\npolygons = [")
    for i, polygon in enumerate(polygons):
        end_char = ", " if (i + 1) % polygons_per_row != 0 else ",\n"
        print(f"    {polygon}", end=end_char)
    print("]")

    # Print boundaries
    boundaries = mesh_data['boundaries']
    print("\nboundaries = {")
    for boundary_name, boundary_edges in boundaries.items():
        print(f"    '{boundary_name}': [")
        for i, edge in enumerate(boundary_edges):
            end_char = ", " if (i + 1) % boundaries_per_row != 0 else ",\n"
            print(f"        {edge}", end=end_char)
        print("    ],")
    print("}")

def create_quadrilateral_mesh_grid(x_size: int, y_size: int, obstacle_mask: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, List[Sequence], Dict[str, List[Sequence]]]]:
    """
    Create a mesh grid of quadrilaterals.

    If an obstacle edge is added twice to the dictionary of obstacle edges and then it is not 
    added to any boundaries because it is a part of the mesh that is not meshed at all. For 
    example, consider a case when we have a 5 x 5 quadrilateral grid and the vertex numbers
    are as follows:
     0  1  2  3  4
     5  6  7  8  9
     9 10 11 12 13
    14 15 16 17 18
    19 20 21 22 23
    If the quadrilaterals that corresponded to vertices (6, 7, 11, 10) and (7, 8, 12, 11) were 
    specified to be obstacles, then the edge (7, 11) would fall within the obstacle part of the 
    mesh. Therefore, it wouldn't be an obstacle edge like edges (6, 7), (12, 11), etc., which 
    would mean it is not added to any boundary edges list.
    This also means that although edges like (6, 7), (12, 11), etc. belong to cells that were 
    labeled as obstacle cells, they are shared with non-obstacle cells of the mesh. That means
    they make up the boundary between the domain of the flow and obstacle and as a result are
    labeled as obstacle edges.

    Args:
        x_size (int): The number of cells horizontally.
        y_size (int): The number of cells vertically.
        obstacle_mask (Optional[np.ndarray]): A 2D boolean array where True indicates that the 
            corresponding cell in the mesh is an obstacle and should be deleted from the original 
            mesh.

    Returns:
        Dict[str, Union[np.ndarray, List[Sequence], Dict[str, List[Sequence]]]]: A dictionary containing points, polygons, and boundaries of the mesh.
    """

    # Create vertices
    points = []
    for y in range(y_size + 1):
        for x in range(x_size + 1):
            points.append((x, y))
    points = np.array(points)

    # Initialize data structures
    polygons = []
    polygon_matrix = [[None for _ in range(x_size)] for _ in range(y_size)]
    cell_edges = {}  # Dictionary to store edges and their counts

    for y in range(y_size):
        for x in range(x_size):
            # Calculate vertex indices
            top_left = y * (x_size + 1) + x
            top_right = top_left + 1
            bottom_left = top_left + (x_size + 1)
            bottom_right = bottom_left + 1

            # Create and store the polygon
            polygon = (top_left, top_right, bottom_right, bottom_left)
            polygons.append(polygon)
            polygon_matrix[y][x] = polygon  # Map the matrix position to the polygon

            # Define the edges in a consistent order (sorted tuple)
            edges = [
                tuple(sorted((top_left, top_right))),  # Top edge
                tuple(sorted((top_right, bottom_right))),  # Right edge
                tuple(sorted((bottom_right, bottom_left))),  # Bottom edge
                tuple(sorted((bottom_left, top_left)))  # Left edge
            ]

            # Add edges to the cell_edges dictionary
            for edge in edges:
                if edge in cell_edges:
                    cell_edges[edge] += 1  # Increment count if edge exists
                else:
                    cell_edges[edge] = 1  # Add new edge with count of 1

    # Now cell_edges contains all edges with their respective counts

    # Process the obstacle_mask to remove the corresponding polygons and update edges
    if obstacle_mask is not None:
        for y in range(y_size):
            for x in range(x_size):
                if obstacle_mask[y][x] is True:
                    polygon = polygon_matrix[y][x]
                    polygons.remove(polygon)

                    # Access the edges of the polygon being removed
                    edges = [
                        tuple(sorted((polygon[0], polygon[1]))),  # Top edge
                        tuple(sorted((polygon[1], polygon[2]))),  # Right edge
                        tuple(sorted((polygon[2], polygon[3]))),  # Bottom edge
                        tuple(sorted((polygon[3], polygon[0])))   # Left edge
                    ]

                    # Decrement the corresponding edges from the cell_edges map
                    for edge in edges:
                        if edge in cell_edges:
                            cell_edges[edge] -= 1  # Decrement the count
                            if cell_edges[edge] == 0:
                                del cell_edges[edge]  # Remove edge if count reaches 0

    # Create boundaries
    boundaries = {
        'left_boundary': [],
        'right_boundary': [],
        'top_boundary': [],
        'bottom_boundary': [],
        'obstacle_boundary': [],
    }

    for y in range(y_size):
        # Left boundary
        left_edge = tuple(sorted((y * (x_size + 1), (y + 1) * (x_size + 1))))
        if left_edge in cell_edges:
            boundaries['left_boundary'].append(left_edge)
        # Right boundary
        right_edge = tuple(sorted(((y + 1) * (x_size + 1) - 1, (y + 2) * (x_size + 1) - 1)))
        if right_edge in cell_edges:
            boundaries['right_boundary'].append(right_edge)

    for x in range(x_size):
        # Bottom boundary
        bottom_edge = tuple(sorted((x, x + 1)))
        if bottom_edge in cell_edges:
            boundaries['bottom_boundary'].append(bottom_edge)
        # Top boundary
        top_edge = tuple(sorted(((y_size) * (x_size + 1) + x, (y_size) * (x_size + 1) + x + 1)))
        if top_edge in cell_edges:
            boundaries['top_boundary'].append(top_edge)

    # Add the remaining edges with count of 1 to the obstacle boundary
    for edge, count in cell_edges.items():
        if count == 1:
            if (
                edge not in boundaries['left_boundary'] and
                edge not in boundaries['right_boundary'] and
                edge not in boundaries['bottom_boundary'] and
                edge not in boundaries['top_boundary']
            ):
                boundaries['obstacle_boundary'].append(edge)

    # Combine and return the mesh data
    mesh_data = {
        'points': points,
        'polygons': polygons,
        'boundaries': boundaries
    }
    return mesh_data

obstacles = [
    #  0      1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19 
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 0
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 1
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 2
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 3
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 4
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 5
    [False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True, False, False, False, False, False],  # 6
    [False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True, False, False, False, False, False],  # 7
    [False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True, False, False, False, False, False],  # 8
    [False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True, False, False, False, False, False],  # 9
    [False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True, False, False, False, False, False],  # 10
    [False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True, False, False, False, False, False],  # 11
    [False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True, False, False, False, False, False],  # 12
    [False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True, False, False, False, False, False],  # 13
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 14
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 15
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 16
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 17
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 18
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]   # 19
]

# Example usage:
x_size = 20  # Define the size along the x-axis
y_size = 20  # Define the size along the y-axis
mesh_data = create_quadrilateral_mesh_grid(x_size, y_size, obstacles)

print_mesh_data(mesh_data, x_size, y_size)

# You would then use mesh_data['points'], mesh_data['polygons'], and mesh_data['boundaries']
# to create your mesh using mesh_from_numpy or any similar function you have.
# Example (you would replace this with your actual mesh creation function):
# mesh = geom.mesh_from_numpy(mesh_data['points'], mesh_data['polygons'], mesh_data['boundaries'])

# Use mesh_from_numpy to create the mesh
mesh_test = mesh_from_numpy(
    points=mesh_data['points'],
    polygons=mesh_data['polygons'],
    boundaries=mesh_data['boundaries'],
)

def create_value_grid(x_size: int, y_size: int, value_mask: np.ndarray, value: int, obstacle_mask: Optional[np.ndarray] = None) -> List[List[int]]:
    """
    Create a grid of dimensions x_size by y_size, setting specified values at given coordinates based on a boolean mask.

    Parameters:
    - x_size (int): Number of columns in the grid.
    - y_size (int): Number of rows in the grid.
    - value_mask (np.ndarray): 2D boolean array indicating which cells should be set to 'value'.
    - value (int or float): The value to set at the specified coordinates.
    - obstacle_mask (Optional[np.ndarray]): 2D boolean array indicating obstacle cells.

    Returns:
    - List[List[int]]: The resulting grid with the specified values set.
    """
    # Create the initial grid of zeros
    grid = np.zeros((y_size, x_size), dtype=int)

    # If obstacle_mask is provided, create a non-uniform grid
    if obstacle_mask is not None:
        non_uniform_grid = []
        for y in range(y_size):
            grid_row = []
            for x in range(x_size):
                if not obstacle_mask[y][x]:
                    if value_mask[y][x]:
                        grid_row.append(value)
                    else:
                        grid_row.append(0)
            if grid_row:  # Only append rows that have non-obstacle cells
                non_uniform_grid.append(grid_row)
        
        return non_uniform_grid
    else:
        # Apply the value to the cells where the value_mask is True
        grid[value_mask] = value
        return grid.tolist()

should_set_the_given_value = [
    #  0      1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19 
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 0
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 1
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 2
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 3
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 4
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 5
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 6
    [False, False, False,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 7
    [False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 8
    [False,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 9
    [False,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 10
    [False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 11
    [False, False, False,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 12
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 13
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 14
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 15
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 16
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 17
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # 18
    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]   # 19
]

value = 10
value_mask = np.array(should_set_the_given_value)
obstacle_mask = np.array(obstacles)
values_array = create_value_grid(x_size, y_size, value_mask, value, obstacle_mask)

from typing import Tuple
from phi.field import laplace, divergence, spatial_gradient, where, CenteredGrid, PointCloud, Field, resample
from phi.physics.fluid import _pressure_extrapolation
from phiml.math._magic_ops import copy_with
laplace_jit = jit_compile_linear(laplace)
def make_incompressible_mesh(velocity: Field,
                             dt: float = 1.0,
                             mesh_structure: Mesh = None,
                             solve: Solve = Solve(),
                             order: int = 2) -> Tuple[Field, Field]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.
    
    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
        velocity: Vector field sampled on a grid.
        obstacles: `Obstacle` or `phi.geom.Geometry` or tuple/list thereof to specify boundary conditions inside the domain.
        solve: `Solve` object specifying method and tolerances for the implicit pressure solve.
            order: spatial order for derivative computations.
            For Higher-order schemes, the laplace operation is not conducted with a stencil exactly corresponding to the one used in divergence calculations but a smaller one instead.
            While this disrupts the formal correctness of the method it only induces insignificant errors and yields considerable performance gains.
            supported: explicit 2/4th order - implicit 6th order (obstacles are only supported with explicit 2nd order)

    Returns:
        velocity: divergence-free velocity of type `type(velocity)`
        pressure: solved pressure field
    """
    input_velocity = velocity
    div = (1.0 / dt) * divergence(velocity, order=order)
    assert not channel(div), f"Divergence must not have any channel dimensions. This is likely caused by an improper velocity field v={input_velocity}"
    # --- Linear solve ---
    if solve.x0 is None:
        pressure_extrapolation = _pressure_extrapolation(input_velocity.extrapolation)
        solve = copy_with(solve, x0=Field(mesh_structure, 0, pressure_extrapolation))
    pressure = math.solve_linear(laplace_jit, div, solve, order=order, correct_skew=False)
    # --- Subtract grad p ---
    grad_pressure = field.spatial_gradient(pressure, input_velocity.extrapolation, at=velocity.sampled_at, order=order, scheme='green-gauss')
    velocity = (velocity - dt * grad_pressure).with_extrapolation(input_velocity.extrapolation)
    return velocity, pressure
    
values_array_uniform = np.array([
    #0   1   2   3   4   5   6  7  8  9 10 11 12 13 14 15 16 17 18 19
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
    [0,  0,  0, 10, 10,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
    [0,  0, 10, 10, 10, 10,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
    [0, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
    [0, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
    [0,  0, 10, 10, 10, 10,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
    [0,  0,  0, 10, 10,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 17
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 18
    [0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 19
])

flat_array = np.concatenate(values_array)
values_tensor = tensor(flat_array, instance('cells'))
values_vector = vec(x=values_tensor, y=values_tensor * .1)

# Initialize the simulation
dt = 0.01  # The time step size
velocity = Field(mesh_test, values_vector, {'left_boundary': ZERO_GRADIENT, 'right_boundary': ZERO_GRADIENT, 'top_boundary': ZERO_GRADIENT,'bottom_boundary': ZERO_GRADIENT, 'obstacle_boundary': vec(x=0, y=0)})
u_prev_mesh = velocity
show(velocity)
totalTimeSteps = 1000  # Set this to your actual total time steps!

# Define the explicit step function for the Burger's equation
@jit_compile_linear
def explicit_burgers_equation_time_step(u, dt):
    diffusivity = 0.1  # The diffusivity constant; adjust as needed
    diffusion_term = dt * diffuse.differential(u, diffusivity, correct_skew=False)
    advection_term = dt * advect.differential(u, u_prev_mesh, order=1)
    return u + advection_term + diffusion_term

def implicit_burgers_equation_time_step(u, dt):
  global u_prev_mesh  # Tell Python to use the global u_prev_mesh
  old_u = u
  A, bias = math.matrix_from_function(explicit_burgers_equation_time_step, u, dt)
  b = u
  u_next = math.solve_linear(A, b - bias, Solve('biCG-stab(2)', x0=u))
  u_prev_mesh = old_u
  return u_next

def fvm_explicit_step(v, p, dt):
    v = implicit_burgers_equation_time_step(v, dt=-dt)
    # --- make incompressible ---
    v, p = make_incompressible_mesh(v, dt, mesh_structure=mesh_test, solve=Solve(x0=p))
    return v, p

velocity0, pressure0 = make_incompressible_mesh(velocity, dt, mesh_test)
trajectory = math.iterate(fvm_explicit_step, batch(time=totalTimeSteps), velocity0, pressure0, dt=dt)
show(trajectory)

# Initialize the simulation
dt = 0.01  # The time step size
values_tensor_grid = tensor(values_array_uniform, spatial('y', 'x'))
values_tensor_grid = math.transpose(values_tensor_grid, 'x, y')
velocity_field_grid = CenteredGrid(vec(x=values_tensor_grid, y=values_tensor_grid * .1), {'x': ZERO_GRADIENT, 'y':ZERO_GRADIENT}, x=20, y=20)
velocity_field_grid = velocity_field_grid.at_faces(dot_face_normal=True)
show(velocity_field_grid.at_centers().as_points())
obstacle = Box(x=(10, 15), y=(6, 14))
totalTimeSteps = 1000  # Set this to your actual total time steps!

def operator_split_step(v, p, dt):
    diffusivity = 0.1
    v = diffuse.explicit(v, diffusivity, dt)
    v = advect.semi_lagrangian(v, v, dt)  # velocity self-advection
    v, p = fluid.make_incompressible(v, obstacle, Solve('biCG-stab(2)', x0=p))
    return v, p

# Visualize the results
velocity0_grid, pressure0_grid = fluid.make_incompressible(velocity_field_grid, obstacle, Solve('biCG-stab(2)'))
v_trj, p_trj = math.iterate(operator_split_step, batch(time=totalTimeSteps), velocity0_grid, pressure0_grid, dt=dt)
v_points = v_trj.at_centers().as_points()
p_points = p_trj.at_centers().as_points()
show(v_points)
show(p_points)