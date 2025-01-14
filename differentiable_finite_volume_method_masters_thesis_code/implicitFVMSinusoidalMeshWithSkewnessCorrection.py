import phi
phi.verify()
from phi.torch.flow import *
from phi.geom import *

import numpy as np
from typing import Dict, List, Sequence, Union, Optional

def create_quadrilateral_mesh_grid(x_size: int, y_size: int, amplitude=1.0, frequency=1.0) -> Dict[str, Union[np.ndarray, List[Sequence], Dict[str, List[Sequence]]]]:
    # Create vertices with sinusoidal y-coordinates
    points = []
    for y in range(y_size + 1):
        for x in range(x_size + 1):
            # Modify y-coordinate based on a sinusoidal function
            adjusted_y = y + amplitude * np.sin(frequency * np.pi * x / x_size)
            points.append((x, adjusted_y))
    points = np.array(points)

    # Create quadrilaterals
    polygons = []
    for y in range(y_size):
        for x in range(x_size):
            top_left = y * (x_size + 1) + x
            top_right = top_left + 1
            bottom_left = top_left + (x_size + 1)
            bottom_right = bottom_left + 1
            polygons.append((top_left, top_right, bottom_right, bottom_left))

    # Create boundaries
    boundaries = {
        'left_boundary': [],
        'right_boundary': [],
        'top_boundary': [],
        'bottom_boundary': [],
    }
    for y in range(y_size):
        boundaries['left_boundary'].append((y * (x_size + 1), (y + 1) * (x_size + 1)))
        boundaries['right_boundary'].append(((y + 1) * (x_size + 1) - 1, (y + 2) * (x_size + 1) - 1))
    for x in range(x_size):
        boundaries['bottom_boundary'].append((x, x + 1))
        boundaries['top_boundary'].append(((y_size) * (x_size + 1) + x, (y_size) * (x_size + 1) + x + 1))

    # Combine and return the mesh data
    mesh_data = {
        'points': points,
        'polygons': polygons,
        'boundaries': boundaries
    }
    return mesh_data

# Example usage
x_size = 20  # Define the size along the x-axis
y_size = 20  # Define the size along the y-axis
amplitude = 0.5  # Amplitude of the sinusoidal wave
frequency = 2    # Frequency of the sinusoidal wave
mesh_data = create_quadrilateral_mesh_grid(x_size, y_size, amplitude, frequency)

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
    
values_array = np.array([
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

values_tensor = tensor(values_array, spatial('y', 'x'))
values_tensor = pack_dims(values_tensor, spatial, instance(mesh_test))
values_vector = vec(x=values_tensor, y=values_tensor * .1)

# Initialize the simulation
dt = 0.01  # The time step size
velocity = Field(mesh_test, values_vector, {'left_boundary': ZERO_GRADIENT, 'right_boundary': ZERO_GRADIENT, 'top_boundary': ZERO_GRADIENT,'bottom_boundary': ZERO_GRADIENT})
show(velocity)
u_prev_mesh = velocity
totalTimeSteps = 1000  # Set this to your actual total time steps!

# Define the explicit step function for the Burger's equation
@jit_compile_linear
def explicit_burgers_equation_time_step(u, dt):
    diffusivity = 0.1  # The diffusivity constant; adjust as needed
    velocity_spatial_gradient = field.spatial_gradient(u, u.extrapolation, at='center', stack_dim=dual('vector'), order=2, scheme='green-gauss')
    diffusion_term = dt * diffuse.differential(u, diffusivity, gradient=velocity_spatial_gradient, correct_skew=True)
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