import meshio
import jax
import jax.numpy as np
import os

# Force JAX to use GPU
jax.config.update("jax_platform_name", "gpu")

import logging
from jax_fem import logger
logger.setLevel(logging.DEBUG)

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type

# ---------------------------------------------------------------------
# Monkey-patch JAX-FEM solver to be more robust for ill-conditioned meshes
# ---------------------------------------------------------------------
import jax_fem.solver as solver_module
import scipy.sparse
from jax.experimental.sparse import BCOO

def robust_jax_solve(A, b, x0, precond):
    """More robust JAX solver with relaxed convergence and GMRES fallback."""
    logger.debug(f"Robust JAX Solver - Solving linear system on GPU")
    indptr, indices, data = A.getValuesCSR()
    A_sp_scipy = scipy.sparse.csr_array((data, indices, indptr), shape=A.getSize())
    A_bcoo = BCOO.from_scipy_sparse(A_sp_scipy).sort_indices()
    
    # Jacobi preconditioner
    jacobi = np.array(A_sp_scipy.diagonal())
    jacobi = np.where(np.abs(jacobi) < 1e-14, 1.0, jacobi)  # Avoid division by zero
    pc = lambda x: x / jacobi if precond else None
    
    # Try BiCGSTAB first with more iterations
    x, info = jax.scipy.sparse.linalg.bicgstab(
        A_bcoo, b, x0=x0, M=pc,
        tol=1e-8, atol=1e-8, maxiter=20000
    )
    
    err = np.linalg.norm(A_bcoo @ x - b) / (np.linalg.norm(b) + 1e-14)
    logger.debug(f"Robust JAX Solver - BiCGSTAB relative residual = {err}")
    
    # If BiCGSTAB didn't converge well, try GMRES
    if err > 0.01:
        logger.debug(f"BiCGSTAB residual too high, trying GMRES...")
        x, info = jax.scipy.sparse.linalg.gmres(
            A_bcoo, b, x0=x, M=pc,
            tol=1e-6, atol=1e-6, maxiter=30000, restart=100
        )
        err = np.linalg.norm(A_bcoo @ x - b) / (np.linalg.norm(b) + 1e-14)
        logger.debug(f"Robust JAX Solver - GMRES relative residual = {err}")
    
    # Relaxed convergence check (relative residual)
    if err > 1.0:
        logger.warning(f"JAX solver converged poorly (rel_err={err}), but continuing...")
    
    return x

# Patch the module
solver_module.jax_solve = robust_jax_solve

# Print device info
print(f"JAX devices: {jax.devices()}")
print(f"Using device: {jax.devices()[0]}")

# Load volume mesh from Gmsh
import sys
name = sys.argv[1]
meshio_mesh = meshio.read(f"./tetrahedral/{name}.msh")

# ----- choose element type for JAX-FEM -----
ele_type = "TET4"                          # 4-node tetrahedron
cell_type = get_meshio_cell_type(ele_type) # this will be "tetra"

# Extract only the tetrahedral cells
points = meshio_mesh.points
cells = meshio_mesh.cells_dict[cell_type]

mesh = Mesh(points, cells)

print("Loaded mesh with", points.shape[0], "nodes and", cells.shape[0], "tets")

# ---------------------------------------------------------------------
# Material properties (adjust for your object)
# ---------------------------------------------------------------------
E = 3.0e9     #this is for PLA
nu = 0.3
mu = E / (2*(1+nu))
lmbda = E*nu / ((1+nu)*(1-2*nu))

class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda*np.trace(eps)*np.eye(3) + 2*mu*eps
            return sigma
        return stress
    
    def get_surface_maps(self):
        traction = np.array([0.0, 0.0, -1.0])  # adjust magnitude/direction
        def surface_map(u, x):
            return traction
        return [surface_map]  

# ---------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------

# Fix bottom by z-coordinate
zmin = points[:, 2].min()
def bottom(p): return np.isclose(p[2], zmin, atol=1e-5)

def zero_val(p): return 0.0

dirichlet_bc_info = [
    [bottom]*3,        # apply to all 3 components
    [0, 1, 2],         # x,y,z
    [zero_val]*3
]

# Apply traction on top
import numpy as onp
import jax.numpy as jnp

pts = onp.asarray(points)
xy = pts[:, :2]
z  = pts[:, 2]

# --- choose bin size (column width) ---
xy_span = xy.max(axis=0) - xy.min(axis=0)
bin_size = 0.01 * float(xy_span.min())   # 1% of XY span (tune this)
bin_size = max(bin_size, 1e-12)

xy_min = xy.min(axis=0)

# integer bin indices per node
bins = onp.floor((xy - xy_min) / bin_size).astype(onp.int32)
bi = bins[:, 0]
bj = bins[:, 1]

nx = int(bi.max()) + 1
ny = int(bj.max()) + 1

# build zmax grid (numpy side)
zmax_grid_np = onp.full((nx, ny), -onp.inf, dtype=pts.dtype)
for i in range(pts.shape[0]):
    ii, jj = int(bi[i]), int(bj[i])
    if z[i] > zmax_grid_np[ii, jj]:
        zmax_grid_np[ii, jj] = z[i]

# convert to JAX arrays (used inside top())
xy_min_j = jnp.array(xy_min)
bin_size_j = jnp.array(bin_size)
zmax_grid = jnp.array(zmax_grid_np)
nx_j = nx
ny_j = ny

# optional: make it a band instead of a razor-thin envelope (recommended)
band = 1e-6  # in your length units; increase if selection is too sparse

def top(p):
    # p is a JAX array
    b = jnp.floor((p[:2] - xy_min_j) / bin_size_j).astype(jnp.int32)
    ii = jnp.clip(b[0], 0, nx_j - 1)
    jj = jnp.clip(b[1], 0, ny_j - 1)

    zmax_local = zmax_grid[ii, jj]

    # Envelope + small thickness band (usually better for FEM traction area):
    return p[2] >= (zmax_local - band)

location_fns = [top]

# ---------------------------------------------------------------------
# Solve (GPU-optimized: use iterative solver instead of UMFPACK)
# ---------------------------------------------------------------------
problem = LinearElasticity(
    mesh=mesh, vec=3, dim=3,
    ele_type=ele_type,
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=location_fns,
)

# GPU solver options - use patched JAX solver (runs entirely on GPU)
solver_options = {
    'jax_solver': {
        'precond': True,
    },
    'tol': 1e-5,
    'rel_tol': 1e-6,
}

logger.setLevel(logging.DEBUG)

print("Solving on GPU…")
import time
start_time = time.time()

u_list = solver(problem, solver_options)

solve_time = time.time() - start_time
print(f"Solve completed in {solve_time:.2f} seconds")

u = u_list[0]

u_grad = problem.fes[0].sol_to_grad(u)
eps = 0.5 * (u_grad + u_grad.transpose(0, 1, 3, 2))
sigma = lmbda * np.trace(eps, axis1=2, axis2=3)[:, :, None, None] * np.eye(3) + 2.0 * mu * eps
cells_JxW = problem.JxW[:, 0, :]
sigma_avg = np.sum(sigma * cells_JxW[:, :, None, None], axis=1) / np.sum(cells_JxW, axis=1)[:, None, None]
dim = 3
mean_stress = np.trace(sigma_avg, axis1=1, axis2=2) / dim
sigma_dev = sigma_avg - mean_stress[:, None, None] * np.eye(dim)
von_mises = np.sqrt(1.5 * np.sum(sigma_dev * sigma_dev, axis=(1, 2)))

# ---------------------------------------------------------------------
# Save results to VTK
# ---------------------------------------------------------------------
os.makedirs("vtk", exist_ok=True)
save_sol(problem.fes[0], u, f"vtk/{name}.vtu", cell_infos=[("von_mises", von_mises)])

print(f"Done! Open vtk/{name}.vtu in ParaView")
