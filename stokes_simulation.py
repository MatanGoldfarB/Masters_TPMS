#!/usr/bin/env python3
"""
Stokes flow simulation through a TPMS fluid domain mesh.

Solves the incompressible Stokes equations:
    -mu * laplacian(u) + grad(p) = 0
    div(u) = 0

with boundary conditions:
    inlet:     u = (0, 0, u_in)   prescribed inflow velocity
    outlet:    p = 0              zero pressure (natural BC)
    walls:     u = 0              no-slip (side walls + TPMS surface)

Uses FEniCSx (dolfinx) with Taylor-Hood elements (P2 velocity / P1 pressure).

Usage:
    python stokes_simulation.py tetrahedral/output_fluid.msh \\
        --u-in 0.001 --mu 1e-3 --out-dir vtk/

Requirements: fenics-dolfinx, petsc4py, mpi4py
    Install: conda install -c conda-forge fenics-dolfinx mpich petsc
"""

import argparse
import os
import numpy as np

# ── macOS: Use system clang for JIT to avoid conda LTO linker error ───
if os.uname().sysname == "Darwin" and "CC" not in os.environ:
    for _cc, _var in [("/usr/bin/clang", "CC"), ("/usr/bin/clang++", "CXX")]:
        if os.path.exists(_cc):
            os.environ[_var] = _cc
# ──────────────────────────────────────────────────────────────────────

# ── Fix PETSc/numpy type mismatch ─────────────────────────────────────
# petsc4py exports IntType/ScalarType/RealType as distinct classes that
# *look like* numpy types but aren't recognised by numba or ctypeslib.
# Remap them to the real numpy types before dolfinx tries to use them.
from petsc4py import PETSc as _PETSc
for _attr in ("IntType", "ScalarType", "RealType"):
    _pt = getattr(_PETSc, _attr, None)
    if _pt is not None and hasattr(_pt, "__name__"):
        setattr(_PETSc, _attr, np.dtype(_pt.__name__).type)
# ──────────────────────────────────────────────────────────────────────

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element
import ufl


def main():
    parser = argparse.ArgumentParser(
        description="Stokes flow simulation through TPMS fluid domain"
    )
    parser.add_argument("mesh", help="Path to fluid domain .msh file")
    parser.add_argument("--u-in", type=float, default=0.001,
                        help="Inlet velocity magnitude [m/s] (default: 0.001)")
    parser.add_argument("--mu", type=float, default=1e-3,
                        help="Dynamic viscosity [Pa*s] (default: 1e-3, water)")
    parser.add_argument("--out-dir", default="vtk",
                        help="Output directory for VTU files (default: vtk/)")
    parser.add_argument("--flow-axis", choices=["x", "y", "z"], default="z",
                        help="Flow direction axis (must match mesh_fluid_domain.py)")
    args = parser.parse_args()

    flow_idx = {"x": 0, "y": 1, "z": 2}[args.flow_axis]

    # Physical group tags from mesh_fluid_domain.py
    INLET_TAG = 1
    OUTLET_TAG = 2
    SIDE_WALLS_TAG = 3
    TPMS_WALL_TAG = 4

    # ── 1. Load mesh ─────────────────────────────────────────────────
    print(f"[1/4] Loading mesh: {args.mesh}")
    domain, cell_tags, facet_tags = io.gmshio.read_from_msh(
        args.mesh, MPI.COMM_WORLD, rank=0, gdim=3
    )
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    print(f"  {domain.topology.index_map(tdim).size_local} cells, "
          f"{domain.topology.index_map(0).size_local} vertices")

    # ── 2. Define function spaces (Taylor-Hood: P2/P1) ───────────────
    print("[2/4] Setting up Stokes problem")

    cell = domain.basix_cell()
    P2 = element("Lagrange", cell, 2, shape=(3,))   # velocity vector
    P1 = element("Lagrange", cell, 1)                # pressure scalar
    TH = mixed_element([P2, P1])
    W = fem.functionspace(domain, TH)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    mu = fem.Constant(domain, default_scalar_type(args.mu))

    # ── 3. Boundary conditions ────────────────────────────────────────
    print("[3/4] Applying boundary conditions")

    W0 = W.sub(0)
    V, V_to_W = W0.collapse()

    # --- No-slip on TPMS walls and side walls ---
    noslip_func = fem.Function(V)
    noslip_func.x.array[:] = 0.0

    tpms_facets = facet_tags.find(TPMS_WALL_TAG)
    tpms_dofs = fem.locate_dofs_topological((W0, V), fdim, tpms_facets)
    bc_tpms = fem.dirichletbc(noslip_func, tpms_dofs, W0)

    side_facets = facet_tags.find(SIDE_WALLS_TAG)
    side_dofs = fem.locate_dofs_topological((W0, V), fdim, side_facets)
    bc_sides = fem.dirichletbc(noslip_func, side_dofs, W0)

    # --- Prescribed inflow velocity on inlet ---
    inflow_func = fem.Function(V)
    inflow_vel = np.zeros(3, dtype=default_scalar_type)
    inflow_vel[flow_idx] = args.u_in
    inflow_func.interpolate(lambda x: np.tile(inflow_vel[:, None], (1, x.shape[1])))

    inlet_facets = facet_tags.find(INLET_TAG)
    inlet_dofs = fem.locate_dofs_topological((W0, V), fdim, inlet_facets)
    bc_inlet = fem.dirichletbc(inflow_func, inlet_dofs, W0)

    bcs = [bc_tpms, bc_sides, bc_inlet]

    print(f"  Inlet: {len(inlet_facets)} facets, velocity[{args.flow_axis}] = {args.u_in}")
    print(f"  TPMS wall: {len(tpms_facets)} facets (no-slip)")
    print(f"  Side walls: {len(side_facets)} facets (no-slip)")

    # ── Variational formulation ───────────────────────────────────────
    a = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)

    f = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0)))
    L = ufl.inner(f, v) * ufl.dx

    # ── 4. Solve ──────────────────────────────────────────────────────
    n_cells = domain.topology.index_map(tdim).size_local
    print(f"[4/4] Solving Stokes system ({n_cells} cells) ...")

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )
    wh = problem.solve()

    # ── Extract velocity and pressure ─────────────────────────────────
    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()
    uh.name = "velocity"
    ph.name = "pressure"

    # ── Post-process: interpolate to P1 for output ───────────────────
    V1_vec = fem.functionspace(domain, element("Lagrange", cell, 1, shape=(3,)))
    V1 = fem.functionspace(domain, element("Lagrange", cell, 1))

    u_out = fem.Function(V1_vec, name="velocity")
    u_out.interpolate(uh)

    u_mag = fem.Function(V1, name="velocity_magnitude")
    u_mag.interpolate(fem.Expression(
        ufl.sqrt(ufl.dot(uh, uh)),
        V1.element.interpolation_points()
    ))

    p_out = fem.Function(V1, name="pressure")
    p_out.interpolate(ph)

    # ── Export results ────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.mesh))[0]

    # Write a single VTU file with all fields (easier to use in ParaView)
    import meshio
    coords = domain.geometry.x[:, :3]
    cells = domain.topology.connectivity(3, 0)
    cell_list = [cells.links(i) for i in range(cells.num_nodes)]
    cell_array = np.array(cell_list)

    u_arr = u_out.x.array.reshape(-1, 3)
    u_mag_arr = u_mag.x.array.copy()
    p_arr_out = p_out.x.array.copy()

    vtu_path = os.path.join(args.out_dir, f"{stem}_flow.vtu")
    mesh_out = meshio.Mesh(
        points=coords,
        cells=[("tetra", cell_array)],
        point_data={
            "velocity": u_arr,
            "velocity_magnitude": u_mag_arr,
            "pressure": p_arr_out,
        },
    )
    meshio.write(vtu_path, mesh_out)

    print(f"\nResults written to: {vtu_path}")
    print("  Open in ParaView to visualize.")

    # Summary statistics
    u_arr = u_out.x.array.reshape(-1, 3)
    u_max = np.max(np.linalg.norm(u_arr, axis=1))
    p_arr = p_out.x.array
    print(f"\nSummary:")
    print(f"  Max velocity magnitude: {u_max:.6e} m/s")
    print(f"  Pressure range: [{p_arr.min():.6e}, {p_arr.max():.6e}] Pa")

    coords = domain.geometry.x
    L_char = np.max(coords[:, flow_idx]) - np.min(coords[:, flow_idx])
    rho = 1000.0  # water
    Re = rho * u_max * L_char / args.mu
    print(f"  Estimated Re: {Re:.2f} (L={L_char:.3f} m)")
    if Re > 1:
        print("  NOTE: Re > 1; Stokes assumption may not hold.")

    print("\nDone!")


if __name__ == "__main__":
    main()
