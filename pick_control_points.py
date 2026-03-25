#!/usr/bin/env python3
"""Interactive control point picker for geodesic TPMS.

Uses Open3D VisualizerWithVertexSelection for vertex picking.
Shift+Click to pick vertices, then enter n_cells and thickness
in the terminal. Saves to a control points file on exit.

Usage:
    python3 pick_control_points.py <mesh_file> [output.txt]

Controls:
    Shift + Left Click  : Pick a vertex
    Q / close window    : Finish and save
"""

import sys
import os
import numpy as np
import open3d as o3d
import trimesh


def load_mesh_o3d(path):
    """Load mesh from OFF/OBJ/STL/PLY using trimesh, convert to Open3D."""
    tm = trimesh.load(path, force='mesh')
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(tm.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tm.faces))
    mesh.compute_vertex_normals()
    return mesh


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pick_control_points.py <mesh_file> [output.txt]")
        sys.exit(1)

    mesh_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "control_points.txt"

    if not os.path.exists(mesh_path):
        print(f"Error: mesh file not found: {mesh_path}")
        sys.exit(1)

    mesh = load_mesh_o3d(mesh_path)
    mesh.paint_uniform_color([0.55, 0.75, 0.85])

    bbox = mesh.get_axis_aligned_bounding_box()
    bb_min = np.asarray(bbox.min_bound)
    bb_max = np.asarray(bbox.max_bound)
    bb_size = bb_max - bb_min
    diag = np.linalg.norm(bb_size)

    print(f"\n{'='*50}")
    print(f"  Control Point Picker — {os.path.basename(mesh_path)}")
    print(f"{'='*50}")
    print(f"  Bounding box:")
    print(f"    X: [{bb_min[0]:.3f}, {bb_max[0]:.3f}]  ({bb_size[0]:.3f})")
    print(f"    Y: [{bb_min[1]:.3f}, {bb_max[1]:.3f}]  ({bb_size[1]:.3f})")
    print(f"    Z: [{bb_min[2]:.3f}, {bb_max[2]:.3f}]  ({bb_size[2]:.3f})")
    print(f"    Diagonal: {diag:.3f}")
    print(f"{'='*50}")
    print(f"  Shift+Click to pick vertices on the mesh.")
    print(f"  After each pick, enter n_cells and thickness here.")
    print(f"  Close the window or press Q when done.")
    print(f"{'='*50}\n")

    # Dict: vertex_index -> {coord, n_cells, thickness}
    control_points = {}
    pick_order = []  # track insertion order for display
    seen_indices = set()  # vertex indices we've already prompted for

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(
        window_name=f"Pick Control Points — {os.path.basename(mesh_path)}",
        width=1200, height=800
    )
    vis.add_geometry(mesh)

    print("  Waiting for picks... (Shift+Click on mesh)")

    # Manual event loop: poll, detect new picks, pause for input
    while vis.poll_events():
        vis.update_renderer()

        picked = vis.get_picked_points()
        # Find vertices we haven't processed yet
        new_picks = []
        for v in picked:
            if v.index not in seen_indices:
                new_picks.append(v)

        if not new_picks:
            continue

        # Process each new vertex
        for v in new_picks:
            idx = v.index
            coord = np.asarray(v.coord)
            seen_indices.add(idx)

            if idx in control_points:
                old = control_points[idx]
                print(f"\n  *** Vertex #{idx} already picked! ***")
                print(f"      At ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})")
                print(f"      Previous: n_cells={old['n_cells']}, thickness={old['thickness']}")
                print(f"      Overriding with new values...")
            else:
                print(f"\n  Picked vertex #{idx}: ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})")

            # Get n_cells
            while True:
                try:
                    nc = input(f"    n_cells [5.0]: ").strip()
                    n_cells = float(nc) if nc else 5.0
                    if n_cells <= 0:
                        print("    Must be positive.")
                        continue
                    break
                except ValueError:
                    print("    Invalid number, try again.")

            # Get thickness
            while True:
                try:
                    th = input(f"    thickness [0.1]: ").strip()
                    thickness = float(th) if th else 0.1
                    if thickness <= 0:
                        print("    Must be positive.")
                        continue
                    break
                except ValueError:
                    print("    Invalid number, try again.")

            control_points[idx] = {
                'coord': coord,
                'n_cells': n_cells,
                'thickness': thickness,
            }
            if idx not in pick_order:
                pick_order.append(idx)

            total = len(control_points)
            print(f"    ✓ Point set: n_cells={n_cells}, thickness={thickness}  (total: {total} points)")
            print(f"\n  Pick another vertex or close window to finish...")

    vis.destroy_window()

    # Save
    if not control_points:
        print("\nNo points picked. Nothing to save.")
        return

    with open(output_path, 'w') as f:
        f.write(f"# Control points for geodesic TPMS\n")
        f.write(f"# Picked from: {mesh_path}\n")
        f.write(f"# n_cells  x  y  z  thickness\n")
        for idx in pick_order:
            p = control_points[idx]
            c = p['coord']
            f.write(f"{p['n_cells']}  {c[0]:.6f}  {c[1]:.6f}  {c[2]:.6f}  {p['thickness']}\n")

    print(f"\nSaved {len(control_points)} control points to {output_path}")
    print("Points:")
    for i, idx in enumerate(pick_order):
        p = control_points[idx]
        c = p['coord']
        print(f"  {i+1}. ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})  n_cells={p['n_cells']}  thickness={p['thickness']}")


if __name__ == "__main__":
    main()
