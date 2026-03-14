#!/usr/bin/env python3
"""
Generate a tetrahedral fluid-domain mesh from a TPMS solid STL.

Workflow:
  1. (Optional) Decimate the STL to reduce triangle count
  2. Import STL mesh, classify surfaces, create topology
  3. Build a bounding box (GEO kernel), define fluid volume as box minus TPMS
  4. Tag physical surfaces: inlet, outlet, side walls, TPMS wall
  5. Mesh with tetrahedra (refined near TPMS walls)
  6. Export as .msh with physical group tags

Usage:
  python mesh_fluid_domain.py tpms_models/output.stl \\
      --out tetrahedral/output_fluid.msh \\
      --lc 0.08 --refine 0.04 --pad 0.05

Requirements: gmsh, open3d (for decimation of large meshes)
"""

import argparse
import math
import os
import sys
import tempfile
import gmsh


def decimate_stl(stl_path, target_faces, output_path):
    """Decimate STL using open3d quadric decimation."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(stl_path)
    n_orig = len(mesh.triangles)
    if n_orig <= target_faces:
        print(f"  STL has {n_orig} triangles (<= {target_faces}), skipping decimation")
        return stl_path

    print(f"  Decimating {n_orig} -> {target_faces} triangles ...")
    mesh = mesh.simplify_quadric_decimation(target_faces)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"  Decimated STL written to {output_path}")
    return output_path


def create_box_geo(xmin, ymin, zmin, xmax, ymax, zmax, lc):
    """Create a box using GEO kernel primitives. Returns (surface_loop, face_dict)."""
    geo = gmsh.model.geo

    # 8 corners: 0=LBN, 1=RBN, 2=RFN, 3=LFN, 4=LBF, 5=RBF, 6=RFF, 7=LFF
    # (L/R=x, B/F=y, N/F=z)
    corners = [
        (xmin, ymin, zmin), (xmax, ymin, zmin),
        (xmax, ymax, zmin), (xmin, ymax, zmin),
        (xmin, ymin, zmax), (xmax, ymin, zmax),
        (xmax, ymax, zmax), (xmin, ymax, zmax),
    ]
    p = [geo.addPoint(*c, lc) for c in corners]

    # Bottom edges (z=zmin): 0->1, 1->2, 2->3, 3->0
    eb = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    # Top edges (z=zmax): 4->5, 5->6, 6->7, 7->4
    et = [geo.addLine(p[4 + i], p[4 + (i + 1) % 4]) for i in range(4)]
    # Vertical edges: 0->4, 1->5, 2->6, 3->7
    ev = [geo.addLine(p[i], p[i + 4]) for i in range(4)]

    # Bottom face (z=zmin)
    ll = geo.addCurveLoop([eb[0], eb[1], eb[2], eb[3]])
    f_zmin = geo.addPlaneSurface([ll])

    # Top face (z=zmax)
    ll = geo.addCurveLoop([et[0], et[1], et[2], et[3]])
    f_zmax = geo.addPlaneSurface([ll])

    # 4 side faces
    f_ymin = geo.addPlaneSurface([geo.addCurveLoop([eb[0], ev[1], -et[0], -ev[0]])])
    f_xmax = geo.addPlaneSurface([geo.addCurveLoop([eb[1], ev[2], -et[1], -ev[1]])])
    f_ymax = geo.addPlaneSurface([geo.addCurveLoop([eb[2], ev[3], -et[2], -ev[2]])])
    f_xmin = geo.addPlaneSurface([geo.addCurveLoop([eb[3], ev[0], -et[3], -ev[3]])])

    all_faces = [f_zmin, f_zmax, f_ymin, f_xmax, f_ymax, f_xmin]
    sl = geo.addSurfaceLoop(all_faces)

    faces = {
        "xmin": f_xmin, "xmax": f_xmax,
        "ymin": f_ymin, "ymax": f_ymax,
        "zmin": f_zmin, "zmax": f_zmax,
    }
    return sl, faces


def main():
    parser = argparse.ArgumentParser(
        description="Generate fluid-domain tetrahedral mesh from TPMS solid STL"
    )
    parser.add_argument("stl", help="Path to TPMS solid STL file")
    parser.add_argument("--out", default=None,
                        help="Output .msh path (default: <stl_stem>_fluid.msh)")
    parser.add_argument("--lc", type=float, default=0.08,
                        help="Global characteristic length (mesh size)")
    parser.add_argument("--refine", type=float, default=0.04,
                        help="Mesh size near TPMS walls (refinement)")
    parser.add_argument("--pad", type=float, default=0.05,
                        help="Padding around TPMS bounding box")
    parser.add_argument("--flow-axis", choices=["x", "y", "z"], default="z",
                        help="Flow direction: inlet at min, outlet at max (default: z)")
    parser.add_argument("--decimate", type=int, default=200000,
                        help="Target triangle count for decimation (0 = no decimation)")
    parser.add_argument("--gui", action="store_true",
                        help="Open Gmsh GUI after meshing")
    args = parser.parse_args()

    if args.out is None:
        stem = args.stl.rsplit(".", 1)[0]
        args.out = stem + "_fluid.msh"

    flow_axis = args.flow_axis
    stl_path = args.stl

    # ── 1. Decimate large STLs ────────────────────────────────────────
    tmp_stl = None
    if args.decimate > 0:
        print(f"[1/6] Checking decimation (target: {args.decimate} faces)")
        tmp_stl = tempfile.NamedTemporaryFile(suffix=".stl", delete=False).name
        stl_path = decimate_stl(args.stl, args.decimate, tmp_stl)
    else:
        print("[1/6] Decimation disabled")

    # ── 2. Import STL and classify surfaces ───────────────────────────
    print(f"[2/6] Importing and classifying: {stl_path}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("fluid_domain")

    gmsh.merge(stl_path)

    # Classify surfaces (forceParametrizable=False avoids the
    # "wrong topology" error on high-genus TPMS surfaces)
    gmsh.model.mesh.classifySurfaces(
        40 * math.pi / 180,  # angle threshold
        True,                # include boundary
        False,               # do NOT force parametrizable patches
        180 * math.pi / 180  # curve angle (no splitting on curvature)
    )
    gmsh.model.mesh.createTopology()

    tpms_surfs = [s[1] for s in gmsh.model.getEntities(2)]
    print(f"  {len(tpms_surfs)} TPMS surface patches")

    # ── 3. Create TPMS surface loop + bounding box ───────────────────
    print("[3/6] Creating bounding box and fluid volume")
    tpms_sl = gmsh.model.geo.addSurfaceLoop(tpms_surfs)

    bb = gmsh.model.getBoundingBox(-1, -1)
    xmin = bb[0] - args.pad
    ymin = bb[1] - args.pad
    zmin = bb[2] - args.pad
    xmax = bb[3] + args.pad
    ymax = bb[4] + args.pad
    zmax = bb[5] + args.pad

    box_sl, box_faces = create_box_geo(xmin, ymin, zmin, xmax, ymax, zmax, args.lc)

    # Fluid volume = box with TPMS solid as hole
    fluid_vol = gmsh.model.geo.addVolume([box_sl, tpms_sl])
    gmsh.model.geo.synchronize()

    print(f"  Bounding box: [{xmin:.3f},{ymin:.3f},{zmin:.3f}] to [{xmax:.3f},{ymax:.3f},{zmax:.3f}]")

    # ── 4. Tag boundary surfaces ──────────────────────────────────────
    print("[4/6] Tagging boundaries")

    inlet_key = flow_axis + "min"
    outlet_key = flow_axis + "max"
    side_keys = [k for k in box_faces if k not in (inlet_key, outlet_key)]

    gmsh.model.addPhysicalGroup(2, [box_faces[inlet_key]], name="inlet")
    gmsh.model.addPhysicalGroup(2, [box_faces[outlet_key]], name="outlet")
    gmsh.model.addPhysicalGroup(2, [box_faces[k] for k in side_keys], name="side_walls")
    gmsh.model.addPhysicalGroup(2, tpms_surfs, name="tpms_wall")
    gmsh.model.addPhysicalGroup(3, [fluid_vol], name="fluid")

    print(f"  inlet:      {inlet_key} face")
    print(f"  outlet:     {outlet_key} face")
    print(f"  side_walls: {len(side_keys)} faces")
    print(f"  tpms_wall:  {len(tpms_surfs)} surface patches")

    # ── 5. Mesh ───────────────────────────────────────────────────────
    print("[5/6] Meshing")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", args.lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.refine * 0.5)

    # Refine near TPMS walls using a distance field
    if tpms_surfs:
        field_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_dist, "SurfacesList", tpms_surfs)
        gmsh.model.mesh.field.setNumber(field_dist, "Sampling", 100)

        field_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_thresh, "InField", field_dist)
        gmsh.model.mesh.field.setNumber(field_thresh, "SizeMin", args.refine)
        gmsh.model.mesh.field.setNumber(field_thresh, "SizeMax", args.lc)
        gmsh.model.mesh.field.setNumber(field_thresh, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(field_thresh, "DistMax", args.pad * 2)

        gmsh.model.mesh.field.setAsBackgroundMesh(field_thresh)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    gmsh.model.mesh.generate(3)

    # Stats
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    _, elem_tags, _ = gmsh.model.mesh.getElements(3)
    n_tets = sum(len(t) for t in elem_tags)
    print(f"  Nodes: {len(node_tags)}, Tetrahedra: {n_tets}")

    # ── 6. Export ─────────────────────────────────────────────────────
    print("[6/6] Exporting")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gmsh.write(args.out)
    print(f"  Written: {args.out}")

    if args.gui:
        gmsh.fltk.run()

    gmsh.finalize()

    if tmp_stl and os.path.exists(tmp_stl):
        os.remove(tmp_stl)

    print("Done!")


if __name__ == "__main__":
    main()
