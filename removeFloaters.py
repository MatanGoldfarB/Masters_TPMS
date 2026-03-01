# keep_largest_component.py
# Usage:
#   python removeFloaters.py input.obj output.obj [--min-thickness T] [--fix-thin] [--remove-thin]
#
# Options:
#   --min-thickness T   Minimum wall thickness (default: auto-detect based on mesh)
#   --fix-thin          Attempt to thicken thin regions by moving vertices
#   --remove-thin       Remove faces in thin regions instead of fixing
#   --report-only       Only report thin regions, don't modify mesh

import sys
import argparse
import numpy as np
import trimesh  # type: ignore
from typing import Tuple, List, Optional


def compute_local_thickness(mesh: trimesh.Trimesh, sample_count: int = 5000) -> np.ndarray:
    """
    Compute approximate local thickness at each vertex using ray casting.
    For each vertex, cast rays inward (opposite to vertex normal) and measure
    distance to the opposing surface.
    
    Returns array of thickness values per vertex.
    """
    # Get vertex normals (pointing outward)
    vertex_normals = mesh.vertex_normals
    vertices = mesh.vertices
    
    # Initialize thickness array with inf (will be updated)
    thickness = np.full(len(vertices), np.inf)
    
    # For efficiency, sample a subset of vertices if mesh is large
    if len(vertices) > sample_count:
        sample_indices = np.random.choice(len(vertices), sample_count, replace=False)
    else:
        sample_indices = np.arange(len(vertices))
    
    # Cast rays inward from each sampled vertex
    ray_origins = vertices[sample_indices]
    ray_directions = -vertex_normals[sample_indices]  # Inward direction
    
    # Offset origins slightly to avoid self-intersection
    ray_origins = ray_origins + ray_directions * 1e-6
    
    # Use trimesh ray casting
    try:
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )
        
        if len(locations) > 0:
            # Compute distances
            for i, (loc, ray_idx) in enumerate(zip(locations, index_ray)):
                origin = ray_origins[ray_idx]
                dist = np.linalg.norm(loc - origin)
                vertex_idx = sample_indices[ray_idx]
                thickness[vertex_idx] = min(thickness[vertex_idx], dist)
    except Exception as e:
        print(f"Warning: Ray casting failed: {e}")
        return thickness
    
    # For vertices not sampled, interpolate from neighbors
    if len(vertices) > sample_count:
        thickness = interpolate_thickness(mesh, thickness, sample_indices)
    
    return thickness


def interpolate_thickness(mesh: trimesh.Trimesh, thickness: np.ndarray, 
                          sampled_indices: np.ndarray) -> np.ndarray:
    """
    Interpolate thickness values for non-sampled vertices from their neighbors.
    """
    # Build vertex adjacency
    edges = mesh.edges_unique
    adjacency = {i: set() for i in range(len(mesh.vertices))}
    for e in edges:
        adjacency[e[0]].add(e[1])
        adjacency[e[1]].add(e[0])
    
    sampled_set = set(sampled_indices)
    
    for v_idx in range(len(mesh.vertices)):
        if v_idx not in sampled_set and np.isinf(thickness[v_idx]):
            # Get neighbors that have valid thickness
            neighbors = adjacency[v_idx]
            valid_neighbors = [n for n in neighbors if not np.isinf(thickness[n])]
            if valid_neighbors:
                thickness[v_idx] = np.mean([thickness[n] for n in valid_neighbors])
    
    return thickness


def detect_thin_faces(mesh: trimesh.Trimesh, thickness: np.ndarray, 
                      min_thickness: float) -> np.ndarray:
    """
    Detect faces where any vertex has thickness below minimum.
    Returns boolean mask of thin faces.
    """
    # A face is thin if any of its vertices are thin
    vertex_is_thin = thickness < min_thickness
    
    # For each face, check if any vertex is thin
    thin_faces = np.any(vertex_is_thin[mesh.faces], axis=1)
    
    return thin_faces


def estimate_min_thickness(mesh: trimesh.Trimesh, thickness: np.ndarray) -> float:
    """
    Auto-estimate minimum thickness based on mesh statistics.
    Uses a percentile of the thickness distribution to avoid outliers.
    """
    valid_thickness = thickness[~np.isinf(thickness)]
    if len(valid_thickness) == 0:
        # Fallback: use bounding box diagonal / 100
        return mesh.bounding_box.primitive.extents.min() / 50.0
    
    # Use 10th percentile as a reasonable minimum
    # This catches genuinely thin regions while ignoring measurement noise
    median_thickness = np.median(valid_thickness)
    return median_thickness * 0.3  # 30% of median thickness


def compute_expected_tpms_thickness(tpms_thickness: float, omega: float) -> float:
    """
    Compute expected physical wall thickness from TPMS parameters.
    
    In TPMS generation:
        phi = |f(x,y,z)| - tpms_thickness
    
    where f is the TPMS function (e.g., cos(ωx) + cos(ωy) + cos(ωz) for Schwarz-P).
    
    The gradient magnitude of f determines how fast the function changes in space.
    For Schwarz-P: |∇f| ≈ ω√3 at the zero-crossings.
    
    Physical thickness ≈ 2 * tpms_thickness / |∇f| ≈ 2 * tpms_thickness / (ω * √3)
    
    We use a slightly smaller factor (0.8) to be conservative about detecting thin regions.
    """
    # For Schwarz-P type TPMS, gradient magnitude at zero-crossing ≈ ω * sqrt(3)
    # But this varies by position. Use ω as approximation.
    expected_thickness = 2.0 * tpms_thickness / omega
    
    # Return 80% of expected as threshold to catch regions that are thinner than expected
    return expected_thickness * 0.8


def fix_thin_regions_by_thickening(mesh: trimesh.Trimesh, thickness: np.ndarray,
                                    min_thickness: float, iterations: int = 3) -> trimesh.Trimesh:
    """
    Attempt to fix thin regions by moving vertices outward.
    This inflates thin areas to meet minimum thickness requirement.
    """
    vertices = mesh.vertices.copy()
    vertex_normals = mesh.vertex_normals
    
    for iteration in range(iterations):
        # Find vertices that are too thin
        thin_mask = (thickness < min_thickness) & ~np.isinf(thickness)
        
        if not np.any(thin_mask):
            break
        
        # Calculate how much to move each thin vertex
        deficit = min_thickness - thickness[thin_mask]
        deficit = np.clip(deficit, 0, min_thickness * 0.5)  # Limit movement per iteration
        
        # Move vertices outward along their normals
        # Move by half the deficit (since both sides should move)
        displacement = vertex_normals[thin_mask] * (deficit[:, np.newaxis] * 0.5)
        vertices[thin_mask] += displacement
        
        # Update mesh and recompute thickness
        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
        mesh.fix_normals()
        thickness = compute_local_thickness(mesh)
        
        print(f"  Thickening iteration {iteration + 1}: {np.sum(thin_mask)} thin vertices adjusted")
    
    return mesh


def remove_thin_faces(mesh: trimesh.Trimesh, thin_faces: np.ndarray) -> trimesh.Trimesh:
    """
    Remove faces that are in thin regions.
    """
    # Keep faces that are NOT thin
    keep_mask = ~thin_faces
    new_faces = mesh.faces[keep_mask]
    
    # Create new mesh
    new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=new_faces)
    new_mesh.remove_unreferenced_vertices()
    new_mesh.update_faces(new_mesh.nondegenerate_faces())
    
    return new_mesh


def analyze_and_fix_thin_regions(mesh: trimesh.Trimesh, 
                                  min_thickness: Optional[float] = None,
                                  fix_method: str = 'none',
                                  verbose: bool = True) -> Tuple[trimesh.Trimesh, dict]:
    """
    Analyze mesh for thin regions and optionally fix them.
    
    Args:
        mesh: Input trimesh
        min_thickness: Minimum acceptable thickness (auto-detected if None)
        fix_method: 'none', 'thicken', or 'remove'
        verbose: Print progress info
    
    Returns:
        Tuple of (processed_mesh, statistics_dict)
    """
    stats = {}
    
    if verbose:
        print("Computing local thickness...")
    
    thickness = compute_local_thickness(mesh)
    valid_thickness = thickness[~np.isinf(thickness)]
    
    if len(valid_thickness) == 0:
        if verbose:
            print("Warning: Could not compute thickness (ray casting failed)")
        stats['error'] = 'thickness_computation_failed'
        return mesh, stats
    
    # Auto-detect min thickness if not provided
    if min_thickness is None:
        min_thickness = estimate_min_thickness(mesh, thickness)
        if verbose:
            print(f"Auto-detected minimum thickness threshold: {min_thickness:.4f}")
    
    stats['min_thickness_threshold'] = min_thickness
    stats['thickness_min'] = float(np.min(valid_thickness))
    stats['thickness_max'] = float(np.max(valid_thickness))
    stats['thickness_mean'] = float(np.mean(valid_thickness))
    stats['thickness_median'] = float(np.median(valid_thickness))
    
    # Detect thin faces
    thin_faces = detect_thin_faces(mesh, thickness, min_thickness)
    num_thin_faces = np.sum(thin_faces)
    thin_ratio = num_thin_faces / len(mesh.faces) * 100
    
    stats['thin_faces_count'] = int(num_thin_faces)
    stats['thin_faces_ratio'] = float(thin_ratio)
    stats['total_faces'] = len(mesh.faces)
    
    if verbose:
        print(f"Thickness stats: min={stats['thickness_min']:.4f}, "
              f"max={stats['thickness_max']:.4f}, mean={stats['thickness_mean']:.4f}")
        print(f"Thin faces: {num_thin_faces} / {len(mesh.faces)} ({thin_ratio:.1f}%)")
    
    # Apply fix if requested
    if fix_method == 'thicken' and num_thin_faces > 0:
        if verbose:
            print("Attempting to thicken thin regions...")
        mesh = fix_thin_regions_by_thickening(mesh, thickness, min_thickness)
        stats['fix_applied'] = 'thicken'
        
    elif fix_method == 'remove' and num_thin_faces > 0:
        if verbose:
            print(f"Removing {num_thin_faces} thin faces...")
        mesh = remove_thin_faces(mesh, thin_faces)
        stats['fix_applied'] = 'remove'
        stats['faces_after_removal'] = len(mesh.faces)
    else:
        stats['fix_applied'] = 'none'
    
    return mesh, stats


def keep_largest_component(mesh: trimesh.Trimesh, verbose: bool = True) -> trimesh.Trimesh:
    """
    Keep only the largest connected component of the mesh.
    """
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty.")

    # Split into connected components (by face adjacency)
    parts = mesh.split(only_watertight=False)

    if len(parts) == 0:
        raise ValueError("No components found.")

    # Pick the largest component.
    def score(m: trimesh.Trimesh) -> float:
        if m.is_watertight:
            return abs(m.volume)
        return float(m.faces.shape[0])

    largest = max(parts, key=score)

    # Clean up
    largest.remove_unreferenced_vertices()
    largest.update_faces(largest.nondegenerate_faces())

    if verbose:
        print(f"Components: {len(parts)} | Kept faces: {largest.faces.shape[0]}")
    
    return largest


def process_mesh(in_path: str, out_path: str, 
                 min_thickness: Optional[float] = None,
                 fix_method: str = 'none',
                 keep_largest: bool = True,
                 verbose: bool = True) -> dict:
    """
    Full pipeline: load mesh, remove floaters, analyze/fix thin regions, save.
    
    Args:
        in_path: Input mesh file path
        out_path: Output mesh file path
        min_thickness: Minimum wall thickness (None for auto-detect)
        fix_method: 'none', 'thicken', or 'remove'
        keep_largest: Whether to keep only largest component
        verbose: Print progress
    
    Returns:
        Dictionary with processing statistics
    """
    if verbose:
        print(f"Loading mesh: {in_path}")
    
    mesh = trimesh.load(in_path, force='mesh')
    stats = {'input_faces': len(mesh.faces), 'input_vertices': len(mesh.vertices)}
    
    # Step 1: Keep largest component (remove floaters)
    if keep_largest:
        mesh = keep_largest_component(mesh, verbose=verbose)
        stats['after_floater_removal_faces'] = len(mesh.faces)
    
    # Step 2: Analyze and fix thin regions
    mesh, thin_stats = analyze_and_fix_thin_regions(
        mesh, 
        min_thickness=min_thickness,
        fix_method=fix_method,
        verbose=verbose
    )
    stats.update(thin_stats)
    
    # Step 3: Final cleanup - keep largest component again if we removed faces
    if fix_method == 'remove' and stats.get('thin_faces_count', 0) > 0:
        if verbose:
            print("Running final floater removal after thin face removal...")
        mesh = keep_largest_component(mesh, verbose=verbose)
    
    stats['output_faces'] = len(mesh.faces)
    stats['output_vertices'] = len(mesh.vertices)
    
    # Export
    mesh.export(out_path)
    if verbose:
        print(f"Wrote: {out_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Remove floaters and fix thin regions in TPMS meshes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic floater removal (original behavior)
  python removeFloaters.py input.obj output.obj
  
  # Analyze thin regions without fixing
  python removeFloaters.py input.obj output.obj --report-only
  
  # Fix thin regions by thickening
  python removeFloaters.py input.obj output.obj --fix-thin --min-thickness 0.5
  
  # Remove thin faces instead of thickening
  python removeFloaters.py input.obj output.obj --remove-thin --min-thickness 0.3
  
  # Use TPMS parameters to compute expected thickness (recommended)
  # If you used thickness=0.28 and omega=0.8 in TPMS_matan:
  python removeFloaters.py input.obj output.obj --fix-thin --tpms-thickness 0.28 --tpms-omega 0.8
        """
    )
    
    parser.add_argument('input', help='Input mesh file (OBJ, STL, etc.)')
    parser.add_argument('output', help='Output mesh file')
    parser.add_argument('--min-thickness', type=float, default=None,
                        help='Minimum wall thickness (auto-detected if not specified)')
    parser.add_argument('--tpms-thickness', type=float, default=None,
                        help='TPMS thickness parameter (the value you input in TPMS_matan)')
    parser.add_argument('--tpms-omega', type=float, default=None,
                        help='TPMS frequency parameter (omega)')
    parser.add_argument('--fix-thin', action='store_true',
                        help='Attempt to thicken thin regions')
    parser.add_argument('--remove-thin', action='store_true',
                        help='Remove faces in thin regions')
    parser.add_argument('--report-only', action='store_true',
                        help='Only report thin regions, do not modify mesh')
    parser.add_argument('--no-floater-removal', action='store_true',
                        help='Skip floater removal step')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Determine minimum thickness
    min_thickness = args.min_thickness
    if min_thickness is None and args.tpms_thickness is not None and args.tpms_omega is not None:
        # Compute from TPMS parameters
        min_thickness = compute_expected_tpms_thickness(args.tpms_thickness, args.tpms_omega)
        if not args.quiet:
            print(f"Computed min thickness from TPMS params: {min_thickness:.4f}")
            print(f"  (tpms_thickness={args.tpms_thickness}, omega={args.tpms_omega})")
    
    # Determine fix method
    if args.fix_thin and args.remove_thin:
        print("Error: Cannot use both --fix-thin and --remove-thin")
        sys.exit(1)
    
    if args.fix_thin:
        fix_method = 'thicken'
    elif args.remove_thin:
        fix_method = 'remove'
    else:
        fix_method = 'none'
    
    # If report-only, just analyze without fixing
    if args.report_only:
        fix_method = 'none'
    
    stats = process_mesh(
        in_path=args.input,
        out_path=args.output,
        min_thickness=min_thickness,
        fix_method=fix_method,
        keep_largest=not args.no_floater_removal,
        verbose=not args.quiet
    )
    
    if not args.quiet:
        print("\n--- Summary ---")
        print(f"Input:  {stats['input_faces']} faces, {stats['input_vertices']} vertices")
        print(f"Output: {stats['output_faces']} faces, {stats['output_vertices']} vertices")
        if 'thin_faces_count' in stats:
            print(f"Thin regions: {stats['thin_faces_count']} faces ({stats['thin_faces_ratio']:.1f}%)")
            if stats.get('fix_applied') != 'none':
                print(f"Fix applied: {stats['fix_applied']}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and not sys.argv[1].startswith('-'):
        # Backwards compatible: simple two-argument mode
        stats = process_mesh(sys.argv[1], sys.argv[2], fix_method='none')
    else:
        main()