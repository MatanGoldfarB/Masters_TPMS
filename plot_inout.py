#!/usr/bin/env python3
"""Plot colored PLY point cloud + input mesh overlay from visualize_inout."""
import sys
import numpy as np

def load_ply(path):
    """Stream-load PLY without holding raw text in memory."""
    with open(path, 'r') as f:
        line = f.readline().strip()
        assert line == 'ply', f"Not a PLY file: {line}"
        n_verts = 0
        while True:
            line = f.readline().strip()
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
            if line == 'end_header':
                break
        points = np.zeros((n_verts, 3))
        colors = np.zeros((n_verts, 3))
        for i in range(n_verts):
            parts = f.readline().split()
            points[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
            colors[i] = [int(parts[3])/255.0, int(parts[4])/255.0, int(parts[5])/255.0]
    return points, colors

def load_mesh(path):
    """Load mesh vertices and triangle faces from OBJ/OFF/STL/PLY via trimesh."""
    try:
        import trimesh
        m = trimesh.load(path, force='mesh')
        return np.array(m.vertices), np.array(m.faces)
    except ImportError:
        print("Warning: trimesh not installed, skipping mesh overlay.")
        print("  Install with: pip3 install trimesh")
        return None, None
    except Exception as e:
        print(f"Warning: could not load mesh {path}: {e}")
        return None, None

def main():
    ply_path = sys.argv[1] if len(sys.argv) > 1 else 'inout_vis.ply'
    mesh_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading {ply_path}...")
    points, colors = load_ply(ply_path)
    print(f"Loaded {len(points)} points")

    # Load input mesh for overlay
    mesh_verts, mesh_faces = None, None
    if mesh_path:
        print(f"Loading mesh {mesh_path}...")
        mesh_verts, mesh_faces = load_mesh(mesh_path)
        if mesh_verts is not None:
            print(f"Mesh: {len(mesh_verts)} vertices, {len(mesh_faces)} faces")

    # Keep only inside points (green) with y > 20
    inside_mask = (colors[:, 1] > 0) & (colors[:, 0] == 0) & (colors[:, 2] == 0)
    inside_mask &= (points[:, 1] > 20)

    # Also keep outside points (red) in the box [135:155, 75:95, -10:5]
    outside_mask = (colors[:, 0] > 0) & (colors[:, 1] == 0) & (colors[:, 2] == 0)
    outside_mask &= ((points[:, 0] >= 135) & (points[:, 0] <= 155) &
                     (points[:, 1] >= 75) & (points[:, 1] <= 95) &
                     (points[:, 2] >= -10) & (points[:, 2] <= 5))

    combined_mask = inside_mask | outside_mask
    points = points[combined_mask]
    colors = colors[combined_mask]
    print(f"Plotting {len(points)} points (inside y>20 + outside in box)")

    # Subsample if too many points for interactive plotting
    max_plot = 200000
    if len(points) > max_plot:
        idx = np.random.default_rng(42).choice(len(points), max_plot, replace=False)
        points = points[idx]
        colors = colors[idx]
        print(f"Subsampled to {max_plot} points for plotting")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot mesh surface with very low opacity
    if mesh_verts is not None and mesh_faces is not None:
        triangles = mesh_verts[mesh_faces]
        poly = Poly3DCollection(triangles, alpha=0.05, linewidths=0.1,
                                edgecolors=(0.3, 0.3, 0.3, 0.1),
                                facecolors=(0.6, 0.6, 0.8, 0.05))
        ax.add_collection3d(poly)

    # Plot classified points
    ax.scatter(points[:,0], points[:,1], points[:,2],
               c=colors, s=0.5, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Inside (green) / Outside (red) / Boundary (blue)')

    # Equal aspect ratio
    maxr = np.max(np.ptp(points, axis=0)) / 2
    mid = np.mean(points, axis=0)
    ax.set_xlim(mid[0]-maxr, mid[0]+maxr)
    ax.set_ylim(mid[1]-maxr, mid[1]+maxr)
    ax.set_zlim(mid[2]-maxr, mid[2]+maxr)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
