# keep_largest_component.py
# Usage:
#   python keep_largest_component.py input.obj output.obj

import sys
import trimesh

def keep_largest_component(in_path: str, out_path: str) -> None:
    mesh = trimesh.load(in_path, force='mesh')

    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty.")

    # Split into connected components (by face adjacency)
    parts = mesh.split(only_watertight=False)

    if len(parts) == 0:
        raise ValueError("No components found.")

    # Pick the largest component by face count (most robust for both solid and sheet meshes).
    def score(m: trimesh.Trimesh) -> float:
        return float(m.faces.shape[0])

    largest = max(parts, key=score)

    # Optional: clean up
    largest.remove_unreferenced_vertices()
    # largest.remove_degenerate_faces()  # skip — too aggressive on thin TPMS sheets

    largest.export(out_path)
    print(f"Components: {len(parts)} | Kept faces: {largest.faces.shape[0]} | Wrote: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python keep_largest_component.py input.obj output.obj")
        sys.exit(1)

    keep_largest_component(sys.argv[1], sys.argv[2])