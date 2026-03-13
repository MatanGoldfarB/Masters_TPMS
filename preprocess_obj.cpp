// Preprocess an OBJ: polygon-soup load, repair, orient, triangulate, close holes
// Output: a watertight triangle mesh OBJ ready for tpms_unified
#define CGAL_NO_PRECONDITIONS
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/OBJ.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point;
typedef CGAL::Surface_mesh<Point> SurfaceMesh;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.obj output.obj" << std::endl;
        return 1;
    }
    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    // 1. Read as polygon soup (tolerates non-manifold input)
    std::vector<Point> points;
    std::vector<std::vector<std::size_t>> polygons;
    std::ifstream in(input_path);
    if (!in) { std::cerr << "Cannot open " << input_path << std::endl; return 1; }
    if (!CGAL::IO::read_OBJ(in, points, polygons)) {
        std::cerr << "Failed to read OBJ" << std::endl; return 1;
    }
    in.close();
    std::cout << "Read: " << points.size() << " points, " << polygons.size() << " polygons" << std::endl;

    // 2. Repair soup (merge duplicates, remove degenerate polygons)
    CGAL::Polygon_mesh_processing::repair_polygon_soup(points, polygons);
    std::cout << "After repair: " << points.size() << " points, " << polygons.size() << " polygons" << std::endl;

    // 3. Orient soup
    CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);

    // 4. Convert to surface mesh
    SurfaceMesh mesh;
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, polygons, mesh);

    // 5. Triangulate
    CGAL::Polygon_mesh_processing::triangulate_faces(mesh);
    std::cout << "Triangulated: " << mesh.number_of_vertices() << " verts, "
              << mesh.number_of_faces() << " faces" << std::endl;

    // 6. Stitch borders (merge coincident boundary edges)
    CGAL::Polygon_mesh_processing::stitch_borders(mesh);
    std::cout << "After stitch: " << mesh.number_of_vertices() << " verts, "
              << mesh.number_of_faces() << " faces" << std::endl;

    // 7. Close holes
    std::vector<SurfaceMesh::Halfedge_index> border_cycles;
    CGAL::Polygon_mesh_processing::extract_boundary_cycles(mesh, std::back_inserter(border_cycles));
    std::cout << "Border cycles (holes): " << border_cycles.size() << std::endl;
    int closed = 0;
    for (auto h : border_cycles) {
        // Count edges in this boundary cycle
        int len = 0;
        auto start = h;
        auto cur = h;
        do { len++; cur = mesh.next(cur); } while (cur != start);
        // Only close reasonably sized holes
        if (len <= 500) {
            try {
                std::vector<SurfaceMesh::Face_index> new_faces;
                CGAL::Polygon_mesh_processing::triangulate_hole(mesh, h, std::back_inserter(new_faces));
                closed++;
            } catch (...) {
                std::cout << "  Failed to close hole with " << len << " edges (skipping)" << std::endl;
            }
        } else {
            std::cout << "  Skipping large hole with " << len << " edges" << std::endl;
        }
    }
    std::cout << "Closed " << closed << "/" << border_cycles.size() << " holes" << std::endl;

    // Check final state
    std::cout << "Final mesh: " << mesh.number_of_vertices() << " verts, "
              << mesh.number_of_faces() << " faces" << std::endl;
    std::cout << "is_valid: " << mesh.is_valid() << std::endl;
    std::cout << "is_closed: " << CGAL::is_closed(mesh) << std::endl;
    std::cout << "is_triangle_mesh: " << CGAL::is_triangle_mesh(mesh) << std::endl;

    // 8. Write output
    if (!CGAL::IO::write_polygon_mesh(output_path, mesh)) {
        std::cerr << "Failed to write " << output_path << std::endl;
        return 1;
    }
    std::cout << "Written to " << output_path << std::endl;
    return 0;
}
