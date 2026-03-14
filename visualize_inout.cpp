#include <CGAL/Simple_cartesian.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/bounding_box.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel K;
typedef CGAL::Surface_mesh<Point> SurfaceMesh;

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " input_mesh [output.ply]" << std::endl;
		std::cerr << "  Visualize inside/outside classification of voxel grid points." << std::endl;
		std::cerr << "  Green = inside, Red = outside, Blue = on boundary" << std::endl;
		return 1;
	}

	std::string input_path = argv[1];
	std::string output_path = (argc >= 3) ? argv[2] : "inout_vis.ply";

	// --- Load mesh ---
	SurfaceMesh mesh;
	if (!CGAL::IO::read_polygon_mesh(input_path, mesh) || mesh.is_empty()) {
		std::cerr << "Error: could not read mesh from " << input_path << std::endl;
		return 1;
	}
	if (!CGAL::is_triangle_mesh(mesh)) {
		std::cerr << "Error: input is not a triangle mesh" << std::endl;
		return 1;
	}
	std::cout << "Loaded mesh: " << num_vertices(mesh) << " vertices, "
	          << num_faces(mesh) << " faces" << std::endl;

	// Inside test
	CGAL::Side_of_triangle_mesh<SurfaceMesh, K> inside(mesh);

	// Bounding box
	K::Iso_cuboid_3 bbox = CGAL::bounding_box(mesh.points().begin(), mesh.points().end());
	double xmin = bbox.xmin(), xmax = bbox.xmax();
	double ymin = bbox.ymin(), ymax = bbox.ymax();
	double zmin = bbox.zmin(), zmax = bbox.zmax();
	double xgap = xmax - xmin;
	double ygap = ymax - ymin;
	double zgap = zmax - zmin;

	std::cout << "Bounding box:" << std::endl;
	std::cout << "  X: [" << xmin << ", " << xmax << "]  (" << xgap << ")" << std::endl;
	std::cout << "  Y: [" << ymin << ", " << ymax << "]  (" << ygap << ")" << std::endl;
	std::cout << "  Z: [" << zmin << ", " << zmax << "]  (" << zgap << ")" << std::endl;

	// --- Resolution input (same as tpms_unified) ---
	std::cout << "Input resolution multiplier (voxels per unit length): ";
	double resolution;
	std::cin >> resolution;

	int x_num = static_cast<int>(xgap * resolution);
	int y_num = static_cast<int>(ygap * resolution);
	int z_num = static_cast<int>(zgap * resolution);
	x_num = std::max(x_num, 2);
	y_num = std::max(y_num, 2);
	z_num = std::max(z_num, 2);
	long long total = (long long)x_num * y_num * z_num;
	std::cout << "  Grid: " << x_num << " x " << y_num << " x " << z_num
	          << " = " << total << " voxels" << std::endl;

	// Slight padding (same as tpms_unified)
	double px0 = xmin - 0.1, px1 = xmax + 0.1;
	double py0 = ymin - 0.1, py1 = ymax + 0.1;
	double pz0 = zmin - 0.1, pz1 = zmax + 0.1;

	// --- Write PLY header with known vertex count, then stream points ---
	std::ofstream ply(output_path);
	if (!ply.is_open()) {
		std::cerr << "Error: could not open " << output_path << " for writing" << std::endl;
		return 1;
	}

	ply << "ply\n";
	ply << "format ascii 1.0\n";
	ply << "element vertex " << total << "\n";
	ply << "property float x\n";
	ply << "property float y\n";
	ply << "property float z\n";
	ply << "property uchar red\n";
	ply << "property uchar green\n";
	ply << "property uchar blue\n";
	ply << "end_header\n";

	long long inside_count = 0, outside_count = 0, boundary_count = 0;
	long long done = 0;

	for (int iz = 0; iz < z_num; iz++) {
		double z = pz0 + (double)iz / (double)(z_num - 1) * (pz1 - pz0);
		for (int iy = 0; iy < y_num; iy++) {
			double y = py0 + (double)iy / (double)(y_num - 1) * (py1 - py0);
			for (int ix = 0; ix < x_num; ix++) {
				double x = px0 + (double)ix / (double)(x_num - 1) * (px1 - px0);
				Point q(x, y, z);
				CGAL::Bounded_side side = inside(q);

				int r, g, b;
				if (side == CGAL::ON_BOUNDED_SIDE) {
					r = 0; g = 200; b = 0;
					inside_count++;
				} else if (side == CGAL::ON_BOUNDARY) {
					r = 0; g = 0; b = 255;
					boundary_count++;
				} else {
					r = 200; g = 0; b = 0;
					outside_count++;
				}

				ply << x << " " << y << " " << z << " "
				    << r << " " << g << " " << b << "\n";

				done++;
				if (done % 100000 == 0 || done == total) {
					double pct = 100.0 * done / total;
					std::cout << "\r  Classifying: " << (int)pct << "% (" << done << "/" << total << ")"
					          << std::flush;
				}
			}
		}
	}
	ply.close();
	std::cout << std::endl;

	std::cout << "Results:" << std::endl;
	std::cout << "  Inside:   " << inside_count << std::endl;
	std::cout << "  Outside:  " << outside_count << std::endl;
	std::cout << "  Boundary: " << boundary_count << std::endl;
	std::cout << "Written " << total << " colored points to " << output_path << std::endl;

	// Launch matplotlib plot automatically
	std::string cmd = "python3 plot_inout.py " + output_path + " " + input_path;
	std::cout << "Launching plot..." << std::endl;
	system(cmd.c_str());

	return 0;
}
