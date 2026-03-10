#include <CGAL/Simple_cartesian.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>

#include <igl/copyleft/marching_cubes.h>
#include <igl/writeSTL.h>

#include <iostream>
#include <string>
#include <cmath>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel K;
typedef CGAL::Surface_mesh<Point> SurfaceMesh;
typedef CGAL::AABB_face_graph_triangle_primitive<SurfaceMesh> Primitive;
typedef CGAL::AABB_traits_3<K, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> Tree;

enum TpmsType { kP = 0, kD = 1, kG = 2 };

double tpms_function(TpmsType type, double omega, double x, double y, double z)
{
	switch (type) {
	case kP:
		return cos(omega * x) + cos(omega * y) + cos(omega * z);
	case kD:
		return cos(omega * x) * cos(omega * y) * cos(omega * z)
		     - sin(omega * x) * sin(omega * y) * sin(omega * z);
	case kG:
		return sin(omega * x) * cos(omega * y)
		     + sin(omega * y) * cos(omega * z)
		     + sin(omega * z) * cos(omega * x);
	}
	return 0.0;
}

void generate(const std::string& input_path, const std::string& output_path)
{
	// --- Load mesh ---
	SurfaceMesh mesh;
	if (!CGAL::IO::read_polygon_mesh(input_path, mesh) || mesh.is_empty()) {
		std::cerr << "Error: could not read mesh from " << input_path << std::endl;
		return;
	}
	if (!CGAL::is_triangle_mesh(mesh)) {
		std::cerr << "Error: input is not a triangle mesh" << std::endl;
		return;
	}

	// Inside test
	CGAL::Side_of_triangle_mesh<SurfaceMesh, K> inside(mesh);

	// AABB tree for distance queries (build once, queries are thread-safe)
	Tree tree(faces(mesh).first, faces(mesh).second, mesh);
	tree.accelerate_distance_queries();

	// Bounding box
	K::Iso_cuboid_3 bbox = CGAL::bounding_box(mesh.points().begin(), mesh.points().end());
	double xmin = bbox.xmin(), xmax = bbox.xmax();
	double ymin = bbox.ymin(), ymax = bbox.ymax();
	double zmin = bbox.zmin(), zmax = bbox.zmax();
	double xgap = xmax - xmin;
	double ygap = ymax - ymin;
	double zgap = zmax - zmin;
	double max_dim = std::max({xgap, ygap, zgap});

	std::cout << "Bounding box:" << std::endl;
	std::cout << "  X: [" << xmin << ", " << xmax << "]  (" << xgap << ")" << std::endl;
	std::cout << "  Y: [" << ymin << ", " << ymax << "]  (" << ygap << ")" << std::endl;
	std::cout << "  Z: [" << zmin << ", " << zmax << "]  (" << zgap << ")" << std::endl;

	// --- Interactive parameters ---
	std::cout << "Input TPMS type (0=kP, 1=kD, 2=kG): ";
	int tpms_input;
	std::cin >> tpms_input;
	if (tpms_input < 0 || tpms_input > 2) {
		std::cerr << "Error: invalid TPMS type" << std::endl;
		return;
	}
	TpmsType tpms_type = static_cast<TpmsType>(tpms_input);

	std::cout << "Input resolution multiplier (voxels per unit length): ";
	double resolution;
	std::cin >> resolution;
	int x_num = static_cast<int>(xgap * resolution);
	int y_num = static_cast<int>(ygap * resolution);
	int z_num = static_cast<int>(zgap * resolution);
	// Ensure at least 2 voxels per axis
	x_num = std::max(x_num, 2);
	y_num = std::max(y_num, 2);
	z_num = std::max(z_num, 2);
	std::cout << "  Grid: " << x_num << " x " << y_num << " x " << z_num
	          << " = " << (long long)x_num * y_num * z_num << " voxels" << std::endl;

	std::cout << "Input wall thickness (full thickness in model units): ";
	double thickness;
	std::cin >> thickness;

	std::cout << "Input number of unit cells: ";
	double n_cells;
	std::cin >> n_cells;
	double omega = n_cells * 2.0 * M_PI / max_dim;
	std::cout << "  omega = " << omega << std::endl;

	// Convert physical thickness to function-space threshold
	// Near the zero-isosurface, |nabla f| ~ omega, so function-space offset = physical * omega
	double half_t = (thickness / 2.0) * omega;

	// Smoothing factor for R-function boundary transition
	double wavelength = 2.0 * M_PI / omega;
	double smf = wavelength / 10.0;

	// --- Build voxel grid ---
	Eigen::RowVector3d Vmin = { xmin - 0.1, ymin - 0.1, zmin - 0.1 };
	Eigen::RowVector3d Vmax = { xmax + 0.1, ymax + 0.1, zmax + 0.1 };
	Eigen::RowVector3i res = { x_num, y_num, z_num };

	Eigen::MatrixXd grid(res(0) * res(1) * res(2), 3);

	const auto lerp = [&](const int di, const int d) -> double {
		return Vmin(d) + (double)di / (double)(res(d) - 1) * (Vmax(d) - Vmin(d));
	};

#pragma omp parallel for
	for (int zi = 0; zi < res(2); zi++) {
		const double z = lerp(zi, 2);
		for (int yi = 0; yi < res(1); yi++) {
			const double y = lerp(yi, 1);
			for (int xi = 0; xi < res(0); xi++) {
				const double x = lerp(xi, 0);
				grid.row(xi + res(0) * (yi + res(1) * zi)) = Eigen::RowVector3d(x, y, z);
			}
		}
	}

	// --- Compute scalar field ---
	Eigen::VectorXd B(grid.rows());
	int negcount = 0;

#pragma omp parallel for reduction(+:negcount)
	for (int i = 0; i < grid.rows(); i++) {
		const double x = grid(i, 0), y = grid(i, 1), z = grid(i, 2);
		Point q(x, y, z);

		// TPMS shell field: negative = inside wall
		double f = tpms_function(tpms_type, omega, x, y, z);
		double phi_tpms = std::abs(f) - half_t;

		// Signed distance to mesh boundary: positive inside, negative outside
		Point closest = tree.closest_point(q);
		double unsigned_d = std::sqrt(CGAL::squared_distance(q, closest));
		CGAL::Bounded_side side = inside(q);
		double d = (side == CGAL::ON_BOUNDED_SIDE) ? unsigned_d : -unsigned_d;

		// R-function smooth-max: max(phi_tpms, -d)
		// Combines TPMS shell with boundary clipping via smooth transition
		double phi;
		if (d < -3.0 * smf) {
			// Far outside mesh — hard clip
			phi = 1000.0;
		} else {
			phi = smf * std::log(std::exp(phi_tpms / smf) + std::exp(-d / smf));
		}

		B(i) = phi;
		if (phi < 0.0) negcount++;
	}

	double vof = (double)negcount / (double)grid.rows();
	std::cout << "Volume fraction: " << vof << std::endl;

	// --- Marching cubes + output ---
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	igl::copyleft::marching_cubes(B, grid, res(0), res(1), res(2), V, F);

	std::cout << "Output mesh: " << V.rows() << " vertices, " << F.rows() << " faces" << std::endl;

	igl::writeSTL(output_path, V, F, igl::FileEncoding::Binary);
	std::cout << "Written to " << output_path << std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " input_mesh output.stl" << std::endl;
		std::cerr << "  Supported input formats: .off, .obj, .stl, .ply" << std::endl;
		return 1;
	}
	generate(argv[1], argv[2]);
	return 0;
}
