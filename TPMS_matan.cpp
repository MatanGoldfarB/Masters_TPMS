#include <CGAL/Simple_cartesian.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/bounding_box.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/writeOBJ.h>
#include <fstream>

#include <iostream>
#include <fstream>
#include <string>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
typedef Kernel::Point_3 Point;
typedef Kernel K;
enum TpmsType {
    kP,
    kD,
    kG,
    kL,
    KI_WP,
    kTubularP,
    kTubularG,
    KF_RD,
    P_W,
    N,
	changekP
};



void generateTPMSwithPlate69(const std::string& input_path, const std::string& output_path)
{
	Polyhedron mesh;
	// Checks if the the input was opened correctly, 
	std::ifstream input(input_path);
	// Exit if couldn't open
	if (!input)
	{
		std::cerr << "Input is" << std::endl;
		return;
	}
	// Exit if CGAL couldnt read the input file. (protocol)
	if (!(input >> mesh))
	{
		std::cerr << "Input is not" << std::endl;
		return;
	}
	// Exit if the input is empty
	if (mesh.empty())
	{
		std::cerr << "Input is not a" << std::endl;
		return;
	}
	// makes an inside() function for points in the future
	CGAL::Side_of_triangle_mesh<Polyhedron, K> inside(mesh);
	// computes bounding box of the mesh
	K::Iso_cuboid_3 boudingBox = CGAL::bounding_box(mesh.points_begin(), mesh.points_end());

	double xmax = boudingBox.xmax();
	double xmin = boudingBox.xmin();
	double ymax = boudingBox.ymax();
	double ymin = boudingBox.ymin();
	double zmax = boudingBox.zmax();
	double zmin = boudingBox.zmin();

	// Length per axis
	double xgap = xmax - xmin;
	double ygap = ymax - ymin;
	double zgap = zmax - zmin;


	std::cout << xmin << " " << xmax << std::endl;
	std::cout << ymin << " " << ymax << std::endl;
	std::cout << zmin << " " << zmax << std::endl;

	TpmsType tpms_type = changekP;
	// Input tpms type
	std::cout << "Input TPMS type: 0-kP, 1-kD, 2-kG, 10-changekP" << std::endl;
	int tpms_input;
	std::cin >> tpms_input;
	tpms_type = static_cast<TpmsType>(tpms_input);

    // Resolution - for each length get times for the samples
	// Number of points goes into marching cubes: (gapx*res)*(gapy*res)*(gapz*res)
	std::cout << "Input resolution multiplier" << std::endl;
	double resolution = 4;
	std::cin >> resolution;
	int x_num = xgap * resolution;
	int y_num = ygap * resolution;
	int z_num = zgap * resolution;



	// Half thickness (-t, +t)
	std::cout << "Input thickness" << std::endl;
	double thickness = 0.28;
	std::cin >> thickness;

	// W in sin(wx)
	std::cout << "Input frequency" << std::endl;
	double omega = 0.8;
	std::cin >> omega;

	Eigen::RowVector3d Vmin = { xmin - 0.1, ymin - 0.1, zmin - 0.1 };
	Eigen::RowVector3d Vmax = { xmax + 0.1, ymax + 0.1, zmax + 0.1 };
	Eigen::RowVector3i res = { x_num, y_num, z_num };

	Eigen::MatrixXd grid(res(0) * res(1) * res(2), 3);

	const auto lerp = [&](const int di, const int d)->double
	{return Vmin(d) + (double)di / (double)(res(d) - 1) * (Vmax(d) - Vmin(d)); };
// In Grid now there is world coorinates of every samples vertics
#pragma omp parallel for
	for (int zi = 0; zi < res(2); zi++)
	{
		const double z = lerp(zi, 2);
		for (int yi = 0; yi < res(1); yi++)
		{
			const double y = lerp(yi, 1);

			for (int xi = 0; xi < res(0); xi++)
			{
				const double x = lerp(xi, 0);
				grid.row(xi + res(0) * (yi + res(1) * zi)) = Eigen::RowVector3d(x, y, z);
			}
		}
	}


	const auto& tpms_function = [tpms_type, omega](double x, double y, double z)-> double {
		double tpms_value = 0;

		// TPMS Functions
		if (tpms_type == kP)
			tpms_value = cos(omega * x) + cos(omega * y) + cos(omega * z);
		else if (tpms_type == kD)
			tpms_value = cos(x) * cos(y) * cos(z) - sin(x) * sin(y) * sin(z);
		else if (tpms_type == kG)
			//tpms_value = sin(0.5 * x) * cos(0.5 * y) + sin(0.5 * y) * cos(0.5 * z) + sin(0.5 * z) * cos(0.5 * x);
			tpms_value = sin(omega * x) * cos(omega * (y)) + sin(omega * (y)) * cos(omega * z) + sin(omega * z) * cos(omega * x);
		else if (tpms_type == kL)
			tpms_value = 0.5 * (sin(2 * x) * cos(y) * sin(z) + sin(2 * y) * cos(z) * sin(x) + sin(2 * z) * cos(x) * sin(y))
			- 0.5 * (cos(2 * x) * cos(2 * y) + cos(2 * y) * cos(2 * z) + cos(2 * z) * cos(2 * x))
			+ 0.15;//0.15
		else if (tpms_type == KI_WP)
			tpms_value = 2 * (cos(x) * cos(y) + cos(y) * cos(z) + cos(z) * cos(x)) -
			(cos(2 * x) + cos(2 * y) + cos(2 * z));
		else if (tpms_type == kTubularP)
			tpms_value = 10 * (cos(x) + cos(y) + cos(z))
			- 5.1 * (cos(x) * cos(y) + cos(y) * cos(z) + cos(z) * cos(x))
			- 14.6;//-14.6
		else if (tpms_type == kTubularG)
			tpms_value = 10 * (cos(x) * sin(y) + cos(y) * sin(z) + cos(z) * sin(x))
			- 0.5 * (cos(2 * x) * cos(2 * y) + cos(2 * y) * cos(2 * z) + cos(2 * z) * cos(2 * x))
			- 14;//-14
		else if (tpms_type == KF_RD)
			tpms_value = 4 * cos(x) * cos(y) * cos(z) -
			(cos(2 * x) * cos(2 * y) + cos(2 * x) * cos(2 * z) + cos(2 * y) * cos(2 * z));
		else if (tpms_type == P_W)
			tpms_value = 4 * (cos(x) * cos(y) + cos(y) * cos(z) + cos(z) * cos(x)) - 3 * cos(x) * cos(y) * cos(z);
		else if (tpms_type == N)
			tpms_value = 3 * (cos(x) + cos(y) + cos(z)) + 4 * cos(x) * cos(y) * cos(z);
		else if (tpms_type == changekP){
			const double r = sqrt(x*x + y*y + z*z);
			const double w_r_inc = (omega * sqrt(1.0 + 2*r));
			tpms_value = cos(w_r_inc * x) + cos(w_r_inc * y) + cos(w_r_inc * z);
		}

		return tpms_value;
	};

	// Field for marching cubes
	Eigen::VectorXd B(grid.rows());

#pragma omp parallel for
	for (int i = 0; i < grid.rows(); i++) {
		const double x = grid(i,0), y = grid(i,1), z = grid(i,2);

		// signed field: negative = inside
		double f = tpms_function(x,y,z);
		double phi = std::abs(f) - thickness;   // <0 inside, 0 surface, >0 outside
		B(i) = phi;

		// keep only inside the input mesh's bounded region
		Point q(x,y,z);
		CGAL::Bounded_side res = inside(q);
		if (res != CGAL::ON_BOUNDED_SIDE) {
			B(i) = 1000.0;  // force outside
		}
	}

	Eigen::MatrixXd V, heV1;
	Eigen::MatrixXi F, heF1;
	// Reminder B(i) < 0 = inside
	igl::copyleft::marching_cubes(B, grid, res(0), res(1), res(2), V, F);
	igl::writeOBJ(output_path, V, F);
}



int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./tpms_boundary input.off output.obj" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    generateTPMSwithPlate69(input_path, output_path);
    return 0;
}

// for andrei.off 5 - 0.5 - 0.5 - 0 - 5
// for objects -1 to 1: 100 - 0.2 - 13 - 0 - 1