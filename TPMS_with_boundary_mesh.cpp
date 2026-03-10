#include <CGAL/Simple_cartesian.h>
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
    N
};

const double tpms_constant_KP_ = 0.0;
const double tpms_constant_kD_ = 0.0;
const double tpms_constant_KG_ = 0.0;
const double tpms_constant_KI_WP_ = 0.0;
const double tpms_constant_KF_RD_ = 0.0;
const double tpms_constant_P_W_ = 0.0;
const double tpms_constant_N_ = 0.0;




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
	// Exit if the input was not trinagular surface
	if ((!CGAL::is_triangle_mesh(mesh))) {
		std::cerr << "Input is not a triangle mesh" << std::endl;
		return;
	}
	//CGAL::set_halfedgeds_items_id(mesh);
	//std::size_t facet_id = 0;
	//for (Polyhedron::Facet_iterator facet_it = mesh.facets_begin();
	//		facet_it != mesh.facets_end(); ++facet_it, ++facet_id) {
	//		facet_it->id() = facet_id;
	//	}

	//点与mesh的位置关系
	// makes an inside() function for points in the future
	CGAL::Side_of_triangle_mesh<Polyhedron, K> inside(mesh);
	//读取mesh计算包围盒
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

	// Resolution - for each length get times for the samples
	int x_num = xgap * 4;
	int y_num = ygap * 4;
	int z_num = zgap * 4;

	std::cout << xmin << " " << xmax << std::endl;
	std::cout << ymin << " " << ymax << std::endl;
	std::cout << zmin << " " << zmax << std::endl;




	TpmsType tpms_type = kG;
	//double xhalflen = xgap / 2, yhalflen = ygap / 2, zhalflen = zgap / 2;
	//double xhalflen = xgap / 8, yhalflen = ygap / 8, zhalflen = zgap / 8;
	// Half thickness (-t, +t)
	std::cout << "Input thickness" << std::endl;
	double thickness = 0.28;
	std::cin >> thickness;
	// plate thickness Y plane
	std::cout << "Input plate thickness" << std::endl;
	double thicknessplate = 0.28;
	std::cin >> thicknessplate;
	// floor plate thickness Z plane
	std::cout << "Input floor thickness" << std::endl;
	double thicknessplate_updown = 0.28;
	std::cin >> thicknessplate_updown;

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

	double zmintest = zmin;
	double zmaxtest = zmax;
	// W in sin(wx)
	std::cout << "Input frequency" << std::endl;
	double omega = 0.8;
	std::cin >> omega;
	// phase -> sin(w(x+p))
	std::cout << "Input move distance" << std::endl;
	double movey = 1.73;
	std::cin >> movey;


	const auto& function = [tpms_type, thickness, zmintest, zmaxtest, thicknessplate, thicknessplate_updown, omega, movey](double x, double y, double z)->std::pair<double, double> {
		
		// Distance from a fixed Y-Plane signed or unsigned
		const auto& disfunction = [](double x, double y, double z, int how, int way = 0)->double {
			if (how == 0)
			{
				double planey = 57.98;
				double dis = 0;
				if (way == 0)
				{
					if (y >= planey)
					{
						dis = y - planey;
					}
					else {
						dis = abs(y - planey);
					}
					return dis;
				}
				else
					return y - planey;
			}
			else if (how == 1)
			{
				double planey = 43.61;
				double dis = 0;
				if (way == 0)
				{
					if (y >= planey)
					{
						dis = y - planey;
					}
					else {
						dis = abs(y - planey);
					}
					return dis;
				}
				else
					return y - planey;
			}
			else if (how == 2)
			{
				double planey = 30.28;
				double dis = 0;
				if (way == 0)
				{
					if (y >= planey)
					{
						dis = y - planey;
					}
					else {
						dis = abs(y - planey);
					}
					return dis;
				}
				else
					return y - planey;
			}
		};

		double resdisin = 0;
		double resdisout = 0;
		//double resplate = 0;
		double smf3 = 0.5;
		double tpms_value = 0;

		// TPMS Functions
		if (tpms_type == kP)
			tpms_value = cos(x) + cos(y) + cos(z) + tpms_constant_KP_;
		else if (tpms_type == kD)
			tpms_value = cos(x) * cos(y) * cos(z) - sin(x) * sin(y) * sin(z) + tpms_constant_kD_;
		else if (tpms_type == kG)
			//tpms_value = sin(0.5 * x) * cos(0.5 * y) + sin(0.5 * y) * cos(0.5 * z) + sin(0.5 * z) * cos(0.5 * x) + tpms_constant_KG_;
			tpms_value = sin(omega * x) * cos(omega * (y + movey)) + sin(omega * (y + movey)) * cos(omega * z) + sin(omega * z) * cos(omega * x) + tpms_constant_KG_;
		else if (tpms_type == kL)
			tpms_value = 0.5 * (sin(2 * x) * cos(y) * sin(z) + sin(2 * y) * cos(z) * sin(x) + sin(2 * z) * cos(x) * sin(y))
			- 0.5 * (cos(2 * x) * cos(2 * y) + cos(2 * y) * cos(2 * z) + cos(2 * z) * cos(2 * x))
			+ 0.15;//0.15
		else if (tpms_type == KI_WP)
			tpms_value = 2 * (cos(x) * cos(y) + cos(y) * cos(z) + cos(z) * cos(x)) -
			(cos(2 * x) + cos(2 * y) + cos(2 * z)) + tpms_constant_KI_WP_;
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
			(cos(2 * x) * cos(2 * y) + cos(2 * x) * cos(2 * z) + cos(2 * y) * cos(2 * z)) + tpms_constant_KF_RD_;
		else if (tpms_type == P_W)
			tpms_value = 4 * (cos(x) * cos(y) + cos(y) * cos(z) + cos(z) * cos(x)) - 3 * cos(x) * cos(y) * cos(z) + tpms_constant_P_W_;
		else if (tpms_type == N)
			tpms_value = 3 * (cos(x) + cos(y) + cos(z)) + 4 * cos(x) * cos(y) * cos(z) + tpms_constant_N_;

		double tpmsthickness = thickness;
		double tpms_value1 = tpms_value - tpmsthickness;
		double tpms_value2 = tpms_value + tpmsthickness;
		resdisin += exp(-1 / smf3 * tpms_value1);
		resdisout += exp(1 / smf3 * tpms_value2);

		double disto_0 = disfunction(x, y, z, 0);
		double disto_1 = disfunction(x, y, z, 1);
		double disto_2 = disfunction(x, y, z, 2);
		double distoplane_0 = disfunction(x, y, z, 0, 1);
		double distoplane_1 = disfunction(x, y, z, 1, 1);
		double distoplane_2 = disfunction(x, y, z, 2, 1);

		//{
		//	double plate_value1 = disto_0 - thicknessplate;
		//	resdisin += exp(-1 / smf3 * plate_value1);
		//}
		if (z >= zmintest + 29)
		{
			double plate_value1 = disto_1 - thicknessplate;
			resdisin += exp(-1 / smf3 * plate_value1);
		}
		//{
		//	double plate_value1 = disto_2 - thicknessplate;
		//	resdisin += exp(-1 / smf3 * plate_value1);
		//}
		double platez = zmintest + thicknessplate_updown;

		if ((z >= (platez - (thicknessplate_updown))) && (z <= (platez + (thicknessplate_updown))) && (y >= 57.98))
		{
			if (z >= platez)
			{
				double plate_value1 = z - platez - thicknessplate_updown;
				resdisin += exp(-1.0 / smf3 * plate_value1);
			}
			else
			{
				double plate_value1 = platez - z - thicknessplate_updown;
				resdisin += exp(-1.0 / smf3 * plate_value1);
			}
		}

		if ((z >= (platez - (thicknessplate_updown))) && (z <= (platez + (thicknessplate_updown))) && (y <= 30.28))
		{
			if (z >= platez)
			{
				double plate_value1 = z - platez - thicknessplate_updown;
				resdisin += exp(-1.0 / smf3 * plate_value1);
			}
			else
			{
				double plate_value1 = platez - z - thicknessplate_updown;
				resdisin += exp(-1.0 / smf3 * plate_value1);
			}
		}



		

		if ((z <= zmaxtest - 16.2))//1白
		{
			double plate_value1 = thicknessplate - disto_0;;
			resdisout += exp(1.0 / smf3 * plate_value1);
		}
		if ((z >= zmintest + 14.1))//1白
		{
			double plate_value1 = thicknessplate - disto_1;;
			resdisout += exp(1.0 / smf3 * plate_value1);
		}
		if ((z <= zmaxtest - 14.8))//1白
		{
			double plate_value1 = thicknessplate - disto_2;;
			resdisout += exp(1.0 / smf3 * plate_value1);
		}

		double platez2 = zmaxtest - thicknessplate_updown;

		if (z >= (platez2 - (thicknessplate_updown)) && z <= (platez2 + (thicknessplate_updown)))
		{
			if (z >= platez2)
			{
				double plate_value1 = platez2 + thicknessplate_updown - z;
				resdisout += exp(1.0 / smf3 * plate_value1);
			}
			else
			{
				double plate_value1 = z - platez2 + thicknessplate_updown;
				resdisout += exp(1.0 / smf3 * plate_value1);
			}
		}
		if ((z >= (platez - (thicknessplate_updown))) && (z <= (platez + (thicknessplate_updown))) && ((y <= 57.98) && (y >= 30.28)))
		{
			if (z >= platez)
			{
				double plate_value1 = platez + thicknessplate_updown - z;
				resdisout += exp(1.0 / smf3 * plate_value1);
			}
			else
			{
				double plate_value1 = z - platez + thicknessplate_updown;
				resdisout += exp(1.0 / smf3 * plate_value1);
			}
		}
		
		// soft max/min functions to smooth between meeting points of a fixed plate and the TPMS
		double inner_indicator = -smf3 * log(resdisin);
		double outer_indicator = smf3 * log(resdisout);
		return std::make_pair(inner_indicator, outer_indicator);
	};

	// Field for marching cubes
	Eigen::VectorXd B(grid.rows());
	// Inner indicator for debugging
	Eigen::VectorXd Inner(grid.rows());
	double smf1 = 0.5;
#pragma omp parallel for
	for (int i = 0; i < grid.rows(); i++) {
		auto indicator = function(grid(i, 0), grid(i, 1), grid(i, 2));

		double inner = indicator.first;//-
		double outer = indicator.second;//+

		//double middle = indicator.first + indicator.second;
		Inner(i) = inner;
		double tempin = sin(omega * grid(i, 0)) * cos(omega * (grid(i, 1) + movey)) + sin(omega * (grid(i, 1) + movey)) * cos(omega * grid(i, 2)) + sin(omega * grid(i, 2)) * cos(omega * grid(i, 0)) - thickness;
		tempin = exp(-1 / smf1 * tempin);
		tempin = -smf1 * log(tempin);
		double tempout = sin(omega * grid(i, 0)) * cos(omega * (grid(i, 1) + movey)) + sin(omega * (grid(i, 1) + movey)) * cos(omega * grid(i, 2)) + sin(omega * grid(i, 2)) * cos(omega * grid(i, 0)) + thickness;
		tempout = exp(1 / smf1 * tempout);
		tempout = smf1 * log(tempout);


		if (tempin >= 0)
		{
			B(i) = inner;
		}
		else if (tempout <= 0) {
			B(i) = -1 * outer;
		}
		else {
			if (tempin + tempout > 0) {
				B(i) = inner;
			}
			else {
				B(i) = -1 * outer;
			}
		}
		Point queryPoint(grid(i, 0), grid(i, 1), grid(i, 2));
		CGAL::Bounded_side res = inside(queryPoint);
		if ((res != CGAL::ON_BOUNDED_SIDE))
		{
			B(i) = 1000;
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