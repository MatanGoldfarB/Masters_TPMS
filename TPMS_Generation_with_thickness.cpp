#include<iostream>
#include "Eigen/Core"
#include<vector>
#include "igl/copyleft/marching_cubes.h"
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/principal_curvature.h>
#include <igl/invert_diag.h>
#include <stdio.h>

#define M_PI 3.1415926
//#define HandIn
using namespace Eigen;
using namespace std;
enum TpmsType : char {
	kP = 1,
	kD,
	kG,
	kL,
	KI_WP,
	kTubularP,
	kTubularG,
	KF_RD,
	P_W,
	N,
	SD,
	Sphere
};

TpmsType tpms_type = kG;//tpms type

double tpms_constant_KP_ = 0;//C value
double tpms_constant_kD_ = 0;
double tpms_constant_KG_ = 0;
double tpms_constant_KL_ = 0;
double tpms_constant_kTubularP_ = 0;
double tpms_constant_kTubularG_ = 0;
double tpms_constant_KP2_ = 0;
double tpms_constant_KI_WP_ = 0;
double tpms_constant_KF_RD_ = 0;
double tpms_constant_P_W_ = -2.4;
double tpms_constant_N_ = 0;


// TPMS Function to devide the cube
double TpmsValue(Eigen::Vector3d coord)
{
	double x = coord[0], y = coord[1], z = coord[2];
	double tpms_value = 0;
	if (tpms_type == kP)
		tpms_value = cos(x) + cos(y) + cos(z) + tpms_constant_KP_;
	else if (tpms_type == kD)
		tpms_value = cos(x) * cos(y) * cos(z) - sin(x) * sin(y) * sin(z) + tpms_constant_kD_;
	else if (tpms_type == kG)
		tpms_value = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x) + tpms_constant_KG_;
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
	else if (tpms_type == SD)
		tpms_value = sin(x) * sin(y) * sin(z) + sin(x) * cos(y) * cos(z) + cos(x) * sin(y) * cos(z) + cos(x) * cos(y) * sin(z);
	else if (tpms_type == Sphere){
		double R = 50.0;
		tpms_value = - x*x - y*y - z*z + R*R;
	}
	return tpms_value;

}


std::vector<std::vector<std::vector<double>>> field_value_;//TPMS function values
std::vector<std::vector<std::vector<double>>> field_value_single;
std::vector<std::vector<std::vector<double>>> distance_field_model_;
std::vector<std::vector<std::vector<double>>> voxel_model_;//voxel field
std::vector<std::vector<std::vector<double>>> distance_field_tpms_;
std::vector<std::vector<std::vector<double>>> distance_field_boolean_;

std::vector<std::vector<std::vector<double>>> integrate_x_field_;
std::vector<std::vector<std::vector<double>>> integrate_y_field_;
std::vector<std::vector<std::vector<double>>> integrate_z_field_;
std::vector<std::vector<std::vector<double>>> omega_field_;

Eigen::MatrixXd mesh_vertex_list_;
Eigen::MatrixXi mesh_facet_list_;

Eigen::MatrixXd V;
Eigen::MatrixXi F;

// Voxel Resolution
int x_voxel_num_ = 100;
int y_voxel_num_ = 100;
int z_voxel_num_ = 100;


// Making the Cube 
double length_x = 100;
double length_y = 100;
double length_z = 100;

double voxel_size_x = length_x / (double)x_voxel_num_;
double voxel_size_y = length_y / (double)y_voxel_num_;
double voxel_size_z = length_z / (double)z_voxel_num_;

// coefficient is the angual frequency: w in sin(wx)
double tpms_cycle_count_ = 11;//1.06low0.98mid0.94
double tpms_coefficient = tpms_cycle_count_ * 2 * M_PI / length_z;

// Centers the cube to (0,0,0)
Eigen::Vector3d model_origin_coordinate_ = { -length_x / 2.0, -length_y / 2.0, -length_z / 2.0 };

// p 16 0.669922
// p 32 0.823975
// p 64 0.909149

void singleTPMS_onePart() {
	//std::cout << "Input the num of unit cells" << std::endl;
	//std::cin >> tpms_cycle_count_;
	// coefficient is the angual frequency: w in sin(wx)
	tpms_cycle_count_ = 3;
	tpms_coefficient = tpms_cycle_count_ * 2 * M_PI / length_z;
	double rx = length_x / 2.0;
	double ry = length_y / 2.0;
	double rz = length_z / 2.0;
	Eigen::RowVector3d Vmin = { -1 * rx - 0.1, -1 * ry - 0.1, -1 * rz - 0.1 }; // cube min (x,y,z)
	Eigen::RowVector3d Vmax = { rx + 0.1, ry + 0.1, rz + 0.1 }; // cube max (x,y,z)
	Eigen::RowVector3i res_mesh{ x_voxel_num_ , y_voxel_num_, z_voxel_num_ }; // resolution of the cube
	Eigen::MatrixXd GV(res_mesh(0) * res_mesh(1) * res_mesh(2), 3); // alocates the matrix for the next loop


	const auto lerp = [&](const int di, const int d)->double
	{return Vmin(d) + (double)di / (double)(res_mesh(d) - 1) * (Vmax(d) - Vmin(d)); };

	// Each indexed voxel gets its real coordinates store in GV
	#pragma omp parallel for
	for (int zi = 0; zi < res_mesh(2); zi++)
	{
		const double z = lerp(zi, 2);
		for (int yi = 0; yi < res_mesh(1); yi++)
		{
			const double y = lerp(yi, 1);
			for (int xi = 0; xi < res_mesh(0); xi++)
			{
				const double x = lerp(xi, 0);
				GV.row(xi + res_mesh(0) * (yi + res_mesh(1) * zi)) = Eigen::RowVector3d(x, y, z);
			}
		}
	}

	//tpms_type = TpmsType(3);
	tpms_type = TpmsType(12);
	Eigen::VectorXd field(GV.rows());

	double thickp = 0.18;
	//std::cout << "Input thickness: " << std::endl;
	//std::cin >> thickp;
	int negnum = 0;
	// Making the field values per voxel
	for (int i = 0; i < GV.rows(); i++){
		Eigen::Vector3d coord(GV(i, 0), GV(i, 1), GV(i, 2));
		//coord.z() += 0.0;
		if (tpms_type != Sphere){
			coord *= tpms_coefficient;
			field(i) = TpmsValue(coord);
			if (field(i) > 0)
				field(i) -= thickp;
			else if (field(i) < 0)
				field(i) = -field(i) - thickp;
		}
		else{
			field(i) = TpmsValue(coord);
		}
		// checks if within the box
		if (abs(GV(i, 0)) > rx || abs(GV(i, 1)) > ry || abs(GV(i, 2)) > rz){
			field(i) = 1000;
		}
		// counts negative voxel fields
		if (field(i) < 0)
			negnum++;
	}
	double vof = (negnum * 1.0) / (GV.rows() * 1.0);
	std::cout << "Vof" << vof << std::endl; 
	/*double vof = 0.1;
	int negnum = (int)vof * GV.rows();*/

	

	igl::copyleft::marching_cubes(field, GV, x_voxel_num_, y_voxel_num_, z_voxel_num_, mesh_vertex_list_, mesh_facet_list_);
	
	std::string name = "thickness";
	igl::writeOBJ(name + ".obj", mesh_vertex_list_, mesh_facet_list_);
}

int main()
{
	singleTPMS_onePart();
}


/*void MarchingCube(const std::vector<std::vector<std::vector<double>>>& field)
{
	VectorXd value(x_voxel_num_ * y_voxel_num_ * z_voxel_num_, 1);
	//std::cout << x_voxel_num_ << " " << y_voxel_num_ << " " << z_voxel_num_ << std::endl;
	MatrixXd coordinate(x_voxel_num_ * y_voxel_num_ * z_voxel_num_, 3);
	int sample_count = 0;
	for (int i = 0; i < x_voxel_num_; i++)
		for (int j = 0; j < y_voxel_num_; j++)
			for (int k = 0; k < z_voxel_num_; k++)
			{
				Vector3d coord(i * voxel_size_x, j * voxel_size_y, k * voxel_size_z);
				coord += model_origin_coordinate_;
				value(sample_count) = field[i][j][k];//store the computed TPMS function value
				coordinate(sample_count, 0) = coord[0];
				coordinate(sample_count, 1) = coord[1];
				coordinate(sample_count, 2) = coord[2];
				++sample_count;
			}
	//cout << "Start Marching Cube..." << endl;
	igl::copyleft::marching_cubes(value, coordinate,
		z_voxel_num_, y_voxel_num_, x_voxel_num_, mesh_vertex_list_, mesh_facet_list_);
	//cout << "Marching Cube Finished." << endl;
	double temp;
	for (int i = 0; i < mesh_facet_list_.rows(); i++)
	{
		temp = mesh_facet_list_(i, 0);
		mesh_facet_list_(i, 0) = mesh_facet_list_(i, 2);
		mesh_facet_list_(i, 2) = temp;
	}
} */

/*void WriteTpmsMesh(std::string filename)
{
	igl::writeOBJ(filename, mesh_vertex_list_, mesh_facet_list_);
}*/