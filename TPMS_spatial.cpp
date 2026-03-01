/**
 * TPMS_spatial.cpp
 * 
 * TPMS generator with spatial thickness mode.
 * In this version, the "thickness" parameter is in world/spatial units,
 * not in function value units.
 * 
 * Example: If your object is 1 meter wide, thickness=0.1 means 
 * walls will be 0.1 meters thick everywhere.
 */

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
#include <CGAL/Polygon_mesh_processing/distance.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/writeOBJ.h>
#include <igl/facet_components.h>
#include <igl/remove_unreferenced.h>
#include <fstream>

#include <iostream>
#include <fstream>
#include <string>
#include <utility>  // for std::pair
#include <filesystem>  // for directory creation

typedef CGAL::Simple_cartesian<double> Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
typedef Kernel::Point_3 Point;
typedef Kernel K;

// AABB tree types for distance queries
typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
typedef CGAL::AABB_traits_3<K, Primitive> AABB_Traits;
typedef CGAL::AABB_tree<AABB_Traits> AABB_Tree;

enum TpmsType {
    kP,             // 0: Schwarz-P
    kG,             // 1: Gyroid
    kD,             // 2: Diamond (Schwarz-D)
    changekP,       // 3: Variable frequency Schwarz-P
    changekG,       // 4: Variable frequency Gyroid
    changekD,       // 5: Variable frequency Diamond
};


void generateTPMSwithSpatialThickness(const std::string& input_path, const std::string& output_path)
{
	Polyhedron mesh;
	std::ifstream input(input_path);
	if (!input)
	{
		std::cerr << "Error: Could not open input file" << std::endl;
		return;
	}
	if (!(input >> mesh))
	{
		std::cerr << "Error: Could not read mesh from input file" << std::endl;
		return;
	}
	if (mesh.empty())
	{
		std::cerr << "Error: Input mesh is empty" << std::endl;
		return;
	}

	// Setup inside/outside query
	CGAL::Side_of_triangle_mesh<Polyhedron, K> inside(mesh);
	
	// Build AABB tree for distance queries
	AABB_Tree tree(faces(mesh).first, faces(mesh).second, mesh);
	tree.accelerate_distance_queries();
	
	// Compute bounding box
	K::Iso_cuboid_3 boundingBox = CGAL::bounding_box(mesh.points_begin(), mesh.points_end());

	double xmax = boundingBox.xmax();
	double xmin = boundingBox.xmin();
	double ymax = boundingBox.ymax();
	double ymin = boundingBox.ymin();
	double zmax = boundingBox.zmax();
	double zmin = boundingBox.zmin();

	double xgap = xmax - xmin;
	double ygap = ymax - ymin;
	double zgap = zmax - zmin;

	std::cout << "Bounding box:" << std::endl;
	std::cout << "  X: [" << xmin << ", " << xmax << "] (size: " << xgap << ")" << std::endl;
	std::cout << "  Y: [" << ymin << ", " << ymax << "] (size: " << ygap << ")" << std::endl;
	std::cout << "  Z: [" << zmin << ", " << zmax << "] (size: " << zgap << ")" << std::endl;

	// Input TPMS type
	std::cout << "\nTPMS types with SPATIAL thickness:" << std::endl;
	std::cout << "  0 - Schwarz-P (kP)" << std::endl;
	std::cout << "  1 - Gyroid (kG)" << std::endl;
	std::cout << "  2 - Diamond (kD)" << std::endl;
	std::cout << "  3 - Variable frequency Schwarz-P (changekP)" << std::endl;
	std::cout << "  4 - Variable frequency Gyroid (changekG)" << std::endl;
	std::cout << "  5 - Variable frequency Diamond (changekD)" << std::endl;
	std::cout << "Input TPMS type: ";
	int tpms_input;
	std::cin >> tpms_input;
	TpmsType tpms_type = static_cast<TpmsType>(tpms_input);

	// Resolution
	std::cout << "Input resolution multiplier: ";
	double resolution = 4;
	std::cin >> resolution;
	int x_num = xgap * resolution;
	int y_num = ygap * resolution;
	int z_num = zgap * resolution;

	// Thickness in SPATIAL units
	std::cout << "Input wall thickness (in spatial units, e.g., 0.1 = 10% of a 1-unit object): ";
	double thickness = 0.1;
	std::cin >> thickness;

	// Frequency (number of TPMS cells)
	std::cout << "Input frequency (omega): ";
	double omega = 5.0;
	std::cin >> omega;

	// Variable frequency types now use distance-to-surface based formula
	// No k input needed - formula is smooth with 2 wavelengths constant, then plummet

	std::cout << "\nGenerating TPMS with:" << std::endl;
	std::cout << "  Physical wall thickness: " << thickness << " units" << std::endl;
	std::cout << "  Frequency: " << omega << std::endl;
	if (tpms_type == changekP || tpms_type == changekG || tpms_type == changekD) {
		std::cout << "  Variable frequency: 2 wavelengths constant, then plummet" << std::endl;
	}
	std::cout << "  Grid resolution: " << x_num << " x " << y_num << " x " << z_num << std::endl;

	Eigen::RowVector3d Vmin = { xmin - 0.1, ymin - 0.1, zmin - 0.1 };
	Eigen::RowVector3d Vmax = { xmax + 0.1, ymax + 0.1, zmax + 0.1 };
	Eigen::RowVector3i res = { x_num, y_num, z_num };

	Eigen::MatrixXd grid(res(0) * res(1) * res(2), 3);

	const auto lerp = [&](const int di, const int d)->double
	{return Vmin(d) + (double)di / (double)(res(d) - 1) * (Vmax(d) - Vmin(d)); };

	// Build grid
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

	// Function to compute TPMS value and gradient magnitude for Schwarz-P
	// Returns {tpms_value, gradient_magnitude}
	const auto compute_schwarz_p = [](double x, double y, double z, double w) -> std::pair<double, double> {
		// f = cos(wx) + cos(wy) + cos(wz)
		double f = cos(w * x) + cos(w * y) + cos(w * z);
		
		// Gradient: ∇f = (-w*sin(wx), -w*sin(wy), -w*sin(wz))
		double gx = -w * sin(w * x);
		double gy = -w * sin(w * y);
		double gz = -w * sin(w * z);
		double grad_mag = sqrt(gx*gx + gy*gy + gz*gz);
		
		// Avoid division by zero - use average gradient magnitude
		if (grad_mag < 1e-10) {
			grad_mag = w * sqrt(3.0);
		}
		
		return {f, grad_mag};
	};

	// Function to compute Gyroid value and gradient magnitude
	const auto compute_gyroid = [](double x, double y, double z, double w) -> std::pair<double, double> {
		// f = sin(wx)*cos(wy) + sin(wy)*cos(wz) + sin(wz)*cos(wx)
		double f = sin(w*x)*cos(w*y) + sin(w*y)*cos(w*z) + sin(w*z)*cos(w*x);
		
		// Gradient components (partial derivatives)
		double gx = w*cos(w*x)*cos(w*y) - w*sin(w*z)*sin(w*x);
		double gy = -w*sin(w*x)*sin(w*y) + w*cos(w*y)*cos(w*z);
		double gz = -w*sin(w*y)*sin(w*z) + w*cos(w*z)*cos(w*x);
		double grad_mag = sqrt(gx*gx + gy*gy + gz*gz);
		
		if (grad_mag < 1e-10) {
			grad_mag = w * sqrt(2.0);  // Approximate average for gyroid
		}
		
		return {f, grad_mag};
	};

	// Function to compute Diamond (Schwarz-D) value and gradient magnitude
	const auto compute_diamond = [](double x, double y, double z, double w) -> std::pair<double, double> {
		// f = cos(wx)*cos(wy)*cos(wz) - sin(wx)*sin(wy)*sin(wz)
		double cx = cos(w*x), cy = cos(w*y), cz = cos(w*z);
		double sx = sin(w*x), sy = sin(w*y), sz = sin(w*z);
		
		double f = cx*cy*cz - sx*sy*sz;
		
		// Gradient components (partial derivatives)
		// df/dx = -w*sin(wx)*cos(wy)*cos(wz) - w*cos(wx)*sin(wy)*sin(wz)
		// df/dy = -w*cos(wx)*sin(wy)*cos(wz) - w*sin(wx)*cos(wy)*sin(wz)
		// df/dz = -w*cos(wx)*cos(wy)*sin(wz) - w*sin(wx)*sin(wy)*cos(wz)
		double gx = -w*sx*cy*cz - w*cx*sy*sz;
		double gy = -w*cx*sy*cz - w*sx*cy*sz;
		double gz = -w*cx*cy*sz - w*sx*sy*cz;
		double grad_mag = sqrt(gx*gx + gy*gy + gz*gz);
		
		if (grad_mag < 1e-10) {
			grad_mag = w * sqrt(3.0);  // Approximate average for diamond
		}
		
		return {f, grad_mag};
	};

	// Field for marching cubes
	Eigen::VectorXd B(grid.rows());

	// Precompute distances to surface for variable frequency types
	std::vector<double> distances(grid.rows(), 0.0);
	if (tpms_type == changekP || tpms_type == changekG || tpms_type == changekD) {
		std::cout << "Precomputing distances to surface..." << std::endl;
		#pragma omp parallel for
		for (int i = 0; i < grid.rows(); i++) {
			Point q(grid(i,0), grid(i,1), grid(i,2));
			distances[i] = sqrt(tree.squared_distance(q));
		}
		std::cout << "Done computing distances." << std::endl;
	}

	std::cout << "Computing TPMS field..." << std::endl;

	#pragma omp parallel for
	for (int i = 0; i < grid.rows(); i++) {
		const double x = grid(i,0), y = grid(i,1), z = grid(i,2);

		double f, grad_mag;
		double w_local = omega;
		
		// Compute local frequency for variable-frequency types
		// 1 wavelength constant near surface, then gradual decrease (C∞ smooth)
		// After 1 unit beyond threshold: drops to ω/2
		if (tpms_type == changekP || tpms_type == changekG || tpms_type == changekD) {
			const double dist_to_surface = distances[i];
			const double lambda = 2.0 * M_PI / omega;
			const double threshold = 1.0 * lambda;  // 1 wavelength
			const double k = lambda;  // k = λ means ω/2 after 1 unit beyond threshold
			
			const double excess = dist_to_surface - threshold;
			// Smooth ReLU (softplus-style): C∞ continuous and differentiable
			const double smooth_excess = 0.5 * (excess + sqrt(excess*excess + 0.01*lambda*lambda));
			w_local = omega / (1.0 + k * smooth_excess / lambda);
		}
		
		// Compute TPMS value and gradient based on type
		if (tpms_type == kP || tpms_type == changekP) {
			auto result = compute_schwarz_p(x, y, z, w_local);
			f = result.first;
			grad_mag = result.second;
		} else if (tpms_type == kG || tpms_type == changekG) {
			auto result = compute_gyroid(x, y, z, w_local);
			f = result.first;
			grad_mag = result.second;
		} else if (tpms_type == kD || tpms_type == changekD) {
			auto result = compute_diamond(x, y, z, w_local);
			f = result.first;
			grad_mag = result.second;
		} else {
			// Default to Schwarz-P
			auto result = compute_schwarz_p(x, y, z, w_local);
			f = result.first;
			grad_mag = result.second;
		}
		
		// Convert spatial thickness to function threshold
		// Relationship: physical_thickness ≈ 2 * threshold / |∇f|
		// Therefore: threshold = physical_thickness * |∇f| / 2
		double local_threshold = thickness * grad_mag / 2.0;
		
		// Compute signed distance field
		double phi = std::abs(f) - local_threshold;  // <0 inside wall, >0 outside wall
		B(i) = phi;

		// Keep only inside the input mesh's bounded region
		Point q(x, y, z);
		CGAL::Bounded_side bound_res = inside(q);
		if (bound_res != CGAL::ON_BOUNDED_SIDE) {
			B(i) = 1000.0;  // Force outside
		}
	}

	// ===== Helper function for floater removal =====
	auto removeFloaters = [](Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
	                         Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out,
	                         const std::string& name) {
		std::cout << "Removing floaters from " << name << "..." << std::endl;
		
		// Find connected components
		Eigen::VectorXi C;
		igl::facet_components(F, C);
		
		int num_components = C.maxCoeff() + 1;
		std::vector<int> component_sizes(num_components, 0);
		for (int i = 0; i < C.rows(); i++) {
			component_sizes[C(i)]++;
		}
		
		// Find largest component
		int largest_component = 0;
		int largest_size = component_sizes[0];
		for (int i = 1; i < num_components; i++) {
			if (component_sizes[i] > largest_size) {
				largest_size = component_sizes[i];
				largest_component = i;
			}
		}
		
		std::cout << "  Found " << num_components << " components, largest: " << largest_size << " faces" << std::endl;
		
		// Extract largest component
		Eigen::MatrixXi F_largest(largest_size, 3);
		int idx = 0;
		for (int i = 0; i < F.rows(); i++) {
			if (C(i) == largest_component) {
				F_largest.row(idx++) = F.row(i);
			}
		}
		
		// Remove unreferenced vertices
		Eigen::VectorXi I;
		igl::remove_unreferenced(V, F_largest, V_out, F_out, I);
		std::cout << "  After cleanup: " << V_out.rows() << " vertices, " << F_out.rows() << " faces" << std::endl;
	};

	// ===== Generate SOLID mesh (TPMS structure) =====
	std::cout << "\n=== Generating SOLID mesh (TPMS structure) ===" << std::endl;
	std::cout << "Running marching cubes..." << std::endl;

	Eigen::MatrixXd V_solid;
	Eigen::MatrixXi F_solid;
	igl::copyleft::marching_cubes(B, grid, res(0), res(1), res(2), V_solid, F_solid);
	std::cout << "Generated solid mesh: " << V_solid.rows() << " vertices, " << F_solid.rows() << " faces" << std::endl;
	
	Eigen::MatrixXd V_solid_clean;
	Eigen::MatrixXi F_solid_clean;
	removeFloaters(V_solid, F_solid, V_solid_clean, F_solid_clean, "solid");
	
	std::string solid_path = output_path;
	igl::writeOBJ(solid_path, V_solid_clean, F_solid_clean);
	std::cout << "Saved solid to: " << solid_path << std::endl;

	// ===== Generate FLUID mesh (complementary - void space for CFD) =====
	std::cout << "\n=== Generating FLUID mesh (void space for aerodynamics) ===" << std::endl;
	
	// Compute complementary field: negate the TPMS field (void becomes inside)
	// But keep the bounding mesh constraint
	Eigen::VectorXd B_fluid(grid.rows());
	
	#pragma omp parallel for
	for (int i = 0; i < grid.rows(); i++) {
		const double x = grid(i,0), y = grid(i,1), z = grid(i,2);
		
		// Check if inside bounding mesh
		Point q(x, y, z);
		CGAL::Bounded_side bound_res = inside(q);
		
		if (bound_res != CGAL::ON_BOUNDED_SIDE) {
			B_fluid(i) = 1000.0;  // Outside bounding mesh = outside fluid domain
		} else {
			// Inside bounding mesh: flip the sign
			// B(i) < 0 means inside solid, so -B(i) < 0 means inside fluid (void)
			B_fluid(i) = -B(i);
		}
	}
	
	std::cout << "Running marching cubes for fluid..." << std::endl;
	Eigen::MatrixXd V_fluid;
	Eigen::MatrixXi F_fluid;
	igl::copyleft::marching_cubes(B_fluid, grid, res(0), res(1), res(2), V_fluid, F_fluid);
	std::cout << "Generated fluid mesh: " << V_fluid.rows() << " vertices, " << F_fluid.rows() << " faces" << std::endl;
	
	// No floater removal for fluid - it's for CFD, not printing
	// The fluid is the exact complement of the solid TPMS structure
	
	// Save fluid mesh in complimentary/ folder
	// Extract just the filename from output_path
	std::string filename = output_path.substr(output_path.find_last_of('/') + 1);
	std::string fluid_path = "./complimentary/" + filename.substr(0, filename.find_last_of('.')) + "_fluid.obj";
	
	// Create complimentary directory if it doesn't exist
	std::filesystem::create_directories("./complimentary");
	
	igl::writeOBJ(fluid_path, V_fluid, F_fluid);
	std::cout << "Saved fluid to: " << fluid_path << std::endl;
	
	std::cout << "\n=== Done ===" << std::endl;
	std::cout << "Solid (TPMS structure): " << solid_path << std::endl;
	std::cout << "Fluid (void for CFD):   " << fluid_path << std::endl;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./tpms_spatial input.off output_name" << std::endl;
        std::cerr << std::endl;
        std::cerr << "This version uses SPATIAL thickness units." << std::endl;
        std::cerr << "If your object is 1 unit wide, thickness=0.1 means 0.1 unit thick walls." << std::endl;
        std::cerr << std::endl;
        std::cerr << "Output files:" << std::endl;
        std::cerr << "  ./tpms_models/<output_name>.obj        - TPMS solid structure" << std::endl;
        std::cerr << "  ./complimentary/<output_name>_fluid.obj  - Void space (for CFD/aerodynamics)" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_name = argv[2];
    
    std::string output_path = "./tpms_models/" + output_name + ".obj";
    
    generateTPMSwithSpatialThickness(input_path, output_path);
    return 0;
}

// IMPORTANT NOTE: 3 - 50 - 0.05 - 13 - 2
