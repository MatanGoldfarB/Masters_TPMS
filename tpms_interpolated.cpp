#include <CGAL/Simple_cartesian.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/bounding_box.h>

#include <igl/copyleft/marching_cubes.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <Eigen/Geometry>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel K;
typedef CGAL::Surface_mesh<Point> SurfaceMesh;
typedef CGAL::AABB_face_graph_triangle_primitive<SurfaceMesh> Primitive;
typedef CGAL::AABB_traits_3<K, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> Tree;

struct ControlPoint {
	double x, y, z;
	double n_cells;
	double thickness;
};

// Parse control points file. Format: one point per line, columns: n_cells x y z thickness
// Lines starting with # are comments.
static std::vector<ControlPoint> parse_control_points(const std::string& path)
{
	std::vector<ControlPoint> pts;
	std::ifstream fin(path);
	if (!fin) {
		std::cerr << "Error: could not open control points file: " << path << std::endl;
		return pts;
	}
	std::string line;
	while (std::getline(fin, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream iss(line);
		ControlPoint cp;
		if (iss >> cp.n_cells >> cp.x >> cp.y >> cp.z >> cp.thickness) {
			pts.push_back(cp);
		}
	}
	std::cout << "  Loaded " << pts.size() << " control points from " << path << std::endl;
	return pts;
}

// kP (Primitive) TPMS function: cos(omega*x) + cos(omega*y) + cos(omega*z)
static double tpms_kP(double omega, double x, double y, double z)
{
	return cos(omega * x) + cos(omega * y) + cos(omega * z);
}

// IDW interpolation of omega and thickness from control points.
// Returns {local_omega, local_thickness}.
static std::pair<double, double> idw_interpolate(
	double x, double y, double z,
	const std::vector<ControlPoint>& pts, double max_dim)
{
	double w_sum = 0.0, omega_sum = 0.0, thick_sum = 0.0;
	for (const auto& p : pts) {
		double dx = x - p.x, dy = y - p.y, dz = z - p.z;
		double dist2 = dx*dx + dy*dy + dz*dz;
		if (dist2 < 1e-12) {
			double omega = p.n_cells * 2.0 * M_PI / max_dim;
			return {omega, p.thickness};
		}
		double w = 1.0 / dist2;
		omega_sum += w * (p.n_cells * 2.0 * M_PI / max_dim);
		thick_sum += w * p.thickness;
		w_sum += w;
	}
	return {omega_sum / w_sum, thick_sum / w_sum};
}

// Write triangles from V,F to binary STL stream. Returns number of triangles written.
static uint32_t write_triangles_stl(FILE* fp, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
	for (int i = 0; i < F.rows(); i++) {
		Eigen::Vector3d a = V.row(F(i,0));
		Eigen::Vector3d b = V.row(F(i,1));
		Eigen::Vector3d c = V.row(F(i,2));
		Eigen::Vector3d e1 = b - a;
		Eigen::Vector3d e2 = c - a;
		Eigen::Vector3d n = e1.cross(e2);
		double len = n.norm();
		if (len > 0) n /= len;

		float nf[3] = {(float)n(0), (float)n(1), (float)n(2)};
		fwrite(nf, sizeof(float), 3, fp);
		for (int j = 0; j < 3; j++) {
			float vf[3] = {(float)V(F(i,j),0), (float)V(F(i,j),1), (float)V(F(i,j),2)};
			fwrite(vf, sizeof(float), 3, fp);
		}
		uint16_t attr = 0;
		fwrite(&attr, sizeof(uint16_t), 1, fp);
	}
	return static_cast<uint32_t>(F.rows());
}

void generate(const std::string& input_path, const std::string& cp_path,
              const std::string& output_path, double resolution)
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

	// Inside test — one instance per thread for thread safety
#ifdef _OPENMP
	int max_threads = omp_get_max_threads();
#else
	int max_threads = 1;
#endif
	std::vector<CGAL::Side_of_triangle_mesh<SurfaceMesh, K>> inside_queries;
	inside_queries.reserve(max_threads);
	for (int t = 0; t < max_threads; t++) {
		inside_queries.emplace_back(mesh);
	}

	// AABB tree for distance queries
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

	// --- Load control points ---
	std::vector<ControlPoint> control_points = parse_control_points(cp_path);
	if (control_points.empty()) {
		std::cerr << "Error: no control points loaded" << std::endl;
		return;
	}

	// --- Grid setup ---
	int x_num = static_cast<int>(xgap * resolution);
	int y_num = static_cast<int>(ygap * resolution);
	int z_num = static_cast<int>(zgap * resolution);
	x_num = std::max(x_num, 2);
	y_num = std::max(y_num, 2);
	z_num = std::max(z_num, 2);
	std::cout << "  Grid: " << x_num << " x " << y_num << " x " << z_num
	          << " = " << (long long)x_num * y_num * z_num << " voxels" << std::endl;

	// Global grid bounds (padded proportionally to mesh size)
	double pad = 0.01 * max_dim;
	Eigen::RowVector3d Vmin = { xmin - pad, ymin - pad, zmin - pad };
	Eigen::RowVector3d Vmax = { xmax + pad, ymax + pad, zmax + pad };

	const auto lerp_global = [&](int idx, int axis, int total) -> double {
		return Vmin(axis) + (double)idx / (double)(total - 1) * (Vmax(axis) - Vmin(axis));
	};

	// --- Chunked marching cubes with streaming STL output ---
	const int BLOCK = 64;

	int nx_blocks = (x_num + BLOCK - 1) / BLOCK;
	int ny_blocks = (y_num + BLOCK - 1) / BLOCK;
	int nz_blocks = (z_num + BLOCK - 1) / BLOCK;
	int total_blocks = nx_blocks * ny_blocks * nz_blocks;

	std::cout << "  Blocks: " << nx_blocks << " x " << ny_blocks << " x " << nz_blocks
	          << " = " << total_blocks << " (block size " << BLOCK << ")" << std::endl;

	FILE* stl_fp = fopen(output_path.c_str(), "wb");
	if (!stl_fp) {
		std::cerr << "Error: could not open " << output_path << " for writing" << std::endl;
		return;
	}

	// Write STL header
	char header[80] = {};
	snprintf(header, 80, "type=interp base=kP res=%.4g cp=%s",
	         resolution, cp_path.c_str());
	fwrite(header, 1, 80, stl_fp);

	uint32_t total_triangles = 0;
	long count_pos = ftell(stl_fp);
	fwrite(&total_triangles, sizeof(uint32_t), 1, stl_fp);

	long long total_negcount = 0;
	long long total_voxels = 0;
	int blocks_done = 0;

	for (int bz = 0; bz < nz_blocks; bz++) {
		for (int by = 0; by < ny_blocks; by++) {
			for (int bx = 0; bx < nx_blocks; bx++) {
				int gx0 = bx * BLOCK;
				int gy0 = by * BLOCK;
				int gz0 = bz * BLOCK;
				int gx1 = std::min(gx0 + BLOCK, x_num);
				int gy1 = std::min(gy0 + BLOCK, y_num);
				int gz1 = std::min(gz0 + BLOCK, z_num);

				int gx1_ov = std::min(gx1 + 1, x_num);
				int gy1_ov = std::min(gy1 + 1, y_num);
				int gz1_ov = std::min(gz1 + 1, z_num);

				int sub_nx = gx1_ov - gx0;
				int sub_ny = gy1_ov - gy0;
				int sub_nz = gz1_ov - gz0;

				if (sub_nx < 2 || sub_ny < 2 || sub_nz < 2) continue;

				int sub_total = sub_nx * sub_ny * sub_nz;

				Eigen::MatrixXd sub_grid(sub_total, 3);
				for (int sz = 0; sz < sub_nz; sz++) {
					double z = lerp_global(gz0 + sz, 2, z_num);
					for (int sy = 0; sy < sub_ny; sy++) {
						double y = lerp_global(gy0 + sy, 1, y_num);
						for (int sx = 0; sx < sub_nx; sx++) {
							double x = lerp_global(gx0 + sx, 0, x_num);
							sub_grid.row(sx + sub_nx * (sy + sub_ny * sz)) =
								Eigen::RowVector3d(x, y, z);
						}
					}
				}

				Eigen::VectorXd sub_B(sub_total);
				int block_negcount = 0;

#pragma omp parallel for reduction(+:block_negcount)
				for (int i = 0; i < sub_total; i++) {
					const double x = sub_grid(i, 0);
					const double y = sub_grid(i, 1);
					const double z = sub_grid(i, 2);
					Point q(x, y, z);

					auto cp_and_prim = tree.closest_point_and_primitive(q);
					Point closest = cp_and_prim.first;
					double unsigned_d = std::sqrt(CGAL::squared_distance(q, closest));
					#ifdef _OPENMP
						int tid = omp_get_thread_num();
					#else
						int tid = 0;
					#endif
					CGAL::Bounded_side side = inside_queries[tid](q);
					double d = (side != CGAL::ON_UNBOUNDED_SIDE) ? unsigned_d : -unsigned_d;

					auto [local_omega, local_thick] = idw_interpolate(x, y, z, control_points, max_dim);
					double local_half_t = (local_thick / 2.0) * local_omega;
					double f = tpms_kP(local_omega, x, y, z);
					double phi_tpms = std::abs(f) - local_half_t;
					double local_smf = (2.0 * M_PI / local_omega) / 10.0;

					double phi;
					if (d < -3.0 * local_smf) {
						phi = 1000.0;
					} else {
						double a = phi_tpms / local_smf;
						double b = -d / local_smf;
						double m = std::max(a, b);
						phi = local_smf * (m + std::log(std::exp(a - m) + std::exp(b - m)));
					}

					sub_B(i) = phi;
					if (phi < 0.0) block_negcount++;
				}

				total_negcount += block_negcount;
				total_voxels += sub_total;

				Eigen::MatrixXd V;
				Eigen::MatrixXi F;
				igl::copyleft::marching_cubes(sub_B, sub_grid, sub_nx, sub_ny, sub_nz, V, F);

				if (F.rows() > 0) {
					total_triangles += write_triangles_stl(stl_fp, V, F);
				}

				blocks_done++;
				double pct = 100.0 * blocks_done / total_blocks;
				std::cout << "\r  Progress: " << (int)pct << "% | Block " << blocks_done << "/" << total_blocks
				          << " | " << total_triangles << " triangles     " << std::flush;
			}
		}
	}
	std::cout << std::endl;

	// Patch triangle count in STL header
	fseek(stl_fp, count_pos, SEEK_SET);
	fwrite(&total_triangles, sizeof(uint32_t), 1, stl_fp);
	fclose(stl_fp);

	double vof = (total_voxels > 0) ? (double)total_negcount / (double)total_voxels : 0.0;
	std::cout << "Volume fraction: " << vof << std::endl;
	std::cout << "Output: " << total_triangles << " triangles" << std::endl;
	std::cout << "Written to " << output_path << std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 5) {
		std::cerr << "Usage: " << argv[0] << " input_mesh control_points.txt output.stl resolution" << std::endl;
		std::cerr << "  input_mesh:       boundary mesh (.off, .obj, .stl, .ply)" << std::endl;
		std::cerr << "  control_points:   file with lines: n_cells x y z thickness" << std::endl;
		std::cerr << "  output.stl:       output TPMS mesh (binary STL)" << std::endl;
		std::cerr << "  resolution:       voxels per unit length" << std::endl;
		return 1;
	}

	std::string input_path = argv[1];
	std::string cp_path = argv[2];
	std::string output_path = argv[3];
	double resolution = std::stod(argv[4]);

	if (resolution <= 0) {
		std::cerr << "Error: resolution must be positive" << std::endl;
		return 1;
	}

	generate(input_path, cp_path, output_path, resolution);
	return 0;
}
