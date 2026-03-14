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

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <Eigen/Geometry>

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

// Write triangles from V,F to binary STL stream. Returns number of triangles written.
static uint32_t write_triangles_stl(FILE* fp, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
	for (int i = 0; i < F.rows(); i++) {
		// Compute face normal
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

	double half_t = (thickness / 2.0) * omega;
	double wavelength = 2.0 * M_PI / omega;
	double smf = wavelength / 10.0;

	// Global grid bounds (slightly padded)
	Eigen::RowVector3d Vmin = { xmin - 0.1, ymin - 0.1, zmin - 0.1 };
	Eigen::RowVector3d Vmax = { xmax + 0.1, ymax + 0.1, zmax + 0.1 };

	// Lerp from global index to world coordinate
	const auto lerp_global = [&](int idx, int axis, int total) -> double {
		return Vmin(axis) + (double)idx / (double)(total - 1) * (Vmax(axis) - Vmin(axis));
	};

	// --- Chunked marching cubes with streaming STL output ---
	const int BLOCK = 64;  // block size per axis (excluding overlap)

	// Number of blocks per axis
	int nx_blocks = (x_num + BLOCK - 1) / BLOCK;
	int ny_blocks = (y_num + BLOCK - 1) / BLOCK;
	int nz_blocks = (z_num + BLOCK - 1) / BLOCK;
	int total_blocks = nx_blocks * ny_blocks * nz_blocks;

	std::cout << "  Blocks: " << nx_blocks << " x " << ny_blocks << " x " << nz_blocks
	          << " = " << total_blocks << " (block size " << BLOCK << ")" << std::endl;

	// Open STL file for binary writing
	FILE* stl_fp = fopen(output_path.c_str(), "wb");
	if (!stl_fp) {
		std::cerr << "Error: could not open " << output_path << " for writing" << std::endl;
		return;
	}

	// Write 80-byte header with metadata
	char header[80] = {};
	snprintf(header, 80, "type=%d res=%.4g thick=%.4g cells=%.4g",
	         tpms_input, resolution, thickness, n_cells);
	fwrite(header, 1, 80, stl_fp);
	// Write placeholder triangle count (will patch later)
	uint32_t total_triangles = 0;
	long count_pos = ftell(stl_fp);
	fwrite(&total_triangles, sizeof(uint32_t), 1, stl_fp);

	long long total_negcount = 0;
	long long total_voxels = 0;
	int blocks_done = 0;

	for (int bz = 0; bz < nz_blocks; bz++) {
		for (int by = 0; by < ny_blocks; by++) {
			for (int bx = 0; bx < nx_blocks; bx++) {
				// Global index range for this block (with 1-voxel overlap)
				int gx0 = bx * BLOCK;
				int gy0 = by * BLOCK;
				int gz0 = bz * BLOCK;
				int gx1 = std::min(gx0 + BLOCK, x_num);  // exclusive end (no overlap)
				int gy1 = std::min(gy0 + BLOCK, y_num);
				int gz1 = std::min(gz0 + BLOCK, z_num);

				// Add 1-voxel overlap at the end (if not at the global boundary)
				int gx1_ov = std::min(gx1 + 1, x_num);
				int gy1_ov = std::min(gy1 + 1, y_num);
				int gz1_ov = std::min(gz1 + 1, z_num);

				int sub_nx = gx1_ov - gx0;
				int sub_ny = gy1_ov - gy0;
				int sub_nz = gz1_ov - gz0;

				if (sub_nx < 2 || sub_ny < 2 || sub_nz < 2) continue;

				int sub_total = sub_nx * sub_ny * sub_nz;

				// Build sub-grid coordinates
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

				// Compute scalar field for this block
				Eigen::VectorXd sub_B(sub_total);
				int block_negcount = 0;

#pragma omp parallel for reduction(+:block_negcount)
				for (int i = 0; i < sub_total; i++) {
					const double x = sub_grid(i, 0);
					const double y = sub_grid(i, 1);
					const double z = sub_grid(i, 2);
					Point q(x, y, z);

					double f = tpms_function(tpms_type, omega, x, y, z);
					double phi_tpms = std::abs(f) - half_t;

					Point closest = tree.closest_point(q);
					double unsigned_d = std::sqrt(CGAL::squared_distance(q, closest));
					CGAL::Bounded_side side = inside(q);
					double d = (side == CGAL::ON_BOUNDED_SIDE) ? unsigned_d : -unsigned_d;

					double phi;
					if (d < -3.0 * smf) {
						phi = 1000.0;
					} else {
						phi = smf * std::log(std::exp(phi_tpms / smf) + std::exp(-d / smf));
					}

					sub_B(i) = phi;
					if (phi < 0.0) block_negcount++;
				}

				total_negcount += block_negcount;
				total_voxels += sub_total;

				// Marching cubes on this block
				Eigen::MatrixXd V;
				Eigen::MatrixXi F;
				igl::copyleft::marching_cubes(sub_B, sub_grid, sub_nx, sub_ny, sub_nz, V, F);

				// Stream triangles to STL file
				if (F.rows() > 0) {
					total_triangles += write_triangles_stl(stl_fp, V, F);
				}

				blocks_done++;
				double pct = 100.0 * blocks_done / total_blocks;
				std::cout << "\r  Progress: " << (int)pct << "% | Block " << blocks_done << "/" << total_blocks
				          << " | " << total_triangles << " triangles     " << std::flush;

				// V, F, sub_grid, sub_B freed here automatically
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
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " input_mesh output.stl" << std::endl;
		std::cerr << "  Supported input formats: .off, .obj, .stl, .ply" << std::endl;
		return 1;
	}
	generate(argv[1], argv[2]);
	return 0;
}

// [-1,1] models input: 0 (kP), 100 (resolution), 0.1 (thickness), 3 (n_cells)