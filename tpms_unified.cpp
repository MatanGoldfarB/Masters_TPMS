#include <CGAL/Simple_cartesian.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>
#include <CGAL/Heat_method_3/Surface_mesh_geodesic_distances_3.h>

#include <igl/copyleft/marching_cubes.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
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

enum TpmsType { kP = 0, kD = 1, kG = 2, kSteppedOmega = 3, kInterpolatedOmega = 4, kGeodesicOmega = 5 };

enum DecayType { kSuperGaussian = 0, kRational = 1 };

struct SteppedZone {
	double omega;
	double thickness;
	double boundary_d;        // cumulative distance from surface where this zone ends
};

struct SteppedOmegaParams {
	TpmsType base_type;
	std::vector<SteppedZone> zones;  // ordered from surface inward; last zone extends to infinity
};

struct ControlPoint {
	double x, y, z;
	double n_cells;
	double thickness;
};

struct InterpolatedOmegaParams {
	TpmsType base_type;
	std::vector<ControlPoint> points;
};

struct GeodesicOmegaParams {
	TpmsType base_type;
	DecayType decay_type;
	double decay_power;          // exponent p (default 3.0)
	double wavelength_scale;     // scale s in decay formula (default 1.0)
	std::vector<ControlPoint> points;
	// Per-vertex precomputed values (indexed by mesh vertex descriptor)
	std::vector<double> vertex_omega;
	std::vector<double> vertex_thickness;
	std::vector<Eigen::Vector3d> vertex_positions;  // cached for fast inner-loop iteration
};

// Decay function: returns a factor in [0,1] that multiplies the surface frequency.
// d = distance from surface, lambda = local wavelength (2*pi/omega), p = decay power, s = wavelength scale.
// SuperGaussian: exp(-(d/(s*lambda))^p)  — slow near surface, steep far away
// Rational:      1 / (1 + (d/(s*lambda))^p)
static double decay_factor(double d, double lambda, DecayType type, double p, double s)
{
	if (d <= 0.0) return 1.0;
	double r = d / (s * lambda);
	switch (type) {
	case kSuperGaussian:
		return std::exp(-std::pow(r, p));
	case kRational:
		return 1.0 / (1.0 + std::pow(r, p));
	}
	return 1.0;
}

// Snap a 3D point to the nearest vertex of the mesh.
// Uses AABB tree to find closest point on surface, then picks the nearest vertex of that face.
static SurfaceMesh::Vertex_index snap_to_nearest_vertex(
	const SurfaceMesh& mesh, const Tree& tree,
	double px, double py, double pz)
{
	Point q(px, py, pz);
	auto loc = tree.closest_point_and_primitive(q);
	// loc.second is a face_descriptor iterator
	auto fd = loc.second;

	// Find the closest vertex among this face's vertices
	double best_dist2 = std::numeric_limits<double>::max();
	SurfaceMesh::Vertex_index best_vi;
	for (auto vi : mesh.vertices_around_face(mesh.halfedge(fd))) {
		const Point& vp = mesh.point(vi);
		double dx = vp.x() - px, dy = vp.y() - py, dz = vp.z() - pz;
		double d2 = dx*dx + dy*dy + dz*dz;
		if (d2 < best_dist2) {
			best_dist2 = d2;
			best_vi = vi;
		}
	}
	return best_vi;
}

// Precompute per-vertex omega and thickness using geodesic IDW interpolation.
// Uses CGAL Heat Method to compute geodesic distances from each control point vertex,
// then IDW-blends at every mesh vertex.
static void precompute_geodesic_vertex_fields(
	SurfaceMesh& mesh, const Tree& tree,
	GeodesicOmegaParams& params, double max_dim)
{
	size_t nv = mesh.number_of_vertices();
	size_t ncp = params.points.size();

	std::cout << "  Precomputing geodesic distances (" << ncp << " sources, "
	          << nv << " vertices)..." << std::endl;

	// Snap control points to nearest vertices
	std::vector<SurfaceMesh::Vertex_index> source_verts(ncp);
	std::vector<double> cp_omega(ncp), cp_thickness(ncp);
	for (size_t i = 0; i < ncp; i++) {
		const auto& cp = params.points[i];
		source_verts[i] = snap_to_nearest_vertex(mesh, tree, cp.x, cp.y, cp.z);
		cp_omega[i] = cp.n_cells * 2.0 * M_PI / max_dim;
		cp_thickness[i] = cp.thickness;
		const Point& sp = mesh.point(source_verts[i]);
		std::cout << "    CP " << i << ": n_cells=" << cp.n_cells
		          << " snapped to vertex " << source_verts[i]
		          << " (" << sp.x() << ", " << sp.y() << ", " << sp.z() << ")" << std::endl;
	}

	// Create vertex property map for geodesic distances
	typedef SurfaceMesh::Property_map<SurfaceMesh::Vertex_index, double> Vertex_distance_map;

	// Build the Heat Method object (prefactorizes Laplacian — done once)
	typedef CGAL::Heat_method_3::Surface_mesh_geodesic_distances_3<SurfaceMesh> Heat_method;
	Heat_method heat(mesh);

	// For each control point, compute geodesic distances to all vertices
	// Store as ncp vectors of size nv
	std::vector<std::vector<double>> geo_dists(ncp, std::vector<double>(nv, 0.0));

	for (size_t ci = 0; ci < ncp; ci++) {
		// Add the source vertex and compute
		Vertex_distance_map dist_map;
		bool created;
		std::string pname = "v:geodist_" + std::to_string(ci);
		std::tie(dist_map, created) = mesh.add_property_map<SurfaceMesh::Vertex_index, double>(pname, 0.0);

		heat.clear_sources();
		heat.add_source(source_verts[ci]);
		heat.estimate_geodesic_distances(dist_map);

		// Copy to our array
		for (auto vi : mesh.vertices()) {
			geo_dists[ci][(size_t)vi] = dist_map[vi];
		}

		// Remove temporary property map
		mesh.remove_property_map(dist_map);

		std::cout << "    Heat solve " << (ci+1) << "/" << ncp << " done" << std::endl;
	}

	// IDW interpolation at each vertex using geodesic distances
	params.vertex_omega.resize(nv);
	params.vertex_thickness.resize(nv);

	for (auto vi : mesh.vertices()) {
		size_t vidx = (size_t)vi;
		double w_sum = 0.0, omega_sum = 0.0, thick_sum = 0.0;
		bool exact_hit = false;

		for (size_t ci = 0; ci < ncp; ci++) {
			double gd = geo_dists[ci][vidx];
			if (gd < 1e-12) {
				// This vertex IS the control point
				params.vertex_omega[vidx] = cp_omega[ci];
				params.vertex_thickness[vidx] = cp_thickness[ci];
				exact_hit = true;
				break;
			}
			double w = 1.0 / (gd * gd);
			omega_sum += w * cp_omega[ci];
			thick_sum += w * cp_thickness[ci];
			w_sum += w;
		}
		if (!exact_hit) {
			params.vertex_omega[vidx] = omega_sum / w_sum;
			params.vertex_thickness[vidx] = thick_sum / w_sum;
		}
	}

	// Cache vertex positions for fast inner-loop access
	params.vertex_positions.resize(nv);
	for (auto vi : mesh.vertices()) {
		const Point& vp = mesh.point(vi);
		params.vertex_positions[(size_t)vi] = Eigen::Vector3d(vp.x(), vp.y(), vp.z());
	}

	std::cout << "  Geodesic vertex field precomputation complete." << std::endl;
}

// Smooth interpolation [0,1] with zero derivative at endpoints
static double smoothstep(double edge0, double edge1, double x)
{
	if (edge1 <= edge0) return (x < edge0) ? 0.0 : 1.0;
	double t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
	return t * t * (3.0 - 2.0 * t);
}

// For N-zone stepped omega: find the two zones that bracket distance d and compute blend.
// Sets idx_a, idx_b (zone indices) and alpha (1.0 = purely zone idx_a, 0.0 = purely zone idx_b).
// When d is fully inside one zone (not in a transition), idx_a == idx_b and alpha == 1.0.
static void stepped_find_zones(double d, const SteppedOmegaParams& p,
	int& idx_a, int& idx_b, double& alpha)
{
	const auto& zones = p.zones;
	int n = static_cast<int>(zones.size());

	// Find which zone d falls into (or which transition)
	for (int i = 0; i < n - 1; i++) {
		double boundary = zones[i].boundary_d;
		// Transition width = 1 wavelength of the slower (lower omega) adjacent zone
		double tw = std::max(2.0 * M_PI / zones[i].omega, 2.0 * M_PI / zones[i+1].omega);
		double blend_start = boundary;
		double blend_end   = boundary + tw;

		if (d < blend_start) {
			// Fully inside zone i
			idx_a = i; idx_b = i; alpha = 1.0;
			return;
		}
		if (d <= blend_end) {
			// In transition between zone i and zone i+1
			idx_a = i; idx_b = i + 1;
			alpha = 1.0 - smoothstep(blend_start, blend_end, d);
			return;
		}
	}
	// Past all boundaries → last zone
	idx_a = n - 1; idx_b = n - 1; alpha = 1.0;
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
			// Coincident with control point
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

// Parse control points file. Format: one point per line, columns: n_cells x y z thickness
// Lines starting with # are comments. Returns empty vector on failure.
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
	default:
		break;
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
	std::cout << "Input TPMS type (0=kP, 1=kD, 2=kG, 3=SteppedOmega, 4=InterpolatedOmega, 5=GeodesicOmega): ";
	int tpms_input;
	std::cin >> tpms_input;
	if (tpms_input < 0 || tpms_input > 5) {
		std::cerr << "Error: invalid TPMS type" << std::endl;
		return;
	}
	TpmsType tpms_type = static_cast<TpmsType>(tpms_input);

	// Parameters for new types (populated below if needed)
	SteppedOmegaParams stepped_params = {};
	InterpolatedOmegaParams interp_params = {};
	GeodesicOmegaParams geodesic_params = {};

	// Base type for stepped/interpolated/geodesic modes
	TpmsType base_type = tpms_type;  // for kP/kD/kG, base_type == tpms_type
	if (tpms_type == kSteppedOmega || tpms_type == kInterpolatedOmega || tpms_type == kGeodesicOmega) {
		std::cout << "Input base TPMS type (0=kP, 1=kD, 2=kG): ";
		int base_input;
		std::cin >> base_input;
		if (base_input < 0 || base_input > 2) {
			std::cerr << "Error: invalid base TPMS type" << std::endl;
			return;
		}
		base_type = static_cast<TpmsType>(base_input);
	}

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

	// Global omega parameters (used by kP/kD/kG)
	double thickness = 0.0, n_cells = 0.0;
	double omega = 0.0, half_t = 0.0, smf = 1.0;

	if (tpms_type == kP || tpms_type == kD || tpms_type == kG) {
		std::cout << "Input wall thickness (full thickness in model units): ";
		std::cin >> thickness;

		std::cout << "Input number of unit cells: ";
		std::cin >> n_cells;
		omega = n_cells * 2.0 * M_PI / max_dim;
		std::cout << "  omega = " << omega << std::endl;

		half_t = (thickness / 2.0) * omega;
		double wavelength = 2.0 * M_PI / omega;
		smf = wavelength / 10.0;
	}

	// --- Stepped omega: N-zone loop ---
	if (tpms_type == kSteppedOmega) {
		stepped_params.base_type = base_type;
		double cumulative_d = 0.0;
		int zone_num = 1;

		while (true) {
			std::cout << "Zone " << zone_num << ":" << std::endl;

			std::cout << "  n_cells: ";
			double zn;
			std::cin >> zn;

			std::cout << "  thickness: ";
			double zt;
			std::cin >> zt;

			double z_omega = zn * 2.0 * M_PI / max_dim;
			double z_lambda = 2.0 * M_PI / z_omega;

			std::cout << "  cycles before next zone (0 = last zone): ";
			double n_cycles;
			std::cin >> n_cycles;

			if (n_cycles <= 0.0) {
				// Last zone — extends to infinity
				SteppedZone zone;
				zone.omega = z_omega;
				zone.thickness = zt;
				zone.boundary_d = 1e10;  // effectively infinite
				stepped_params.zones.push_back(zone);
				std::cout << "  Zone " << zone_num << ": omega=" << z_omega
				          << ", thickness=" << zt << " [final zone]" << std::endl;
				break;
			}

			cumulative_d += n_cycles * z_lambda;

			SteppedZone zone;
			zone.omega = z_omega;
			zone.thickness = zt;
			zone.boundary_d = cumulative_d;
			stepped_params.zones.push_back(zone);

			std::cout << "  Zone " << zone_num << ": omega=" << z_omega
			          << ", thickness=" << zt
			          << ", boundary_d=" << cumulative_d << std::endl;
			zone_num++;
		}

		std::cout << "  Total zones: " << stepped_params.zones.size() << std::endl;
	}

	// --- Interpolated omega: load control points from file ---
	std::string cp_path;
	if (tpms_type == kInterpolatedOmega) {
		interp_params.base_type = base_type;
		std::cout << "Input control points file path: ";
		std::cin >> cp_path;
		interp_params.points = parse_control_points(cp_path);
		if (interp_params.points.empty()) {
			std::cerr << "Error: no control points loaded" << std::endl;
			return;
		}
	}

	// --- Geodesic omega: load control points, compute geodesic surface field ---
	if (tpms_type == kGeodesicOmega) {
		geodesic_params.base_type = base_type;

		std::cout << "Input control points file path: ";
		std::cin >> cp_path;
		geodesic_params.points = parse_control_points(cp_path);
		if (geodesic_params.points.empty()) {
			std::cerr << "Error: no control points loaded" << std::endl;
			return;
		}

		std::cout << "Input decay type (0=SuperGaussian, 1=Rational) [default 0]: ";
		int decay_input;
		std::cin >> decay_input;
		geodesic_params.decay_type = (decay_input == 1) ? kRational : kSuperGaussian;

		std::cout << "Input decay power p (e.g. 3.0 = slow near surface, steep far) [default 3]: ";
		std::cin >> geodesic_params.decay_power;
		if (geodesic_params.decay_power <= 0.0) geodesic_params.decay_power = 3.0;

		std::cout << "Input wavelength scale s (how many wavelengths before significant decay) [default 1]: ";
		std::cin >> geodesic_params.wavelength_scale;
		if (geodesic_params.wavelength_scale <= 0.0) geodesic_params.wavelength_scale = 1.0;

		std::cout << "  Decay: " << (geodesic_params.decay_type == kSuperGaussian ? "SuperGaussian" : "Rational")
		          << ", p=" << geodesic_params.decay_power
		          << ", s=" << geodesic_params.wavelength_scale << std::endl;

		// Precompute per-vertex omega and thickness using Heat Method geodesic distances
		precompute_geodesic_vertex_fields(mesh, tree, geodesic_params, max_dim);
	}

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
	if (tpms_type == kSteppedOmega) {
		// Build zone tuples string: (n,t,cycles) for each zone
		std::string zones_str;
		for (size_t i = 0; i < stepped_params.zones.size(); i++) {
			const auto& z = stepped_params.zones[i];
			double n = z.omega * max_dim / (2.0 * M_PI);
			char buf[40];
			if (i + 1 < stepped_params.zones.size()) {
				double prev_bd = (i == 0) ? 0.0 : stepped_params.zones[i-1].boundary_d;
				double cycles = (z.boundary_d - prev_bd) / (2.0 * M_PI / z.omega);
				snprintf(buf, sizeof(buf), "(%.4g,%.4g,%.4g)", n, z.thickness, cycles);
			} else {
				snprintf(buf, sizeof(buf), "(%.4g,%.4g,0)", n, z.thickness);
			}
			zones_str += buf;
		}
		snprintf(header, 80, "type=3 base=%d res=%.4g %s",
		         (int)stepped_params.base_type, resolution, zones_str.c_str());
	} else if (tpms_type == kInterpolatedOmega) {
		snprintf(header, 80, "type=4 base=%d res=%.4g %s",
		         (int)interp_params.base_type, resolution, cp_path.c_str());
	} else if (tpms_type == kGeodesicOmega) {
		snprintf(header, 80, "type=5 base=%d res=%.4g p=%.2g s=%.2g",
		         (int)geodesic_params.base_type, resolution,
		         geodesic_params.decay_power, geodesic_params.wavelength_scale);
	} else {
		snprintf(header, 80, "type=%d res=%.4g thick=%.4g cells=%.4g",
		         tpms_input, resolution, thickness, n_cells);
	}
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

					// Compute signed distance to boundary mesh
					auto cp_and_prim = tree.closest_point_and_primitive(q);
					Point closest = cp_and_prim.first;
					auto closest_face = cp_and_prim.second;
					double unsigned_d = std::sqrt(CGAL::squared_distance(q, closest));
					CGAL::Bounded_side side = inside(q);
					double d = (side == CGAL::ON_BOUNDED_SIDE) ? unsigned_d : -unsigned_d;

					double phi_tpms, local_smf;

					if (tpms_type == kSteppedOmega) {
						// Find the two zones that bracket this distance
						int idx_a, idx_b;
						double alpha;
						stepped_find_zones(d, stepped_params, idx_a, idx_b, alpha);

						const auto& za = stepped_params.zones[idx_a];
						double fa = tpms_function(stepped_params.base_type, za.omega, x, y, z);
						double hta = (za.thickness / 2.0) * za.omega;
						double phi_a = std::abs(fa) - hta;

						if (idx_a == idx_b) {
							// Fully inside one zone
							phi_tpms = phi_a;
							local_smf = (2.0 * M_PI / za.omega) / 10.0;
						} else {
							// Blending between two zones
							const auto& zb = stepped_params.zones[idx_b];
							double fb = tpms_function(stepped_params.base_type, zb.omega, x, y, z);
							double htb = (zb.thickness / 2.0) * zb.omega;
							double phi_b = std::abs(fb) - htb;

							phi_tpms = alpha * phi_a + (1.0 - alpha) * phi_b;

							double local_omega = alpha * za.omega + (1.0 - alpha) * zb.omega;
							local_smf = (2.0 * M_PI / local_omega) / 10.0;
						}

					} else if (tpms_type == kInterpolatedOmega) {
						auto [local_omega, local_thick] = idw_interpolate(x, y, z, interp_params.points, max_dim);
						double local_half_t = (local_thick / 2.0) * local_omega;
						double f = tpms_function(interp_params.base_type, local_omega, x, y, z);
						phi_tpms = std::abs(f) - local_half_t;
						local_smf = (2.0 * M_PI / local_omega) / 10.0;

					} else if (tpms_type == kGeodesicOmega) {
						// Smooth volumetric IDW from all mesh vertices.
						// Avoids medial-axis discontinuity that closest-face projection causes.
						size_t nv = geodesic_params.vertex_positions.size();
						double w_sum = 0.0, omega_sum = 0.0, thick_sum = 0.0;
						for (size_t vi = 0; vi < nv; vi++) {
							const Eigen::Vector3d& vp = geodesic_params.vertex_positions[vi];
							double dx = x - vp.x(), dy = y - vp.y(), dz = z - vp.z();
							double dist2 = dx*dx + dy*dy + dz*dz;
							if (dist2 < 1e-12) {
								w_sum = 1.0;
								omega_sum = geodesic_params.vertex_omega[vi];
								thick_sum = geodesic_params.vertex_thickness[vi];
								break;
							}
							double w = 1.0 / dist2;
							omega_sum += w * geodesic_params.vertex_omega[vi];
							thick_sum += w * geodesic_params.vertex_thickness[vi];
							w_sum += w;
						}
						double omega_surf = omega_sum / w_sum;
						double thick_surf = thick_sum / w_sum;

						// Apply decay based on distance from surface
						double lambda = 2.0 * M_PI / omega_surf;
						double df = decay_factor(unsigned_d, lambda,
							geodesic_params.decay_type, geodesic_params.decay_power,
							geodesic_params.wavelength_scale);
						double local_omega = omega_surf * df;
						double local_thick = thick_surf;  // thickness stays constant (larger cells = thicker walls)

						// Clamp omega to avoid degenerate zero-frequency
						double min_omega = 0.01;
						if (local_omega < min_omega) {
							// Deep interior — effectively no TPMS, force exterior
							phi_tpms = 1000.0;
							local_smf = 1.0;
						} else {
							double local_half_t = (local_thick / 2.0) * local_omega;
							double f = tpms_function(geodesic_params.base_type, local_omega, x, y, z);
							phi_tpms = std::abs(f) - local_half_t;
							local_smf = (2.0 * M_PI / local_omega) / 10.0;
						}

					} else {
						// Standard kP/kD/kG
						double f = tpms_function(tpms_type, omega, x, y, z);
						phi_tpms = std::abs(f) - half_t;
						local_smf = smf;
					}

					double phi;
					if (d < -3.0 * local_smf) {
						phi = 1000.0;
					} else {
						// Log-sum-exp trick to avoid overflow
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

// [-1,1] models input: 0 (kP), 100 (resolution), 0.1 (thickness), 3 (n_cells) / 3-stepped, 0-kp, 150-res, (8,0.05,2), (5,0.1,0)
// 5-geodesic, 0-kP, 50-res, control_points_geodesic_test.txt, 0-SuperGaussian, 3-power, 1-scale