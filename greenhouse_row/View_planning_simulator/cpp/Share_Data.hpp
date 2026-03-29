#pragma once
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <direct.h>
#include <fstream> 
#include <stdio.h>  
#include <cstdint>
#include <string>  
#include <vector> 
#include <thread>
#include <chrono>
#include <atomic>
#include <random>
#include <limits>
#include <ctime> 
#include <cmath>
#include <mutex>
#include <array>
#include <map>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/point_tests.h>
#include <pcl/common/concatenate.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

using namespace std;

struct OBB {
	Eigen::Vector3d c;       // center
	Eigen::Matrix3d R;       // columns are box axes in world
	Eigen::Vector3d extent;  // full lengths

	Eigen::Vector3d half() const { return 0.5 * extent; }

	bool contains(const Eigen::Vector3d& p, double eps = 1e-6) const {
		Eigen::Vector3d q = R.transpose() * (p - c); // local coords
		Eigen::Vector3d h = half();
		return (std::abs(q.x()) <= h.x() + eps) &&
			(std::abs(q.y()) <= h.y() + eps) &&
			(std::abs(q.z()) <= h.z() + eps);
	}

	std::array<Eigen::Vector3d, 8> corners() const {
		Eigen::Vector3d h = half();
		std::array<Eigen::Vector3d, 8> pts;
		int k = 0;
		for (int sx : {-1, 1})
			for (int sy : {-1, 1})
				for (int sz : {-1, 1}) {
					Eigen::Vector3d local(sx * h.x(), sy * h.y(), sz * h.z());
					pts[k++] = c + R * local;
				}
		return pts;
	}
};

inline std::vector<double> readVecD(const cv::FileNode& n, const char* key) {
	cv::FileNode v = n[key];
	if (v.empty() || !v.isSeq()) {
		throw std::runtime_error(std::string("YAML missing or not a sequence: ") + key);
	}
	std::vector<double> out;
	out.reserve((size_t)v.size());
	for (auto it = v.begin(); it != v.end(); ++it) out.push_back((double)*it);
	return out;
}

inline OBB load_obb_opencv(const std::string& yaml_path, const std::string& node_name) {
	cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		throw std::runtime_error("Cannot open yaml: " + yaml_path);
	}

	cv::FileNode nd = fs[node_name];
	if (nd.empty()) {
		throw std::runtime_error("Node not found in yaml: " + node_name);
	}

	std::vector<double> cvec = readVecD(nd, "center");
	std::vector<double> evec = readVecD(nd, "extent");
	std::vector<double> rflat = readVecD(nd, "R_flat");
	fs.release();

	if (cvec.size() != 3)  throw std::runtime_error(node_name + ".center size != 3");
	if (evec.size() != 3)  throw std::runtime_error(node_name + ".extent size != 3");
	if (rflat.size() != 9) throw std::runtime_error(node_name + ".R_flat size != 9");

	OBB box;
	box.c = Eigen::Vector3d(cvec[0], cvec[1], cvec[2]);
	box.extent = Eigen::Vector3d(evec[0], evec[1], evec[2]);

	// row-major -> Matrix3d
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Rrm;
	for (int i = 0; i < 9; ++i) Rrm(i / 3, i % 3) = rflat[(size_t)i];
	box.R = Rrm;

	// 可选：简单检查（调试时很有用）
	// double det = box.R.determinant();
	// Eigen::Matrix3d I = box.R * box.R.transpose();
	// if (std::abs(det - 1.0) > 1e-3 || (I - Eigen::Matrix3d::Identity()).norm() > 1e-2) {
	//   throw std::runtime_error(node_name + ": R is not a proper rotation matrix");
	// }

	return box;
}

//体素滤波
inline pcl::PointCloud<pcl::PointXYZ>::Ptr voxelDownsample(
	const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& in,
	float voxel_size)
{
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud(in);
	vg.setLeafSize(voxel_size, voxel_size, voxel_size);

	auto out = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	vg.filter(*out);
	return out;
}
inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxelDownsample(
	const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& in,
	float voxel_size)
{
	pcl::VoxelGrid<pcl::PointXYZRGB> vg;
	vg.setInputCloud(in);
	vg.setLeafSize(voxel_size, voxel_size, voxel_size);

	auto out = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	vg.filter(*out);
	return out;
}

//FPS采样
inline std::vector<Eigen::Vector3d> farthestPointSampling(
	const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
	int K,
	std::mt19937& rng)
{
	std::vector<Eigen::Vector3d> pts;
	pts.reserve(cloud->size());
	for (const auto& p : cloud->points) pts.emplace_back(p.x, p.y, p.z);

	const int N = static_cast<int>(pts.size());
	std::vector<Eigen::Vector3d> targets;
	if (N == 0 || K <= 0) return targets;

	K = std::min(K, N);
	targets.reserve(K);

	std::uniform_int_distribution<int> uni(0, N - 1);
	int first = uni(rng);
	targets.push_back(pts[first]);

	std::vector<double> minDist2(N, std::numeric_limits<double>::infinity());

	auto updateMinDist = [&](const Eigen::Vector3d& sel) {
		for (int i = 0; i < N; ++i) {
			double d2 = (pts[i] - sel).squaredNorm();
			if (d2 < minDist2[i]) minDist2[i] = d2;
		}
		};

	updateMinDist(targets.back());

	for (int t = 1; t < K; ++t) {
		int farthest_i = 0;
		double best = -1.0;
		for (int i = 0; i < N; ++i) {
			if (minDist2[i] > best) {
				best = minDist2[i];
				farthest_i = i;
			}
		}
		targets.push_back(pts[farthest_i]);
		updateMinDist(targets.back());
	}

	return targets;
}

//Round-robin分配
struct ViewAssignResult {
	std::vector<int> used_idx;              // 选中的 candidate 下标（长度 M）
	std::vector<int> assigned_target_idx;   // 与 used_idx 对齐，每个 candidate 对应 target id
	std::vector<Eigen::Vector3d> lookat;    // 与 used_idx 对齐，每个 candidate 的 look-at 坐标
};
inline ViewAssignResult assignViewsRoundRobin(
	const std::vector<Eigen::Vector3d>& candidates,
	const std::vector<Eigen::Vector3d>& targets,
	int num_rounds,
	std::mt19937& rng)
{
	ViewAssignResult out;
	const int N = static_cast<int>(candidates.size());
	const int M = static_cast<int>(targets.size());
	if (N == 0 || M == 0 || num_rounds <= 0) return out;

	std::vector<char> available(N, 1);
	std::vector<int> assigned_target(N, -1);

	std::vector<int> order(M);
	std::iota(order.begin(), order.end(), 0);

	for (int r = 0; r < num_rounds; ++r) {
		std::shuffle(order.begin(), order.end(), rng);

		for (int j : order) {
			int best_i = -1;
			double best_d2 = std::numeric_limits<double>::infinity();

			for (int i = 0; i < N; ++i) {
				if (!available[i]) continue;
				double d2 = (candidates[i] - targets[j]).squaredNorm();
				if (d2 < best_d2) {
					best_d2 = d2;
					best_i = i;
				}
			}

			if (best_i < 0) break; // 没 candidate 了
			available[best_i] = 0;
			assigned_target[best_i] = j;
		}

		bool anyAvail = false;
		for (char a : available) { if (a) { anyAvail = true; break; } }
		if (!anyAvail) break;
	}

	for (int i = 0; i < N; ++i) {
		if (assigned_target[i] >= 0) {
			out.used_idx.push_back(i);
			out.assigned_target_idx.push_back(assigned_target[i]);
			out.lookat.push_back(targets[assigned_target[i]]);
		}
	}

	// 可选：打印统计
	std::vector<int> cnt(M, 0);
	for (int t : out.assigned_target_idx) cnt[t]++;
	std::cout << "Candidates: " << N << ", selected: " << out.used_idx.size() << "\n";
	//for (int j = 0; j < M; ++j) {
	//	if (cnt[j] > 0) std::cout << "  target " << j << ": " << cnt[j] << " views\n";
	//}

	return out;
}

//chamfer_distance_pcl
static inline pcl::PointCloud<pcl::PointXYZ>::Ptr toXYZ(
	const pcl::PointCloud<pcl::PointXYZRGB>::Ptr in)
{
	auto out = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	out->reserve(in->size());
	for (const auto& p : in->points) {
		out->push_back(pcl::PointXYZ(p.x, p.y, p.z));
	}
	out->width = static_cast<uint32_t>(out->size());
	out->height = 1;
	out->is_dense = false;
	return out;
}

static inline pcl::PointCloud<pcl::PointXYZ>::Ptr voxelDownsampleXYZ(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr in,
	float leaf)
{
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setLeafSize(leaf, leaf, leaf);
	vg.setInputCloud(in);
	auto out = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	vg.filter(*out);
	return out;
}

// 计算：对 src 每个点，找 tgt 最近邻距离，并取均值
static inline double meanNearestNeighborDistance(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr src,
	const pcl::PointCloud<pcl::PointXYZ>::Ptr tgt,
	double truncate_dist = -1.0) // >0 则对每个距离做截断(robust)
{
	if (!src || !tgt || src->empty() || tgt->empty()) return 0.0;

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(tgt);

	std::vector<int> nn_idx(1);
	std::vector<float> nn_d2(1);

	double sum = 0.0;
	int cnt = 0;

	for (const auto& p : src->points) {
		if (!pcl::isFinite(p)) continue;
		if (kdtree.nearestKSearch(p, 1, nn_idx, nn_d2) > 0) {
			double d = std::sqrt(static_cast<double>(nn_d2[0]));
			if (truncate_dist > 0.0 && d > truncate_dist) d = truncate_dist;
			sum += d;
			cnt++;
		}
	}

	return (cnt > 0) ? (sum / cnt) : 0.0;
}

static inline double chamferDistance(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr A_rgb,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr B_rgb,
    float voxel_leaf = -1.0f,      // >0 开启体素下采样
    double truncate_dist = -1.0)   // >0 截断距离，增强鲁棒性
{
    if (!A_rgb || !B_rgb || A_rgb->empty() || B_rgb->empty()) return 0.0;

    auto A = toXYZ(A_rgb);
    auto B = toXYZ(B_rgb);

    if (voxel_leaf > 0.0f) {
        A = voxelDownsampleXYZ(A, voxel_leaf);
        B = voxelDownsampleXYZ(B, voxel_leaf);
        if (A->empty() || B->empty()) return 0.0;
    }

    const double dA2B = meanNearestNeighborDistance(A, B, truncate_dist);
    const double dB2A = meanNearestNeighborDistance(B, A, truncate_dist);
    return 0.5 * (dA2B + dB2A);
}

// ===== Voxel key (int grid) =====
struct VoxelKey {
	int ix, iy, iz;
	bool operator==(const VoxelKey& o) const noexcept {
		return ix == o.ix && iy == o.iy && iz == o.iz;
	}
};
struct VoxelKeyHash {
	std::size_t operator()(const VoxelKey& k) const noexcept {
		// 64-bit mix (works fine in practice)
		// Convert to uint64_t to avoid UB on shifts.
		uint64_t x = (uint64_t)(uint32_t)k.ix;
		uint64_t y = (uint64_t)(uint32_t)k.iy;
		uint64_t z = (uint64_t)(uint32_t)k.iz;
		uint64_t h = x * 0x9E3779B185EBCA87ULL ^ y * 0xC2B2AE3D27D4EB4FULL ^ z * 0x165667B19E3779F9ULL;
		// final avalanche
		h ^= (h >> 33);
		h *= 0xff51afd7ed558ccdULL;
		h ^= (h >> 33);
		h *= 0xc4ceb9fe1a85ec53ULL;
		h ^= (h >> 33);
		return (std::size_t)h;
	}
};
static inline VoxelKey ToKey(const pcl::PointXYZ& p, float voxel) {
	// floor quantization (handles negative coords correctly)
	return VoxelKey{
	  (int)std::floor(p.x / voxel),
	  (int)std::floor(p.y / voxel),
	  (int)std::floor(p.z / voxel)
	};
}
static inline pcl::PointXYZ KeyToCenter(const VoxelKey& k, float voxel) {
	// center of voxel cell
	return pcl::PointXYZ(
		(k.ix + 0.5f) * voxel,
		(k.iy + 0.5f) * voxel,
		(k.iz + 0.5f) * voxel
	);
}
// ROI check on voxel index (optional)
static inline bool InRoi(const VoxelKey& k,
	const VoxelKey& min_k,
	const VoxelKey& max_k) {
	return (k.ix >= min_k.ix && k.ix <= max_k.ix &&
		k.iy >= min_k.iy && k.iy <= max_k.iy &&
		k.iz >= min_k.iz && k.iz <= max_k.iz);
}
// Build integer offsets within a sphere of radius r (meters), in voxel units.
static inline std::vector<VoxelKey> MakeSphereOffsets(float radius, float voxel) {
	std::vector<VoxelKey> off;
	if (radius <= 0.f) {
		off.push_back({ 0,0,0 });
		return off;
	}
	const int R = (int)std::ceil(radius / voxel);
	const float r2 = (radius / voxel) * (radius / voxel) + 1e-6f; // in voxel^2
	off.reserve((2 * R + 1) * (2 * R + 1) * (2 * R + 1));
	for (int dx = -R; dx <= R; ++dx)
		for (int dy = -R; dy <= R; ++dy)
			for (int dz = -R; dz <= R; ++dz) {
				const float d2 = float(dx * dx + dy * dy + dz * dz);
				if (d2 <= r2) off.push_back({ dx,dy,dz });
			}
	return off;
}
static inline void VoxelizeCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
	float voxel,
	std::unordered_set<VoxelKey, VoxelKeyHash>& out_set,
	bool use_roi,
	const VoxelKey& roi_min,
	const VoxelKey& roi_max) {
	out_set.clear();
	out_set.reserve(cloud->size() * 2);

	for (const auto& p : cloud->points) {
		if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
		VoxelKey k = ToKey(p, voxel);
		if (use_roi && !InRoi(k, roi_min, roi_max)) continue;
		out_set.insert(k);
	}
}
static inline void DilateSet(const std::unordered_set<VoxelKey, VoxelKeyHash>& in_set,
	const std::vector<VoxelKey>& offsets,
	std::unordered_set<VoxelKey, VoxelKeyHash>& out_set,
	bool use_roi,
	const VoxelKey& roi_min,
	const VoxelKey& roi_max) {
	out_set.clear();
	out_set.reserve(in_set.size() * 8); // rough guess

	for (const auto& k : in_set) {
		for (const auto& o : offsets) {
			VoxelKey nk{ k.ix + o.ix, k.iy + o.iy, k.iz + o.iz };
			if (use_roi && !InRoi(nk, roi_min, roi_max)) continue;
			out_set.insert(nk);
		}
	}
}
// out = A \ B
static inline void SetDifference(const std::unordered_set<VoxelKey, VoxelKeyHash>& A,
	const std::unordered_set<VoxelKey, VoxelKeyHash>& B,
	std::unordered_set<VoxelKey, VoxelKeyHash>& out) {
	out.clear();
	out.reserve(A.size());
	for (const auto& k : A) {
		if (B.find(k) == B.end()) out.insert(k);
	}
}

//get normals using KdTreeFLANN
static pcl::PointCloud<pcl::PointNormal>::Ptr
EstimateNormals_KdTree(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_out,
	Eigen::Vector3d viewpoint = Eigen::Vector3d(0, 0, 0),
	int max_nn = 30)
{
	auto normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud_out);

	//用 pcl::search::KdTree（内部实现就是 FLANN KDTree）
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setKSearch(max_nn);
	ne.setViewPoint(viewpoint(0), viewpoint(1), viewpoint(2));

	ne.compute(*normals);

	auto cloud_with_normals =
		pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>());
	pcl::concatenateFields(*cloud_out, *normals, *cloud_with_normals);

	return cloud_with_normals;
}
// 输入 XYZRGB，输出 XYZRGBNormal（颜色保留 + normal）
static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr
EstimateNormals_KdTree(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud_in,
	const Eigen::Vector3d& viewpoint = Eigen::Vector3d(0, 0, 0),
	int max_nn = 30)
{
	auto normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());

	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setInputCloud(cloud_in);

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	ne.setSearchMethod(tree);
	ne.setKSearch(max_nn);
	ne.setViewPoint(viewpoint(0), viewpoint(1), viewpoint(2));

	ne.compute(*normals);

	auto cloud_rgbn = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::concatenateFields(*cloud_in, *normals, *cloud_rgbn);

	// 可选：过滤 NaN normals，避免后续 ICP/写文件出问题
	std::vector<int> idx;
	pcl::removeNaNNormalsFromPointCloud(*cloud_rgbn, *cloud_rgbn, idx);

	return cloud_rgbn;
}

void savePLY_XYZNormal_ASCII(const std::string& path,
	const pcl::PointCloud<pcl::PointNormal>& cloud)
{
	std::ofstream f(path);
	f << "ply\nformat ascii 1.0\n";
	f << "element vertex " << cloud.size() << "\n";
	f << "property float x\nproperty float y\nproperty float z\n";
	f << "property float nx\nproperty float ny\nproperty float nz\n";
	f << "end_header\n";
	for (const auto& p : cloud.points) {
		f << p.x << " " << p.y << " " << p.z << " "
			<< p.normal_x << " " << p.normal_y << " " << p.normal_z << "\n";
	}
}

//----------------- 以下代码摘自 librealsense2/include/librealsense2/h/rs_types.h -----------------
/** \brief Distortion model: defines how pixel coordinates should be mapped to sensor coordinates. */
typedef enum rs2_distortion
{
	RS2_DISTORTION_NONE, /**< Rectilinear images. No distortion compensation required. */
	RS2_DISTORTION_MODIFIED_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except that tangential distortion is applied to radially distorted points */
	RS2_DISTORTION_INVERSE_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except undistorts image instead of distorting it */
	RS2_DISTORTION_FTHETA, /**< F-Theta fish-eye distortion model */
	RS2_DISTORTION_BROWN_CONRADY, /**< Unmodified Brown-Conrady distortion model */
	RS2_DISTORTION_KANNALA_BRANDT4, /**< Four parameter Kannala Brandt distortion model */
	RS2_DISTORTION_COUNT                   /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
} rs2_distortion;

/** \brief Video stream intrinsics. */
typedef struct rs2_intrinsics
{
	int           width;     /**< Width of the image in pixels */
	int           height;    /**< Height of the image in pixels */
	float         ppx;       /**< Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
	float         ppy;       /**< Vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
	float         fx;        /**< Focal length of the image plane, as a multiple of pixel width */
	float         fy;        /**< Focal length of the image plane, as a multiple of pixel height */
	rs2_distortion model;    /**< Distortion model of the image */
	float         coeffs[5]; /**< Distortion coefficients */
} rs2_intrinsics;

/* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
static void rs2_project_point_to_pixel(float pixel[2], const struct rs2_intrinsics* intrin, const float point[3])
{
	float x = point[0] / point[2], y = point[1] / point[2];

	if ((intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY) ||
		(intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY))
	{

		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		x *= f;
		y *= f;
		float dx = x + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float dy = y + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = dx;
		y = dy;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float rd = (float)(1.0f / intrin->coeffs[0] * atan(2 * r * tan(intrin->coeffs[0] / 2.0f)));
		x *= rd / r;
		y *= rd / r;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float theta = atan(r);
		float theta2 = theta * theta;
		float series = 1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])));
		float rd = theta * series;
		x *= rd / r;
		y *= rd / r;
	}

	pixel[0] = x * intrin->fx + intrin->ppx;
	pixel[1] = y * intrin->fy + intrin->ppy;
}

/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth)
{
	assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
	//assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

	float x = (pixel[0] - intrin->ppx) / intrin->fx;
	float y = (pixel[1] - intrin->ppy) / intrin->fy;
	if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
	{
		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		float ux = x * f + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float uy = y * f + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = ux;
		y = uy;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}

		float theta = rd;
		float theta2 = rd * rd;
		for (int i = 0; i < 4; i++)
		{
			float f = theta * (1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])))) - rd;
			if (abs(f) < FLT_EPSILON)
			{
				break;
			}
			float df = 1 + theta2 * (3 * intrin->coeffs[0] + theta2 * (5 * intrin->coeffs[1] + theta2 * (7 * intrin->coeffs[2] + 9 * theta2 * intrin->coeffs[3])));
			theta -= f / df;
			theta2 = theta * theta;
		}
		float r = tan(theta);
		x *= r / rd;
		y *= r / rd;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}
		float r = (float)(tan(intrin->coeffs[0] * rd) / atan(2 * tan(intrin->coeffs[0] / 2.0f)));
		x *= r / rd;
		y *= r / rd;
	}

	point[0] = depth * x;
	point[1] = depth * y;
	point[2] = depth;
}

//从像素坐标反投影到相机坐标系下的3D点
inline Eigen::Vector3d project_pixel_to_ray_end_eigen(float x, float y, rs2_intrinsics& color_intrinsics, Eigen::Matrix4d& now_camera_pose_world, float max_range) {
	float pixel[2] = { x ,y };
	float point[3];
	rs2_deproject_pixel_to_point(point, &color_intrinsics, pixel, max_range);
	Eigen::Vector4d point_world(point[0], point[1], point[2], 1);
	point_world = now_camera_pose_world * point_world;
	return Eigen::Vector3d(point_world(0), point_world(1), point_world(2));
}

// P-noise SE(3) Gaussian perturbation (Eigen)
// - Translation noise: N(0, sigma_t^2 I) in meters
// - Rotation noise:    axis-angle in so(3), omega ~ N(0, sigma_r^2 I) in radians
// - Right-multiply:    T_noisy = T_GT * DeltaT
namespace se3_noise {

	// Skew-symmetric matrix [w]_x
	inline Eigen::Matrix3d Skew(const Eigen::Vector3d& w) {
		Eigen::Matrix3d W;
		W << 0.0, -w.z(), w.y(),
			w.z(), 0.0, -w.x(),
			-w.y(), w.x(), 0.0;
		return W;
	}

	// SO(3) exponential map: Exp([w]_x) via Rodrigues
	inline Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w) {
		const double theta = w.norm();
		const Eigen::Matrix3d W = Skew(w);

		if (theta < 1e-12) {
			// First-order approximation
			return Eigen::Matrix3d::Identity() + W;
		}

		const double a = std::sin(theta) / theta;
		const double b = (1.0 - std::cos(theta)) / (theta * theta);
		return Eigen::Matrix3d::Identity() + a * W + b * (W * W);
	}

	// Sample a 3D Gaussian vector N(0, sigma^2 I)
	inline Eigen::Vector3d SampleGaussian3(std::mt19937& rng, double sigma) {
		std::normal_distribution<double> nd(0.0, sigma);
		return Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
	}

	// Build SE(3) perturbation DeltaT = [DeltaR, dt]
	inline Eigen::Isometry3d SampleDeltaT(std::mt19937& rng,
		double sigma_t_m,
		double sigma_r_rad) {
		const Eigen::Vector3d dt = SampleGaussian3(rng, sigma_t_m);
		const Eigen::Vector3d w = SampleGaussian3(rng, sigma_r_rad);

		Eigen::Isometry3d dT = Eigen::Isometry3d::Identity();
		dT.linear() = ExpSO3(w);   // DeltaR
		dT.translation() = dt;     // deltat
		return dT;
	}

	// Right-multiply noise: T_noisy = T_GT * DeltaT
	inline Eigen::Isometry3d ApplyRightNoise(const Eigen::Isometry3d& T_GT,
		const Eigen::Isometry3d& DeltaT) {
		return T_GT * DeltaT;
	}

	// If your point cloud is already in WORLD frame and you want to "misplace" it
	// according to noisy pose, apply the *relative* transform:
	//   DeltaT_world = T_noisy * inv(T_GT) = T_GT * DeltaT * inv(T_GT)
	inline Eigen::Isometry3d WorldRelTransformFromRightNoise(const Eigen::Isometry3d& T_GT,
		const Eigen::Isometry3d& DeltaT) {
		return T_GT * DeltaT * T_GT.inverse();
	}

} // namespace se3_noise


/**
 * Build 10 ordered blocks = (5 row segments on current layer) + (5 row segments on the other layer).
 *
 * blocks[0..4] : current layer, starting from current row segment and walking to the row end (with wrap-around)
 * blocks[5..9] : other layer, same row order (or reversed if serpentine_other_layer=true)
 *
 * row_axis_idx   : OBB local axis used as crop-row direction (usually longest axis)
 * layer_axis_idx : OBB local axis used to split upper/lower layer
 */
inline int clampInt(int x, int lo, int hi) {
	return std::max(lo, std::min(x, hi));
}
std::vector<std::vector<int>> buildOrderedRowLayerBlocks(
	const OBB& obb,
	const std::vector<Eigen::Vector3d>& candidates,
	const Eigen::Vector3d& current_view,
	int row_axis_idx,                 // e.g. 0
	int layer_axis_idx,               // e.g. 2
	int num_rows = 5,                 // fixed to 5 for your case
	bool forward_to_row_max = true,   // current layer walk direction
	bool serpentine_other_layer = false, // reverse row order on the other layer
	double layer_split = 0.0,         // local-coordinate threshold for upper/lower
	double eps = 1e-6)
{
	// We internally classify into raw buckets: [layer(0/1)][row(0..num_rows-1)]
	std::vector<std::vector<int>> raw_blocks(2 * num_rows);

	const Eigen::Vector3d h = obb.half();

	// Helper: classify a point into (row_idx, layer_idx)
	auto classifyRowLayer = [&](const Eigen::Vector3d& p, int& row_idx, int& layer_idx) -> bool {
		if (!obb.contains(p, eps)) return false;

		Eigen::Vector3d q = obb.R.transpose() * (p - obb.c);

		// Row index by equal partition along row axis
		double u = q(row_axis_idx);
		double u_min = -h(row_axis_idx);
		double u_max = h(row_axis_idx);
		double du = (u_max - u_min) / static_cast<double>(num_rows);

		if (du <= 1e-12) return false; // degenerate case

		row_idx = static_cast<int>(std::floor((u - u_min) / du));
		row_idx = clampInt(row_idx, 0, num_rows - 1);

		// Layer index by sign/split along layer axis
		double v = q(layer_axis_idx);
		layer_idx = (v >= layer_split) ? 1 : 0;  // 1=upper, 0=lower (you can swap if needed)

		return true;
		};

	// 1) Classify all candidate views into raw row-layer buckets
	for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
		int row_idx = 0, layer_idx = 0;
		if (!classifyRowLayer(candidates[i], row_idx, layer_idx)) continue;
		int raw_id = layer_idx * num_rows + row_idx;
		raw_blocks[raw_id].push_back(i);
	}

	// 2) Determine current row/layer from current_view
	int curr_row = 0, curr_layer = 1; // defaults
	{
		int row_idx = 0, layer_idx = 0;
		if (classifyRowLayer(current_view, row_idx, layer_idx)) {
			curr_row = row_idx;
			curr_layer = layer_idx;
		}
		else {
			// If current_view is outside OBB, project to local coords anyway for robustness
			Eigen::Vector3d q = obb.R.transpose() * (current_view - obb.c);

			double u = q(row_axis_idx);
			double u_min = -h(row_axis_idx);
			double u_max = h(row_axis_idx);
			double du = (u_max - u_min) / static_cast<double>(num_rows);
			if (du > 1e-12) {
				curr_row = static_cast<int>(std::floor((u - u_min) / du));
				curr_row = clampInt(curr_row, 0, num_rows - 1);
			}
			else {
				curr_row = 0;
			}

			curr_layer = (q(layer_axis_idx) >= layer_split) ? 1 : 0;
		}
	}

	const int other_layer = 1 - curr_layer;

	// 3) Build row traversal order starting from current row
	std::vector<int> row_order;
	row_order.reserve(num_rows);

	if (forward_to_row_max) {
		for (int r = curr_row; r < num_rows; ++r) row_order.push_back(r);
		for (int r = 0; r < curr_row; ++r) row_order.push_back(r);
	}
	else {
		for (int r = curr_row; r >= 0; --r) row_order.push_back(r);
		for (int r = num_rows - 1; r > curr_row; --r) row_order.push_back(r);
	}

	std::vector<int> row_order_other = row_order;
	if (serpentine_other_layer) {
		std::reverse(row_order_other.begin(), row_order_other.end());
	}

	// 4) Output already-ordered blocks[0..2*num_rows-1]
	std::vector<std::vector<int>> ordered_blocks(2 * num_rows);

	// blocks[0..4] = current layer
	for (int k = 0; k < num_rows; ++k) {
		int r = row_order[k];
		int raw_id = curr_layer * num_rows + r;
		ordered_blocks[k] = std::move(raw_blocks[raw_id]);
	}

	// blocks[5..9] = other layer
	for (int k = 0; k < num_rows; ++k) {
		int r = row_order_other[k];
		int raw_id = other_layer * num_rows + r;
		ordered_blocks[num_rows + k] = std::move(raw_blocks[raw_id]);
	}

	return ordered_blocks;
}

// 画 OBB 线框（12条边）
void AddObbWireframe(
	pcl::visualization::PCLVisualizer::Ptr& viewer,
	const OBB& obb,
	const std::string& id_prefix,
	double r = 1.0, double g = 60.0 / 255.0, double b = 60.0 / 255.0,
	int line_width = 3
) {
	const auto corners = obb.corners();

	auto addEdge = [&](int i, int j, int k) {
		pcl::PointXYZ p1((float)corners[i].x(), (float)corners[i].y(), (float)corners[i].z());
		pcl::PointXYZ p2((float)corners[j].x(), (float)corners[j].y(), (float)corners[j].z());
		std::string eid = id_prefix + "_e" + std::to_string(k);

		viewer->removeShape(eid);                 // 避免重复叠加
		viewer->addLine(p1, p2, r, g, b, eid);
		viewer->setShapeRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, eid);
		};

	const int edges[12][2] = {
		// z- face (sx,sy) rectangle, but we don't need faces explicitly—just 3-axis edges
		{0,4}, {1,5}, {2,6}, {3,7},  // x edges
		{0,2}, {1,3}, {4,6}, {5,7},  // y edges
		{0,1}, {2,3}, {4,5}, {6,7}   // z edges
	};

	for (int k = 0; k < 12; ++k)
		addEdge(edges[k][0], edges[k][1], k);
}

//These methods are not used
#define MCMF 0
#define OA 1
#define UV 2
#define RSE 3
#define APORA 4
#define Kr 5
#define NBVNET 6
#define PCNBV 9
#define GMC 10
//Used in this paper
#define SCVP 7		// For helping history model covering
#define PriorTemporalRandom 8	// Passive Map + Temporal Prior + Random Selection
#define PriorTemporalCovering 11 // Ours from Passive Map + Temporal Prior and Change to 7 for set covering instead of random selection
#define PriorPassiveRandom 12 // Random with Passive Map (frontier)
#define PriorBBXRandom 13 // Random with BBX only
#define SamplingNBV 14 //Sampling Based NBV


class Share_Data
{
public:
	//可变输入参数
	string pcd_file_path;
	string yaml_file_path;
	string name_of_pcd;
	string nbv_net_path;
	string pcnbv_path;
	string sc_net_path;

	int num_of_views;					//一次采样视点个数
	double cost_weight;
	rs2_intrinsics color_intrinsics;
	double depth_scale;

	//运行参数
	int process_cnt;					//过程编号
	atomic<double> pre_clock;			//系统时钟
	atomic<bool> over;					//过程是否结束
	bool show;
	bool is_save;
	int num_of_max_iteration;
	int max_num_of_thread = 1000000;	//最大线程数
	int iterations;

	//点云数据
	atomic<int> vaild_clouds;
	vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds;							//点云组
	vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds_notable;							//点云组
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pcd;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ground_truth;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final;
	bool move_wait;
	map<string, double> mp_scale;

	//八叉地图
	octomap::ColorOcTree* octo_model;
	octomap::ColorOcTree* cloud_model;
	octomap::ColorOcTree* ground_truth_model;
	octomap::ColorOcTree* GT_sample;
	double octomap_resolution;
	double ground_truth_resolution;
	double map_size;
	double p_unknown_upper_bound; //! Upper bound for voxels to still be considered uncertain. Default: 0.97.
	double p_unknown_lower_bound; //! Lower bound for voxels to still be considered uncertain. Default: 0.12.

	double voxel_resolution_factor; //体素分辨率用于information gain
	
	//工作空间与视点空间
	atomic<bool> now_view_space_processed;
	atomic<bool> now_views_infromation_processed;
	atomic<bool> move_on;

	Eigen::Matrix4d now_camera_pose_world;
	Eigen::Vector3d object_center_world;		//物体中心
	double predicted_size;						//物体BBX半径

	int method_of_IG;
	bool MA_SCVP_on;
	bool Combined_on;
	int num_of_nbvs_combined;
	pcl::visualization::PCLVisualizer::Ptr viewer;

	double stop_thresh_map;
	double stop_thresh_view;

	double skip_coefficient;

	double sum_local_information;
	double sum_global_information;

	double sum_robot_cost;
	double camera_to_object_dis;
	double move_weight;
	bool robot_cost_negtive;
	bool move_cost_on;

	int num_of_max_flow_node;
	double interesting_threshold;

	double skip_threshold;
	double visble_rate;
	double move_rate;

	int init_voxels;     //点云voxel个数
	int voxels_in_BBX;   //地图voxel个数
	double init_entropy; //地图信息熵
	double movement_cost = 0; //总运动代价

	bool evaluate_one_shot; //one-shot pipeline是否需要更新最后的OctoMap用于评估，0表示不更新，直接出点云

	string pre_path;
	string gt_path;
	string save_path;

	int rotate_state = 0; //旋转的编码
	int first_view_id = 0; //初始视点编号
	vector<int> view_label_id;
	double min_z_table;

	int GT_points_number = -1; //预先获取的所有可见体素的个数
	int cloud_points_number; //点云GT中的点数
	int cloud_points_number_file;

	vector<int> f_voxels; //frontier voxels的数量

	int f_stop_iter = -1; //根据f_voxels停止的迭代轮次
	double f_stop_threshold = -1; //根据f_voxels停止的阈值

	int f_stop_iter_lenient = -1; //根据f_voxels停止的迭代轮次
	double f_stop_threshold_lenient = -1; //根据f_voxels停止的阈值

	bool use_saved_cloud = false; //是否使用预先保存的点云

	int mascvp_nbv_needed_views = -1; //MA_SCVP+1NBV需要的视点数

	bool has_table; //是否有桌面，无桌面的情况对应整个球面，注意务必保证前32个视点和半球32个相同，否则预训练的网络无法对应
	int length_of_viewstate;

	bool use_history_model_for_covering; //是否使用历史模型进行覆盖
	string nricp_path; //nricp路径
	double nricp_resolution_factor; //nricp分辨率因子，>1表示使用更低分辨率的图节点尺寸
	double history_resolution_factor; //历史模型分辨率因子是GT的几倍，越大分辨率越低
	bool add_current_inflation; //是否添加当前inflation的体素到历史模型中
	double inflation_resolution_factor; //inflation分辨率因子
	double near_current_inflation_distance_factor; //当前inflation的体素距离小于
	double far_history_inflation_distance_factor; //历史inflation的体素距离大于
	double history_confidence_count; //覆盖置信度阈值，0表示全部需要覆盖，1表示至少访问过两次的体素需要覆盖，2表示至少访问过三次的体素需要覆盖

	//for greenhouse
	string environment_path;
	string bbx_yaml_path;
	string passive_init_views_path;
	string dynamic_candidate_views_path;
	string room_str;
	OBB plant_obb;
	OBB view_obb;
	vector<Eigen::Vector3d> passive_init_views; //被动初始化的视点位置，按照顺序与passive_init_look_ats对应
	vector<Eigen::Vector3d> passive_init_look_ats; //被动初始化的look at target，按照顺序与passive_init_views对应
	vector<Eigen::Vector3d> dynamic_candidate_views; //动态候选视点，需要后续指定look at target
	vector<Eigen::Vector3d> dynamic_candidate_look_ats; //动态候选视点的look at target，需要根据deformation计算
	int gap_between_series; //连续间隔gap_between_series轮次
	std::map<string, string> previous_reconstruction_mapping; //map from previous reconstruction name to current reconstruction name, used for loading history model for covering
	double ray_max_dis; //ray tracing distance (depth cut)
	int num_targets; //look at target数量
	int num_rounds; //迭代轮次 for asigning look at target to candidate view
	int random_seed = 42; //随机数种子，默认为42
	int debug = 0;
	string look_at_group_str = "Undefined";
	double passive_map_cost = 0.0; //被动地图的运动代价
	string passive_map_cost_path; //被动地图运动代价的文件路径
	int last_passive_view_id = 14;
	double view_space_octomap_resolution_factor; //for oracle visble surface covergae cumputation, the octomap resolution factor for building the view space octomap
	int oracle_knn; //for oracle visble surface covergae cumputation
	set<int> set_of_num_best_views_for_compare = {15, 50, 65, 100, 115, 200, 215, 300, 315, 400, 415, 500, 515}; //用于评估时与其他方法比较的最佳视点数量列表
	double frontier_resolution_factor; //体素分辨率用于找frontier lookat

	//data for SamplingNBV
	int current_block = -1; //当前block编号
	int current_budget = -1; //当前block剩余可选视点数量
	vector<vector<int>> ordered_blocks; //排序后的block

	Share_Data(string _config_file_path, string test_name = "", int test_rotate = -1, int test_view = -1, int test_method = -1, int move_test_on = -1, int combined_test_on = -1, int test_random_seed = -1)
	{
		process_cnt = -1;
		yaml_file_path = _config_file_path;
		//读取yaml文件
		cv::FileStorage fs;
		fs.open(yaml_file_path, cv::FileStorage::READ);
		fs["debug"] >> debug;
		fs["pre_path"] >> pre_path;
		fs["gt_path"] >> gt_path;
		fs["model_path"] >> pcd_file_path;
		fs["environment_path"] >> environment_path;
		fs["gap_between_series"] >> gap_between_series;
		fs["ray_max_dis"] >> ray_max_dis;
		fs["num_targets"] >> num_targets;
		fs["num_rounds"] >> num_rounds;
		fs["name_of_pcd"] >> name_of_pcd;
		fs["max_num_of_thread"] >> max_num_of_thread;
		fs["method_of_IG"] >> method_of_IG;
		fs["has_table"] >> has_table;
		fs["length_of_viewstate"] >> length_of_viewstate;
		fs["use_history_model_for_covering"] >> use_history_model_for_covering;
		fs["nricp_resolution_factor"] >> nricp_resolution_factor;
		fs["history_resolution_factor"] >> history_resolution_factor;
		fs["add_current_inflation"] >> add_current_inflation;
		fs["inflation_resolution_factor"] >> inflation_resolution_factor;
		fs["near_current_inflation_distance_factor"] >> near_current_inflation_distance_factor;
		fs["far_history_inflation_distance_factor"] >> far_history_inflation_distance_factor;
		fs["view_space_octomap_resolution_factor"] >> view_space_octomap_resolution_factor;
		fs["frontier_resolution_factor"] >> frontier_resolution_factor;
		fs["oracle_knn"] >> oracle_knn;
		fs["history_confidence_count"] >> history_confidence_count;
		fs["MA_SCVP_on"] >> MA_SCVP_on;
		fs["Combined_on"] >> Combined_on;
		fs["num_of_nbvs_combined"] >> num_of_nbvs_combined;
		fs["octomap_resolution"] >> octomap_resolution;
		fs["ground_truth_resolution"] >> ground_truth_resolution;
		fs["voxel_resolution_factor"] >> voxel_resolution_factor;
		fs["num_of_max_iteration"] >> num_of_max_iteration;
		fs["f_stop_threshold"] >> f_stop_threshold;
		fs["use_saved_cloud"] >> use_saved_cloud;
		fs["show"] >> show;
		fs["is_save"] >> is_save;
		fs["evaluate_one_shot"] >> evaluate_one_shot;
		fs["move_wait"] >> move_wait;
		fs["nbv_net_path"] >> nbv_net_path;
		fs["pcnbv_path"] >> pcnbv_path;
		fs["sc_net_path"] >> sc_net_path;
		fs["nricp_path"] >> nricp_path;
		fs["p_unknown_upper_bound"] >> p_unknown_upper_bound;
		fs["p_unknown_lower_bound"] >> p_unknown_lower_bound;
		fs["num_of_views"] >> num_of_views;
		fs["move_cost_on"] >> move_cost_on;
		fs["move_weight"] >> move_weight;
		fs["cost_weight"] >> cost_weight;
		fs["robot_cost_negtive"] >> robot_cost_negtive;
		fs["skip_coefficient"] >> skip_coefficient;
		fs["num_of_max_flow_node"] >> num_of_max_flow_node;
		fs["interesting_threshold"] >> interesting_threshold;
		fs["skip_threshold"] >> skip_threshold;
		fs["visble_rate"] >> visble_rate;
		fs["move_rate"] >> move_rate;
		fs["color_width"] >> color_intrinsics.width;
		fs["color_height"] >> color_intrinsics.height;
		fs["color_fx"] >> color_intrinsics.fx;
		fs["color_fy"] >> color_intrinsics.fy;
		fs["color_ppx"] >> color_intrinsics.ppx;
		fs["color_ppy"] >> color_intrinsics.ppy;
		fs["color_model"] >> color_intrinsics.model;
		fs["color_k1"] >> color_intrinsics.coeffs[0];
		fs["color_k2"] >> color_intrinsics.coeffs[1];
		fs["color_k3"] >> color_intrinsics.coeffs[2];
		fs["color_p1"] >> color_intrinsics.coeffs[3];
		fs["color_p2"] >> color_intrinsics.coeffs[4];
		fs["depth_scale"] >> depth_scale;
		fs.release();
		//自动化参数
		if (test_name != "") name_of_pcd = test_name;
		if (test_rotate != -1) rotate_state = test_rotate;
		if (test_view != -1) first_view_id = test_view;
		if (test_method != -1) method_of_IG = test_method;
		if (move_test_on != -1) move_cost_on = move_test_on;
		if (combined_test_on != -1) Combined_on = combined_test_on;
		if (test_random_seed != -1) random_seed = test_random_seed;
		if (method_of_IG == 7 || Combined_on == true) {
			if (use_history_model_for_covering) {
				num_of_max_iteration = num_of_views; //历史覆盖系列不限制最大值
			}
			else {
				num_of_max_iteration = 32; //SCVP系列不限制最大值（为viewspace上限）
			}
		}
		if (Combined_on == false) num_of_nbvs_combined = 0; //非综合系统不设置初始NBV个数偏移(用于指示MASCVP)
		/* //最大重建数量为默认20
		if (method_of_IG != 7) {
			string path_scvp = pre_path + name_of_pcd + "_r" + to_string(rotate_state) + "_v" + to_string(first_view_id) + "_m7";
			ifstream fin_all_needed_views;
			fin_all_needed_views.open(path_scvp + "/all_needed_views.txt");
			if (fin_all_needed_views.is_open()) {
				int all_needed_views;
				fin_all_needed_views >> all_needed_views;
				num_of_max_iteration = all_needed_views;
				cout << "max iteration set to " << num_of_max_iteration << endl;
			}
			else {
				cout << "max iteration is default " << num_of_max_iteration << endl;
			}
		}
		*/
		//get from environment yaml
		room_str = name_of_pcd.substr(0, name_of_pcd.find('_'));
		cout << "room_str: " << room_str << endl;
		bbx_yaml_path = environment_path + "/" + room_str + "_bbx_config.yaml";
		plant_obb = load_obb_opencv(bbx_yaml_path, "plant_box");
		view_obb = load_obb_opencv(bbx_yaml_path, "view_box");
		cout << "plant_obb center: " << plant_obb.c.transpose() << endl;
		cout << "view_obb center: " << view_obb.c.transpose() << endl;
		passive_init_views_path = environment_path + "/" + room_str + "_initial_views.txt";
		ifstream fin_passive_init_views(passive_init_views_path);
		if (fin_passive_init_views.is_open()) {
			while (!fin_passive_init_views.eof()) {
				double x, y, z;
				fin_passive_init_views >> x >> y >> z;
				if (fin_passive_init_views.eof()) break;
				passive_init_views.push_back(Eigen::Vector3d(x, y, z));
				fin_passive_init_views >> x >> y >> z;
				passive_init_look_ats.push_back(Eigen::Vector3d(x, y, z));
			}
			fin_passive_init_views.close();
		}
		else {	
			cout << "No passive initial views file found." << endl;
		}
		cout << "passive_init_views size: " << passive_init_views.size() << endl;
		passive_map_cost_path = environment_path + "/" + room_str + "_passive_cost.txt";
		ifstream fin_passive_map_cost(passive_map_cost_path);
		if (fin_passive_map_cost.is_open()) {
			fin_passive_map_cost >> passive_map_cost;
			fin_passive_map_cost.close();
		}
		else {
			cout << "No passive map cost file found. Set to default 0.0" << endl;
			passive_map_cost = 0.0;
		}
		cout << "passive_map_cost: " << passive_map_cost << endl;
		dynamic_candidate_views_path = environment_path + "/" + room_str + "_candidate_views.txt";
		ifstream fin_dynamic_candidate_views(dynamic_candidate_views_path);
		if (fin_dynamic_candidate_views.is_open()) {
			while (!fin_dynamic_candidate_views.eof()) {
				double x, y, z;
				fin_dynamic_candidate_views >> x >> y >> z;
				if (fin_dynamic_candidate_views.eof()) break;
				dynamic_candidate_views.push_back(Eigen::Vector3d(x, y, z));
				if (!view_obb.contains(dynamic_candidate_views.back())) {
					cout << "dynamic_candidate_views out of BBX. check." <<endl;
				}
			}
			fin_dynamic_candidate_views.close();
		}
		else {
			cout << "No dynamic candidate views file found." << endl;
		}
		cout << "dynamic_candidate_views size: " << dynamic_candidate_views.size() << endl;
		assert(passive_init_views.size() + dynamic_candidate_views.size() == num_of_views);
		// 这部分要根据不同的gap的deformation和算法计算look at target
		if (method_of_IG == 8 || method_of_IG == 11 || method_of_IG == 7) {
			look_at_group_str = "PriorTemporal";
		}
		else if (method_of_IG == 12) {
			look_at_group_str = "PriorPassive";
		}
		else if (method_of_IG == 13) {
			look_at_group_str = "PriorBBX";
		}
		cout << "look_at_group_str: " << look_at_group_str << endl;
		string dynamic_look_at_path = environment_path + "/gap_" + to_string(gap_between_series) + "/" + name_of_pcd + "_candidate_look_ats_" + look_at_group_str + ".txt";
		ifstream fin_dynamic_look_ats(dynamic_look_at_path);
		if (fin_dynamic_look_ats.is_open()) {
			while (!fin_dynamic_look_ats.eof()) {
				double x, y, z;
				fin_dynamic_look_ats >> x >> y >> z;
				if (fin_dynamic_look_ats.eof()) break;
				dynamic_candidate_look_ats.push_back(Eigen::Vector3d(x, y, z));
			}
			fin_dynamic_look_ats.close();
		}
		else {
			cout << "No dynamic candidate look ats file found. continue without this is fine but no evaluation." << endl;
		}
		cout << "dynamic_candidate_look_ats size: " << dynamic_candidate_look_ats.size() << endl;
		//读取转换后模型的pcd文件
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_pcd = temp_pcd;
		if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_file_path + name_of_pcd + ".pcd", *cloud_pcd) == -1) {
			cout << "Can not read 3d model file. Check." << endl;
		}
		cout << "Loaded model point cloud has " << cloud_pcd->points.size() << " points." << endl;
		//读取time series映射
		string time_series_mapping_path = pcd_file_path + "previous_reconstruction_mapping_gap" + to_string(gap_between_series) + ".txt";
		ifstream fin_time_series_mapping(time_series_mapping_path);
		if (fin_time_series_mapping.is_open()) {
			while (!fin_time_series_mapping.eof()) {
				string current_name, previous_name;
				fin_time_series_mapping >> current_name >> previous_name;
				if (fin_time_series_mapping.eof()) break;
				previous_reconstruction_mapping[current_name] = previous_name;
			}
			fin_time_series_mapping.close();
		}
		else {
			cout << "No time series mapping file found." << endl;
		}
		gt_path += "/gap_" + to_string(gap_between_series) + "/";
		cout << "previous_reconstruction_mapping size: " << previous_reconstruction_mapping.size() << endl;


		//旋转角度
		Eigen::Matrix3d rotation;
		rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
			Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
			Eigen::AngleAxisd(45 * rotate_state * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
		Eigen::Matrix4d T_pose(Eigen::Matrix4d::Identity(4, 4));
		T_pose(0, 0) = rotation(0, 0); T_pose(0, 1) = rotation(0, 1); T_pose(0, 2) = rotation(0, 2); T_pose(0, 3) = 0;
		T_pose(1, 0) = rotation(1, 0); T_pose(1, 1) = rotation(1, 1); T_pose(1, 2) = rotation(1, 2); T_pose(1, 3) = 0;
		T_pose(2, 0) = rotation(2, 0); T_pose(2, 1) = rotation(2, 1); T_pose(2, 2) = rotation(2, 2); T_pose(2, 3) = 0;
		T_pose(3, 0) = 0;			   T_pose(3, 1) = 0;			  T_pose(3, 2) = 0;			     T_pose(3, 3) = 1;
		pcl::transformPointCloud(*cloud_pcd, *cloud_pcd, T_pose);
		//读GT
		ifstream fin_GT_points_number;
		fin_GT_points_number.open(gt_path + "/GT_OracleVisiblePoisson/" + name_of_pcd + ".txt");
		if (fin_GT_points_number.is_open()) {
			fin_GT_points_number >> GT_points_number >> cloud_points_number_file;
			cout << "GT_points_number is " << GT_points_number << endl;
		}
		else {
			cout << "no GT_points_number, run mode GetOracleVisible first. This process will continue without GT_points_number." << endl;
		}
		fin_GT_points_number.close();

		//octo_model = new octomap::ColorOcTree(octomap_resolution);
		//octo_model->setProbHit(0.95);	//设置传感器命中率,初始0.7
		//octo_model->setProbMiss(0.05);	//设置传感器失误率，初始0.4
		//octo_model->setClampingThresMax(1.0);	//设置地图节点最大值，初始0.971
		//octo_model->setClampingThresMin(0.0);	//设置地图节点最小值，初始0.1192
		//octo_model->setOccupancyThres(0.65);	//设置节点占用阈值，初始0.5
		ground_truth_model = new octomap::ColorOcTree(ground_truth_resolution);
		//ground_truth_model->setProbHit(0.95);	//设置传感器命中率,初始0.7
		//ground_truth_model->setProbMiss(0.05);	//设置传感器失误率，初始0.4
		//ground_truth_model->setClampingThresMax(1.0);	//设置地图节点最大值，初始0.971
		//ground_truth_model->setClampingThresMin(0.0);	//设置地图节点最小值，初始0.1192
		//GT_sample = new octomap::ColorOcTree(octomap_resolution);
		//GT_sample->setProbHit(0.95);	//设置传感器命中率,初始0.7
		//GT_sample->setProbMiss(0.05);	//设置传感器失误率，初始0.4
		//GT_sample->setClampingThresMax(1.0);	//设置地图节点最大值，初始0.971
		//GT_sample->setClampingThresMin(0.0);	//设置地图节点最小值，初始0.1192
		cloud_model = new octomap::ColorOcTree(ground_truth_resolution);
		//cloud_model->setProbHit(0.95);	//设置传感器命中率,初始0.7
		//cloud_model->setProbMiss(0.05);	//设置传感器失误率，初始0.4
		//cloud_model->setClampingThresMax(1.0);	//设置地图节点最大值，初始0.971
		//cloud_model->setClampingThresMin(0.0);	//设置地图节点最小值，初始0.1192
		//cloud_model->setOccupancyThres(0.65);	//设置节点占用阈值，初始0.5*/
		if (num_of_max_flow_node == -1) num_of_max_flow_node = num_of_views;
		now_camera_pose_world = Eigen::Matrix4d::Identity(4, 4);
		over = false;
		pre_clock = clock();
		vaild_clouds = 0;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_final = temp;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_gt(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_ground_truth = temp_gt;
		access_directory(pre_path);
		//save_path = "../" + name_of_pcd + '_' + to_string(method_of_IG);
		save_path = pre_path + name_of_pcd + "_r" + to_string(rotate_state) + "_v" + to_string(first_view_id) + "_m" + to_string(method_of_IG);
		if (method_of_IG==7 && MA_SCVP_on == false) save_path += "_nostate";
		if (move_cost_on == true) save_path += '_' + to_string(move_weight);
		if (method_of_IG==10 && move_rate<= 0.99) save_path += '_' + to_string(move_rate);
		if (Combined_on == true) {
			save_path += "_combined_" + to_string(num_of_nbvs_combined); //default for mascvp+1nbv
			if (use_history_model_for_covering == true) {
				save_path += "_history"; //if use history model instead of mascvp
				if (add_current_inflation == true) {
					save_path += "_inflation"; //if add current inflation to history model
				}
				save_path += "_sc" + to_string(history_resolution_factor); //report history_resolution_factor as sc covering resolution
			}
		}
		//随机方法和基于BBX的随机方法需要添加随机数种子以区分不同的运行结果
		if (method_of_IG == 8 || method_of_IG == 12 || method_of_IG == 13) {
			save_path += "_seed" + to_string(random_seed);
		}
		cout << "pcd and yaml files readed." << endl;
		cout << "save_path is: " << save_path << endl;

		f_stop_threshold_lenient = f_stop_threshold * 5;

		if (method_of_IG == 8) { // 随机方法需要对比的数量
			ifstream fin_mascvp_nbv_needed_views(gt_path + "/Compare/" + name_of_pcd + "_r" + to_string(rotate_state) + "_v" + to_string(first_view_id) + "_m9_combined_1/all_needed_views.txt");
			if (!fin_mascvp_nbv_needed_views) {
				cout << "no all_needed_views from mascvp+1nbv. run mascvp+1nbv first." << endl;
			}
			else {
				fin_mascvp_nbv_needed_views >> mascvp_nbv_needed_views;
				cout << "mascvp_nbv_needed_views num is " << mascvp_nbv_needed_views << endl;
			}
		}

	}

	~Share_Data() {
		if (octo_model != NULL) delete octo_model;
		if (ground_truth_model != NULL) delete ground_truth_model;
		if (cloud_model != NULL) delete cloud_model;
		if (GT_sample != NULL) delete GT_sample;
		cloud_pcd->points.clear();
		cloud_pcd->points.shrink_to_fit();
		cloud_final->points.clear();
		cloud_final->points.shrink_to_fit();
		cloud_ground_truth->points.clear();
		cloud_ground_truth->points.shrink_to_fit();
		for (int i = 0; i < clouds.size(); i++) {
			clouds[i]->points.clear();
			clouds[i]->points.shrink_to_fit();
		}
		clouds.clear();
		clouds.shrink_to_fit();
		for (int i = 0; i < clouds_notable.size(); i++) {
			clouds_notable[i]->points.clear();
			clouds_notable[i]->points.shrink_to_fit();
		}
		clouds_notable.clear();
		clouds_notable.shrink_to_fit();
		if (show) viewer->~PCLVisualizer();
		f_voxels.clear();
		f_voxels.shrink_to_fit();
	}

	Eigen::Matrix4d get_toward_pose(int toward_state)
	{
		Eigen::Matrix4d pose(Eigen::Matrix4d::Identity(4, 4));
		switch (toward_state) {
		case 0://z<->z
			return pose;
		case 1://z<->-z
			pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
			pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
			pose(2, 0) = 0; pose(2, 1) = 0; pose(2, 2) = -1; pose(2, 3) = 0;
			pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
			return pose;
		case 2://z<->x
			pose(0, 0) = 0; pose(0, 1) = 0; pose(0, 2) = 1; pose(0, 3) = 0;
			pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
			pose(2, 0) = 1; pose(2, 1) = 0; pose(2, 2) = 0; pose(2, 3) = 0;
			pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
			return pose;
		case 3://z<->-x
			pose(0, 0) = 0; pose(0, 1) = 0; pose(0, 2) = 1; pose(0, 3) = 0;
			pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
			pose(2, 0) = -1; pose(2, 1) = 0; pose(2, 2) = 0; pose(2, 3) = 0;
			pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
			return pose;
		case 4://z<->y
			pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
			pose(1, 0) = 0; pose(1, 1) = 0; pose(1, 2) = 1; pose(1, 3) = 0;
			pose(2, 0) = 0; pose(2, 1) = 1; pose(2, 2) = 0; pose(2, 3) = 0;
			pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
			return pose;
		case 5://z<->-y
			pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
			pose(1, 0) = 0; pose(1, 1) = 0; pose(1, 2) = 1; pose(1, 3) = 0;
			pose(2, 0) = 0; pose(2, 1) = -1; pose(2, 2) = 0; pose(2, 3) = 0;
			pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
			return pose;
		}
		return pose;
	}

	double out_clock()
	{   //返回用时，并更新时钟
		double now_clock = clock();
		double time_len = now_clock - pre_clock;
		pre_clock = now_clock;
		return time_len;
	}

	void access_directory(string cd)
	{   //检测多级目录的文件夹是否存在，不存在就创建
		string temp;
		for (int i = 0; i < cd.length(); i++)
			if (cd[i] == '/') {
				if (access(temp.c_str(), 0) != 0) mkdir(temp.c_str());
				temp += cd[i];
			}
			else temp += cd[i];
		if (access(temp.c_str(), 0) != 0) mkdir(temp.c_str());
	}

	void save_posetrans_to_disk(Eigen::Matrix4d& T, string cd, string name, int frames_cnt)
	{   //存放旋转矩阵数据至磁盘
		std::stringstream pose_stream, path_stream;
		std::string pose_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		pose_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << frames_cnt << ".txt";
		pose_stream >> pose_file;
		ofstream fout(pose_file);
		fout << T;
	}

	void save_octomap_log_to_disk(int voxels, double entropy, string cd, string name,int iterations)
	{
		std::stringstream log_stream, path_stream;
		std::string log_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		log_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << iterations << ".txt";
		log_stream >> log_file;
		ofstream fout(log_file);
		fout << voxels << " " << entropy << endl;
	}

	void save_cloud_to_disk(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string cd, string name)
	{   //存放点云数据至磁盘，速度很慢，少用
		std::stringstream cloud_stream, path_stream;
		std::string cloud_file, path;
		path_stream << save_path << cd;
		path_stream >> path;
		access_directory(path);
		cloud_stream << save_path << cd << "/" << name << ".pcd";
		cloud_stream >> cloud_file;
		//pcl::io::savePCDFile<pcl::PointXYZRGB>(cloud_file, *cloud);
		pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(cloud_file, *cloud);
	}

	void save_cloud_to_disk(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string cd, string name, int frames_cnt)
	{   //存放点云数据至磁盘，速度很慢，少用
		std::stringstream cloud_stream, path_stream;
		std::string cloud_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		cloud_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << frames_cnt << ".pcd";
		cloud_stream >> cloud_file;
		//pcl::io::savePCDFile<pcl::PointXYZRGB>(cloud_file, *cloud);
		pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(cloud_file, *cloud);
	}

	void save_octomap_to_disk(octomap::ColorOcTree* octo_model, string cd, string name)
	{   //存放点云数据至磁盘，速度很慢，少用
		std::stringstream octomap_stream, path_stream;
		std::string octomap_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		octomap_stream << "../data" << "_" << process_cnt << cd << "/" << name << ".ot";
		octomap_stream >> octomap_file;
		octo_model->write(octomap_file);
	}

};

inline double pow2(double x) {
	return x * x;
}