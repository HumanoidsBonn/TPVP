// MIT License
//
// Copyright (c) 2025 Luca Lobefaro
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "Utilities.hpp"

#include <open3d/Open3D.h>
#include <cilantro/utilities/point_cloud.hpp>
#include <string>
#include "Eigen/src/Core/Matrix.h"

namespace {
std::shared_ptr<open3d::geometry::PointCloud> CilantroToOpen3D(
    const cilantro::PointCloud3f &pc, const Eigen::Vector3d &color) {
    auto o3d_pc = std::make_shared<open3d::geometry::PointCloud>();
    const auto &points = pc.points;

    o3d_pc->points_.resize(points.cols());
    for (int i = 0; i < points.cols(); ++i) {
        o3d_pc->points_[i] = Eigen::Vector3d(static_cast<double>(points(0, i)),
                                             static_cast<double>(points(1, i)),
                                             static_cast<double>(points(2, i)));
    }

    o3d_pc->PaintUniformColor(color);
    return o3d_pc;
}
}  // namespace

cilantro::PointCloud3f load_pcd(const std::string &filename) {
    return cilantro::PointCloud3f(filename);
}

void save_pcd(const cilantro::PointCloud3f &cloud,
              const std::string &filename) {
    cloud.toPLYFile(filename);
}

void visualize_pcds(const cilantro::PointCloud3f &src,
                    const cilantro::PointCloud3f &dst) {
    const auto o3d_src = CilantroToOpen3D(src, Eigen::Vector3d(1.0, 0.0, 0.0));
    const auto o3d_dst = CilantroToOpen3D(dst, Eigen::Vector3d(0.0, 1.0, 0.0));
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> clouds = {
        o3d_src, o3d_dst};
    open3d::visualization::DrawGeometries(clouds);
}
