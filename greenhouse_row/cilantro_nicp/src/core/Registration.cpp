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
#include "Registration.hpp"

#include <cilantro/registration/icp_common_instances.hpp>
#include <cilantro/utilities/point_cloud.hpp>

cilantro::PointCloud3f deform_cloud(const cilantro::PointCloud3f &src,
                                    const cilantro::PointCloud3f &dst) {
    // Initialization
    const float control_res = 0.15f;
    const float src_to_control_sigma = 0.5f * control_res;
    const float regularization_sigma = 3.0f * control_res;
    const float max_correspondence_dist_sq = 0.5f * 0.5f;

    // Get a sparse set of control nodes by downsampling
    const cilantro::VectorSet<float, 3> control_points =
        cilantro::PointsGridDownsampler3f(src.points, control_res)
            .getDownsampledPoints();
    const cilantro::KDTree<float, 3> control_tree(control_points);

    // Find which control nodes affect each point in src
    const cilantro::NeighborhoodSet<float> src_to_control_nn =
        control_tree.search(src.points,
                            cilantro::KNNNeighborhoodSpecification<>(4));

    // Get regularization neighborhoods for control nodes
    const cilantro::NeighborhoodSet<float> regularization_nn =
        control_tree.search(control_points,
                            cilantro::KNNNeighborhoodSpecification<>(8));

    // Initialize ICP
    cilantro::SimpleCombinedMetricSparseRigidWarpFieldICP3f icp(
        dst.points, dst.normals, src.points, src_to_control_nn,
        control_points.cols(), regularization_nn);

    // Set parameters
    icp.correspondenceSearchEngine().setMaxDistance(max_correspondence_dist_sq);
    icp.controlWeightEvaluator().setSigma(src_to_control_sigma);
    icp.regularizationWeightEvaluator().setSigma(regularization_sigma);
    icp.setMaxNumberOfIterations(15).setConvergenceTolerance(2.5e-3f);
    icp.setMaxNumberOfGaussNewtonIterations(10)
        .setGaussNewtonConvergenceTolerance(5e-4f);
    icp.setMaxNumberOfConjugateGradientIterations(500)
        .setConjugateGradientConvergenceTolerance(1e-5f);
    icp.setPointToPointMetricWeight(0.0f)
        .setPointToPlaneMetricWeight(1.0f)
        .setStiffnessRegularizationWeight(2000.0f);
    icp.setHuberLossBoundary(1e-2f);

    // Perform deformation
    const auto tf_est = icp.estimate().getDenseWarpField();
    return src.transformed(tf_est);
}
