#include "Utils.hpp"
#include <cmath>
#include <iostream>

Eigen::Vector2d project(const Eigen::Vector3d& p_world, const Sophus::SE3d& T_cw, const Intrinsic& intrinsic) {
    Eigen::Vector3d p_cam = T_cw * p_world;
    double u = intrinsic.fx * (p_cam.x() / p_cam.z()) + intrinsic.cx;
    double v = intrinsic.fy * (p_cam.y() / p_cam.z()) + intrinsic.cy;
    return Eigen::Vector2d(u, v);
}

Eigen::Vector3d backproject(const cv::KeyPoint& kp, float depth, const Intrinsic& intrinsic) {
    double x = (kp.pt.x - intrinsic.cx) * depth / intrinsic.fx;
    double y = (kp.pt.y - intrinsic.cy) * depth / intrinsic.fy;
    return Eigen::Vector3d(x, y, depth);
}

std::vector<std::pair<int, Eigen::Vector3d>> backprojectKeypoints(const KeyFrame& kf, const cv::Mat& depth, const Intrinsic& intrinsic) {

    std::vector<std::pair<int, Eigen::Vector3d>> observations;
    int num_invalid = 0;

    for (int i = 0; i < static_cast<int>(kf.keypoints.size()); ++i) {
        const auto& kp = kf.keypoints[i];
        int u = static_cast<int>(std::round(kp.pt.x));
        int v = static_cast<int>(std::round(kp.pt.y));

        if (u < 0 || u >= depth.cols || v < 0 || v >= depth.rows) {
            num_invalid++;
            continue;
        }

        float raw_depth = depth.at<uint16_t>(v, u);
        if (raw_depth <= 0) {
            num_invalid++;
            continue;
        }

        float d = raw_depth / intrinsic.depth_scale;
        Eigen::Vector3d p_world = kf.estimated_pose * backproject(kp, d, intrinsic);
        observations.emplace_back(i, p_world);
    }

    return observations;
}
