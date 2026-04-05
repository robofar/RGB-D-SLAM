#pragma once

#include "Types.hpp"
#include "KeyFrameDatabase.hpp"
#include <vector>
#include <utility>

Eigen::Vector3d backproject(const cv::KeyPoint& kp, float depth, const Intrinsic& intrinsic);
Eigen::Vector2d project(const Eigen::Vector3d& p_world, const Sophus::SE3d& T_cw, const Intrinsic& intrinsic);

std::vector<std::pair<int, Eigen::Vector3d>> backprojectKeypoints(const KeyFrame& kf, const cv::Mat& depth, const Intrinsic& intrinsic);
