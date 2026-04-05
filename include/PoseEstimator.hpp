#pragma once

#include "Types.hpp"
#include <sophus/se3.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <optional>

struct PnPResult {
    std::optional<Sophus::SE3d> pose;
    std::vector<int> inlier_indices;
};

PnPResult estimatePosePnP(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    const Intrinsic& intrinsic);

Sophus::SE3d motionBA(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    const Intrinsic& intrinsic,
    const Sophus::SE3d& T_init,
    int num_iterations,
    std::vector<bool>* inlier_mask = nullptr);
