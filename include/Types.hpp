#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

struct Intrinsic {
    double fx, fy, cx, cy;
    int width, height;
    double depth_scale;
};

struct Association {
    std::string rgb_image_path;
    std::string depth_image_path;
    Sophus::SE3d gt_pose;
};

struct Frame {
    int frame_id;
    cv::Mat rgb;
    cv::Mat depth;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Sophus::SE3d gt_pose;
    Sophus::SE3d estimated_pose;
};
