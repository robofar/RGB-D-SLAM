#pragma once

#include "Types.hpp"
#include <unordered_map>

struct KeyFrame {
    int id;
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    Sophus::SE3d gt_pose;
    Sophus::SE3d estimated_pose;
    std::unordered_map<int, int> keypoint_to_map_point;  // keypoint_idx -> map_point_id
    std::unordered_map<int, int> covisible_keyframes;   // neighbor_kf_id -> weight (shared map points)
};

class KeyFrameDatabase {
public:
    const KeyFrame& createKeyFrame(const Frame& frame);
    void removeKeyFrame(int kf_id);
    const KeyFrame& getKeyFrame(int kf_id) const;
    KeyFrame& getKeyFrameMutable(int kf_id);
    bool hasKeyFrame(int kf_id) const;
    size_t size() const { return keyframes_.size(); }
    const std::unordered_map<int, KeyFrame>& getAllKeyFrames() const { return keyframes_; }

private:
    void addKeyFrame(const KeyFrame& kf);

    std::unordered_map<int, KeyFrame> keyframes_;
    int next_kf_id_ = 0;
};
