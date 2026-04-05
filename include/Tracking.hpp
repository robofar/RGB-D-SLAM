#pragma once

#include "Config.hpp"
#include "Types.hpp"
#include "FeatureManager.hpp"
#include "Map.hpp"
#include <opencv2/core.hpp>

class Tracking {
public:
    Tracking(const Config& cfg, const Intrinsic& intrinsic,
             FeatureManager& feature_manager, Map& map);

    bool processFrame(Frame& frame);
    int getLastKeyFrameId() const { return last_kf_id_; }

private:
    // Returns vector of (map_point_id, keypoint_idx) pairs
    // use_local_map=false: last keyframe's map points only
    // use_local_map=true: K1 (share points with last KF) + K2 (covisible neighbors of K1)
    std::vector<std::pair<int, int>> findCorrespondences(
        const Frame& frame,
        const Sophus::SE3d& pose,
        double search_radius,
        bool use_local_map = false
    );

    Config cfg_;
    Intrinsic intrinsic_;
    FeatureManager& feature_manager_;
    Map& map_;

    bool initialized_ = false;
    int last_kf_id_ = -1;
    int frames_since_last_kf_ = 0;
    Sophus::SE3d prev_T_;
    Sophus::SE3d delta_T_;
};
