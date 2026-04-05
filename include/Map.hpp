#pragma once

#include "MapPoint.hpp"
#include "KeyFrameDatabase.hpp"
#include "Config.hpp"
#include <unordered_map>

class Map {
public:
    Map(const Config& cfg) : cfg_(cfg) {}

    // KeyFrame operations (delegates to kf_db_)
    const KeyFrame& createKeyFrame(const Frame& frame);
    const KeyFrame& getKeyFrame(int kf_id) const;
    KeyFrame& getKeyFrameMutable(int kf_id) { return kf_db_.getKeyFrameMutable(kf_id); }
    bool hasKeyFrame(int kf_id) const;
    size_t numKeyFrames() const { return kf_db_.size(); }

    // MapPoint operations
    // Associate existing map points with a keyframe (from tracking correspondences)
    void associateMapPoints(const KeyFrame& kf, const std::vector<std::pair<int, int>>& correspondences);
    // Create new map points for unmatched keypoints
    void addNewMapPoints(const KeyFrame& kf, const std::vector<std::pair<int, Eigen::Vector3d>>& keypoints_3d_world);
    // Update covisibility graph for a keyframe
    void updateCovisibility(const KeyFrame& kf);
    MapPoint& addMapPoint(const Eigen::Vector3d& position);
    void removeMapPoint(int id);
    MapPoint& getMapPoint(int id);
    const MapPoint& getMapPoint(int id) const;
    bool hasMapPoint(int id) const;
    size_t numMapPoints() const { return map_points_.size(); }
    int cullMapPoints(int current_kf_id);

    const std::unordered_map<int, MapPoint>& getAllMapPoints() const { return map_points_; }
    const std::unordered_map<int, KeyFrame>& getAllKeyFrames() const { return kf_db_.getAllKeyFrames(); }

private:
    Config cfg_;
    KeyFrameDatabase kf_db_;
    std::unordered_map<int, MapPoint> map_points_;
    int next_id_ = 0;
};
