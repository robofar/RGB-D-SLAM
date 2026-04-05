#pragma once

#include <Eigen/Dense>
#include <unordered_map>
#include "KeyFrameDatabase.hpp"

class MapPoint {
public:
    MapPoint(int id, const Eigen::Vector3d& position);

    // Getters
    int getId() const { return id_; }
    const Eigen::Vector3d& getPosition() const { return position_; }
    int getRepresentativeKfId() const { return representative_kf_id_; }
    int getRepresentativeDescriptorIdx() const { return representative_descriptor_idx_; }
    const Eigen::Vector3d& getViewingDirection() const { return viewing_direction_; }
    float getMinDistance() const { return min_distance_; }
    float getMaxDistance() const { return max_distance_; }
    const std::unordered_map<int, int>& getObservations() const { return observations_; }

    // Setters
    void setPosition(const Eigen::Vector3d& position) { position_ = position; }

    // Updaters
    void updateDescriptor(const KeyFrameDatabase& kf_db);
    void updateViewingDirection(const KeyFrameDatabase& kf_db);
    void updateDepthRange(const KeyFrameDatabase& kf_db, float scale_factor, int num_levels);

    // Observations
    void addObservation(int kf_id, int keypoint_idx);
    void removeObservation(int kf_id);
    int numObservations() const { return observations_.size(); }

    // Tracking statistics
    void incrementVisible() { ++visible_count_; }
    void incrementFound() { ++found_count_; }
    int getVisibleCount() const { return visible_count_; }
    int getFoundCount() const { return found_count_; }
    double getFoundRatio() const { return visible_count_ > 0 ? static_cast<double>(found_count_) / visible_count_ : 0.0; }

private:
    int id_;
    Eigen::Vector3d position_;

    int representative_kf_id_ = -1;
    int representative_descriptor_idx_ = -1;

    Eigen::Vector3d viewing_direction_ = Eigen::Vector3d::Zero();
    float min_distance_ = 0.0f;
    float max_distance_ = 0.0f;

    std::unordered_map<int, int> observations_;  // kf_id -> keypoint_idx

    int visible_count_ = 0;
    int found_count_ = 0;
};
