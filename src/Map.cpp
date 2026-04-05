#include "Map.hpp"
#include <stdexcept>
#include <iostream>

const KeyFrame& Map::createKeyFrame(const Frame& frame) {
    return kf_db_.createKeyFrame(frame);
}

const KeyFrame& Map::getKeyFrame(int kf_id) const {
    return kf_db_.getKeyFrame(kf_id);
}

bool Map::hasKeyFrame(int kf_id) const {
    return kf_db_.hasKeyFrame(kf_id);
}

void Map::associateMapPoints(const KeyFrame& kf, const std::vector<std::pair<int, int>>& correspondences) {
    KeyFrame& kf_mut = kf_db_.getKeyFrameMutable(kf.id);
    for (const auto& [mp_id, kp_idx] : correspondences) {
        if (!hasMapPoint(mp_id)) continue; // should never happen I guess ...

        MapPoint& mp = getMapPoint(mp_id);
        mp.addObservation(kf.id, kp_idx);
        mp.updateDescriptor(kf_db_);
        mp.updateViewingDirection(kf_db_);
        mp.updateDepthRange(kf_db_, cfg_.orb_scale_factor, cfg_.orb_num_levels);
        kf_mut.keypoint_to_map_point[kp_idx] = mp_id;
    }
}

void Map::addNewMapPoints(const KeyFrame& kf, const std::vector<std::pair<int, Eigen::Vector3d>>& keypoints_3d_world) {
    KeyFrame& kf_mut = kf_db_.getKeyFrameMutable(kf.id);
    for (const auto& [kp_idx, p_world] : keypoints_3d_world) {
        // Skip keypoints that already have an associated map point
        if (kf_mut.keypoint_to_map_point.count(kp_idx)) continue;

        MapPoint& mp = addMapPoint(p_world);
        mp.addObservation(kf.id, kp_idx);
        mp.updateDescriptor(kf_db_);
        mp.updateViewingDirection(kf_db_);
        mp.updateDepthRange(kf_db_, cfg_.orb_scale_factor, cfg_.orb_num_levels);
        kf_mut.keypoint_to_map_point[kp_idx] = mp.getId();
    }
}

void Map::updateCovisibility(const KeyFrame& kf) {
    KeyFrame& kf_mut = kf_db_.getKeyFrameMutable(kf.id);

    // Count shared map points of new keyframe with each other keyframe
    // Iterate through map points observed by new keyframe
    // Go through their observations
    std::unordered_map<int, int> shared_count;
    for (const auto& [kp_idx, mp_id] : kf_mut.keypoint_to_map_point) {
        if (!hasMapPoint(mp_id)) continue;
        for (const auto& [kf_id, obs_kp_idx] : getMapPoint(mp_id).getObservations()) {
            if (kf_id == kf.id) continue;
            shared_count[kf_id]++;
        }
    }

    // Add bidirectional edges where count >= threshold
    kf_mut.covisible_keyframes.clear();
    for (const auto& [neighbor_id, weight] : shared_count) {
        if (weight >= cfg_.covisibility_threshold) {
            if (kf_db_.hasKeyFrame(neighbor_id)) {
                kf_mut.covisible_keyframes[neighbor_id] = weight;
                kf_db_.getKeyFrameMutable(neighbor_id).covisible_keyframes[kf.id] = weight;
            }
        }
    }
}

MapPoint& Map::addMapPoint(const Eigen::Vector3d& position) {
    int id = next_id_++;
    map_points_.emplace(id, MapPoint(id, position));
    return map_points_.at(id);
}

void Map::removeMapPoint(int id) {
    map_points_.erase(id);
}

MapPoint& Map::getMapPoint(int id) {
    return map_points_.at(id);
}

const MapPoint& Map::getMapPoint(int id) const {
    return map_points_.at(id);
}

bool Map::hasMapPoint(int id) const {
    return map_points_.count(id) > 0;
}

int Map::cullMapPoints(int current_kf_id) {
    std::vector<int> to_remove;

    for (const auto& [mp_id, mp] : map_points_) {
        const auto& obs = mp.getObservations();
        if (obs.empty()) {
            to_remove.push_back(mp_id);
            continue;
        }

        // Find earliest observer
        int min_kf_id = std::numeric_limits<int>::max();
        for (const auto& [kf_id, _] : obs) {
            min_kf_id = std::min(min_kf_id, kf_id);
        }

        // If ≥3 keyframes have passed since creation and still <2 observations → cull
        if (current_kf_id - min_kf_id >= 3 && mp.numObservations() < 2) {
            to_remove.push_back(mp_id);
            continue;
        }

        // Cull points that are frequently outliers (visible enough times but rarely inlier)
        if (mp.getVisibleCount() >= 10 && mp.getFoundRatio() < 0.25) {
            to_remove.push_back(mp_id);
        }
    }

    // Clean up keyframe references and remove points
    for (int mp_id : to_remove) {
        const auto& obs = map_points_.at(mp_id).getObservations();
        for (const auto& [kf_id, kp_idx] : obs) {
            if (kf_db_.hasKeyFrame(kf_id)) {
                kf_db_.getKeyFrameMutable(kf_id).keypoint_to_map_point.erase(kp_idx);
            }
        }
        map_points_.erase(mp_id);
    }

    return static_cast<int>(to_remove.size());
}
