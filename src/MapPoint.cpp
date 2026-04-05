#include "MapPoint.hpp"
#include "KeyFrameDatabase.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <opencv2/core.hpp>

MapPoint::MapPoint(int id, const Eigen::Vector3d& position) : id_(id), position_(position) {}

void MapPoint::addObservation(int kf_id, int keypoint_idx) {
    observations_[kf_id] = keypoint_idx;
}

void MapPoint::removeObservation(int kf_id) {
    observations_.erase(kf_id);
    if (representative_kf_id_ == kf_id) {
        representative_kf_id_ = -1;
        representative_descriptor_idx_ = -1;
    }
}

void MapPoint::updateDescriptor(const KeyFrameDatabase& kf_db) {
    std::vector<std::pair<int, int>> obs_list;
    for (const auto& [kf_id, kp_idx] : observations_) {
        if (!kf_db.hasKeyFrame(kf_id)) continue;
        obs_list.emplace_back(kf_id, kp_idx);
    }

    const size_t N = obs_list.size();
    if (N == 0) return;

    // Pairwise Hamming distances
    std::vector<std::vector<int>> distances(N, std::vector<int>(N, 0));
    for (size_t i = 0; i < N; ++i) {
        const auto& desc_i = kf_db.getKeyFrame(obs_list[i].first).descriptors.row(obs_list[i].second);
        for (size_t j = i + 1; j < N; ++j) {
            const auto& desc_j = kf_db.getKeyFrame(obs_list[j].first).descriptors.row(obs_list[j].second);
            int d = cv::norm(desc_i, desc_j, cv::NORM_HAMMING);
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }

    // Select medoid: lowest median distance
    int best_median = INT_MAX;
    size_t best_idx = 0;
    for (size_t i = 0; i < N; ++i) {
        auto dists = distances[i];
        std::sort(dists.begin(), dists.end());
        int median = dists[(N - 1) / 2];
        if (median < best_median) {
            best_median = median;
            best_idx = i;
        }
    }

    representative_kf_id_ = obs_list[best_idx].first;
    representative_descriptor_idx_ = obs_list[best_idx].second;
}

void MapPoint::updateViewingDirection(const KeyFrameDatabase& kf_db) {
    Eigen::Vector3d mean_dir = Eigen::Vector3d::Zero();
    int count = 0;

    for (const auto& [kf_id, _] : observations_) {
        if (!kf_db.hasKeyFrame(kf_id)) continue;
        Eigen::Vector3d dir = position_ - kf_db.getKeyFrame(kf_id).estimated_pose.translation();
        mean_dir += dir.normalized();
        ++count;
    }

    if (count > 0) {
        viewing_direction_ = (mean_dir / count).normalized();
    }
}

void MapPoint::updateDepthRange(const KeyFrameDatabase& kf_db, float scale_factor, int num_levels) {
    if (representative_kf_id_ < 0 || !kf_db.hasKeyFrame(representative_kf_id_)) return;

    float dist = (position_ - kf_db.getKeyFrame(representative_kf_id_).estimated_pose.translation()).norm();
    max_distance_ = dist * scale_factor;
    min_distance_ = max_distance_ / std::pow(scale_factor, num_levels - 1);
}
