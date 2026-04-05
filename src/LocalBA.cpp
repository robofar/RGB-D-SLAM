#include "LocalBA.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>

#include <unordered_set>
#include <algorithm>
#include <iostream>

using gtsam::symbol_shorthand::X;  // poses
using gtsam::symbol_shorthand::L;  // landmarks

static gtsam::Pose3 toGtsam(const Sophus::SE3d& T_wc) {
    return gtsam::Pose3(
        gtsam::Rot3(T_wc.rotationMatrix()),
        gtsam::Point3(T_wc.translation())
    );
}

static Sophus::SE3d fromGtsam(const gtsam::Pose3& pose) {
    Eigen::Quaterniond q(pose.rotation().matrix());
    q.normalize();
    return Sophus::SE3d(q, pose.translation());
}

void runLocalBA(int current_kf_id, Map& map, const Intrinsic& intrinsic, const Config& cfg) {
    const KeyFrame& current_kf = map.getKeyFrame(current_kf_id);

    // 1. Collect local keyframes (current + top-N strongly covisible)
    std::vector<std::pair<int, int>> candidates; // (weight, kf_id)
    for (const auto& [neighbor_id, weight] : current_kf.covisible_keyframes) {
        if (weight >= cfg.local_ba_covisibility_threshold) {
            candidates.emplace_back(weight, neighbor_id);
        }
    }
    std::sort(candidates.begin(), candidates.end(), std::greater<>());

    std::unordered_set<int> local_kf_ids;
    local_kf_ids.insert(current_kf_id);
    int max_neighbors = cfg.local_ba_max_keyframes - 1; // -1 for current KF
    for (int i = 0; i < std::min(static_cast<int>(candidates.size()), max_neighbors); ++i) {
        local_kf_ids.insert(candidates[i].second);
        std::cout << "[BA]   Covisible KF " << candidates[i].second
                  << " (weight: " << candidates[i].first << ")" << std::endl;
    }

    // 2. Collect map points and fixed KFs in a single pass
    std::unordered_set<int> seen_mp_ids;
    std::unordered_set<int> local_mp_ids;
    std::unordered_set<int> fixed_kf_ids;

    for (int kf_id : local_kf_ids) {
        const KeyFrame& kf = map.getKeyFrame(kf_id);
        for (const auto& [kp_idx, mp_id] : kf.keypoint_to_map_point) {
            if (!map.hasMapPoint(mp_id) || seen_mp_ids.count(mp_id)) continue;
            seen_mp_ids.insert(mp_id);

            const auto& obs = map.getMapPoint(mp_id).getObservations();
            int obs_count = 0;
            for (const auto& [obs_kf_id, obs_kp_idx] : obs) {
                if (local_kf_ids.count(obs_kf_id)) {
                    obs_count++;
                } else if (map.hasKeyFrame(obs_kf_id)) {
                    fixed_kf_ids.insert(obs_kf_id);
                    obs_count++;
                }
            }

            if (obs_count >= 2) {
                local_mp_ids.insert(mp_id);
            }
        }
    }

    if (local_mp_ids.empty()) return;

    std::cout << "[BA] Optimizing " << local_kf_ids.size() << " KFs, "
              << fixed_kf_ids.size() << " fixed KFs, "
              << local_mp_ids.size() << " points" << std::endl;

    // 3. Build factor graph
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    auto K = gtsam::make_shared<gtsam::Cal3_S2>(
        intrinsic.fx, intrinsic.fy, 0.0, intrinsic.cx, intrinsic.cy);

    auto measurement_noise = gtsam::noiseModel::Isotropic::Sigma(2, cfg.local_ba_noise_sigma);
    auto huber = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(cfg.local_ba_huber_threshold), measurement_noise);

    // Add local KF poses (optimizable)
    for (int kf_id : local_kf_ids) {
        initial.insert(X(kf_id), toGtsam(map.getKeyFrame(kf_id).estimated_pose));
    }

    // Gauge prior on oldest local KF
    int oldest_kf_id = *std::min_element(local_kf_ids.begin(), local_kf_ids.end());
    auto anchor_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);
    graph.addPrior(X(oldest_kf_id), toGtsam(map.getKeyFrame(oldest_kf_id).estimated_pose), anchor_noise);

    // Add fixed KF poses with strong priors
    auto fixed_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-6);
    for (int kf_id : fixed_kf_ids) {
        gtsam::Pose3 pose = toGtsam(map.getKeyFrame(kf_id).estimated_pose);
        initial.insert(X(kf_id), pose);
        graph.addPrior(X(kf_id), pose, fixed_noise);
    }

    // Add landmarks with depth priors and projection factors
    auto depth_prior_noise = gtsam::noiseModel::Isotropic::Sigma(3, cfg.local_ba_depth_sigma);
    for (int mp_id : local_mp_ids) {
        const MapPoint& mp = map.getMapPoint(mp_id);
        initial.insert(L(mp_id), mp.getPosition());
        graph.addPrior(L(mp_id), mp.getPosition(), depth_prior_noise);

        for (const auto& [kf_id, kp_idx] : mp.getObservations()) {
            if (!local_kf_ids.count(kf_id) && !fixed_kf_ids.count(kf_id)) continue;

            const KeyFrame& kf = map.getKeyFrame(kf_id);
            const cv::KeyPoint& kp = kf.keypoints[kp_idx];
            gtsam::Point2 measurement(kp.pt.x, kp.pt.y);

            graph.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(
                measurement, huber, X(kf_id), L(mp_id), K);
        }
    }

    // 4. Optimize
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = cfg.local_ba_max_iterations;
    params.setVerbosityLM("SILENT");

    try {
        double initial_error = graph.error(initial);
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();
        double final_error = graph.error(result);
        std::cout << "[BA] Error: " << initial_error << " -> " << final_error
                  << " (iterations: " << optimizer.iterations() << ")" << std::endl;

        // 5. Write back results
        for (int kf_id : local_kf_ids) {
            map.getKeyFrameMutable(kf_id).estimated_pose = fromGtsam(result.at<gtsam::Pose3>(X(kf_id)));
        }

        int rejected = 0;
        for (int mp_id : local_mp_ids) {
            gtsam::Point3 optimized = result.at<gtsam::Point3>(L(mp_id));
            double correction = (optimized - map.getMapPoint(mp_id).getPosition()).norm();
            if (correction > cfg.local_ba_max_point_correction) {
                rejected++;
                continue;
            }
            map.getMapPoint(mp_id).setPosition(optimized);
        }
        if (rejected > 0) {
            std::cout << "[BA] Rejected " << rejected << " point corrections > "
                      << cfg.local_ba_max_point_correction << "m" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[BA] Local BA failed: " << e.what() << std::endl;
    }
}
