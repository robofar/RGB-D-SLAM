#include "Tracking.hpp"
#include "Utils.hpp"
#include "PoseEstimator.hpp"
#include "LocalBA.hpp"
#include <iostream>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <chrono>

Tracking::Tracking(const Config& cfg, const Intrinsic& intrinsic, FeatureManager& feature_manager, Map& map)
    : cfg_(cfg), intrinsic_(intrinsic), feature_manager_(feature_manager), map_(map) {}

bool Tracking::processFrame(Frame& frame) {
    auto t0 = std::chrono::high_resolution_clock::now();
    feature_manager_.detectFeatures(frame);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[F] Frame " << frame.frame_id << ": " << frame.keypoints.size() << " keypoints detected." << std::endl;

    if (!initialized_) {
        frame.estimated_pose = frame.gt_pose;
        const KeyFrame& kf = map_.createKeyFrame(frame);

        auto keypoints_3d_world = backprojectKeypoints(kf, frame.depth, intrinsic_);
        int num_invalid_depth = kf.keypoints.size() - keypoints_3d_world.size();
        map_.addNewMapPoints(kf, keypoints_3d_world);

        if (cfg_.verbose) {
            std::cout << "[KF] Created KF " << kf.id
                      << " | associated: 0"
                      << " | new: " << kf.keypoint_to_map_point.size()
                      << " | invalid depth: " << num_invalid_depth
                      << " | covisible: 0" << std::endl;
            std::cout << "[M] Map size: " << map_.numMapPoints() << " points, "
                      << map_.numKeyFrames() << " keyframes" << std::endl;
            std::cout << "--------------------" << std::endl;
        }

        prev_T_ = frame.estimated_pose;
        delta_T_ = Sophus::SE3d();  // identity

        last_kf_id_ = kf.id;
        initialized_ = true;
        frames_since_last_kf_ = 0;
        return true;
    }

    frames_since_last_kf_++;

    // (1) - Predict pose using constant velocity model
    Sophus::SE3d predicted_pose = prev_T_ * delta_T_;

    // (2) - Find 3D-2D correspondences using constant velocity prediction
    auto t2 = std::chrono::high_resolution_clock::now();
    auto correspondences = findCorrespondences(frame, predicted_pose, cfg_.tracking_search_radius_cv);
    if (cfg_.verbose) std::cout << "[T] " << correspondences.size() << " matches (CV model)" << std::endl;

    // Fallback: use previous pose with wider search radius
    if (static_cast<int>(correspondences.size()) < cfg_.tracking_min_matches) {
        correspondences = findCorrespondences(frame, prev_T_, cfg_.tracking_search_radius_wide);
        if (cfg_.verbose) std::cout << "[T] " << correspondences.size() << " matches (wide fallback)" << std::endl;
    }

    if (static_cast<int>(correspondences.size()) < cfg_.tracking_min_matches) {
        if (cfg_.verbose) std::cout << "[T] Tracking lost" << std::endl;
        if (cfg_.verbose) std::cout << "--------------------" << std::endl;
        return false;
    }

    // Build points_3d / points_2d from correspondences for PnP
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    points_3d.reserve(correspondences.size());
    points_2d.reserve(correspondences.size());
    for (const auto& [mp_id, kp_idx] : correspondences) {
        const Eigen::Vector3d& p = map_.getMapPoint(mp_id).getPosition();
        points_3d.emplace_back(p.x(), p.y(), p.z());
        points_2d.emplace_back(frame.keypoints[kp_idx].pt);
    }

    // (3) - Estimate pose (AP3P + RANSAC)
    auto t3 = std::chrono::high_resolution_clock::now();
    auto pnp_result = estimatePosePnP(points_3d, points_2d, intrinsic_);
    if (!pnp_result.pose) {
        if (cfg_.verbose) std::cout << "[T] PnP failed" << std::endl;
        if (cfg_.verbose) std::cout << "--------------------" << std::endl;
        return false;
    }
    if (cfg_.verbose) std::cout << "[T] PnP inliers: " << pnp_result.inlier_indices.size() << "/" << points_3d.size() << std::endl;

    // (3.5) - Optionally refine PnP pose with motion-BA (inliers only)
    Sophus::SE3d refined_pose = pnp_result.pose.value();
    if (cfg_.refine_pnp_pose) {
        std::vector<cv::Point3f> inlier_3d;
        std::vector<cv::Point2f> inlier_2d;
        inlier_3d.reserve(pnp_result.inlier_indices.size());
        inlier_2d.reserve(pnp_result.inlier_indices.size());
        for (int idx : pnp_result.inlier_indices) {
            inlier_3d.push_back(points_3d[idx]);
            inlier_2d.push_back(points_2d[idx]);
        }
        refined_pose = motionBA(inlier_3d, inlier_2d, intrinsic_, pnp_result.pose.value(), cfg_.motion_ba_iterations);
    }

    // (4) - Search local map for more correspondences using refined pose
    auto t4 = std::chrono::high_resolution_clock::now();
    auto local_correspondences = findCorrespondences(frame, refined_pose, cfg_.tracking_search_radius_local, true);
    if (cfg_.verbose) std::cout << "[T] " << local_correspondences.size() << " matches (local map)" << std::endl;

    std::vector<cv::Point3f> local_3d;
    std::vector<cv::Point2f> local_2d;
    local_3d.reserve(local_correspondences.size());
    local_2d.reserve(local_correspondences.size());
    for (const auto& [mp_id, kp_idx] : local_correspondences) {
        const Eigen::Vector3d& p = map_.getMapPoint(mp_id).getPosition();
        local_3d.emplace_back(p.x(), p.y(), p.z());
        local_2d.emplace_back(frame.keypoints[kp_idx].pt);
    }

    // (5) - Motion-BA using local map correspondences
    auto t5 = std::chrono::high_resolution_clock::now();
    std::vector<bool> inlier_mask;
    frame.estimated_pose = motionBA(local_3d, local_2d, intrinsic_, refined_pose, cfg_.motion_ba_iterations, &inlier_mask);
    auto t6 = std::chrono::high_resolution_clock::now();

    // Update map point tracking statistics
    for (int i = 0; i < static_cast<int>(local_correspondences.size()); ++i) {
        int mp_id = local_correspondences[i].first;
        if (!map_.hasMapPoint(mp_id)) continue;
        MapPoint& mp = map_.getMapPoint(mp_id);
        mp.incrementVisible();
        if (inlier_mask[i]) mp.incrementFound();
    }

    if (cfg_.verbose) std::cout << "[T] Motion-BA refined pose with " << local_3d.size() << " points" << std::endl;

    if(cfg_.verbose_perf) {
        std::cout << "[PERF] ORB: " << std::chrono::duration<double, std::milli>(t1 - t0).count()
                << " | CV match: " << std::chrono::duration<double, std::milli>(t3 - t2).count()
                << " | PnP: " << std::chrono::duration<double, std::milli>(t4 - t3).count()
                << " | Local map: " << std::chrono::duration<double, std::milli>(t5 - t4).count()
                << " | Motion-BA: " << std::chrono::duration<double, std::milli>(t6 - t5).count()
                << " ms" << std::endl;
    }

    // Update motion model
    delta_T_ = prev_T_.inverse() * frame.estimated_pose;
    prev_T_ = frame.estimated_pose;

    // (6) - Keyframe decision
    int num_tracked = static_cast<int>(local_correspondences.size());
    int ref_kf_tracked = static_cast<int>(map_.getKeyFrame(last_kf_id_).keypoint_to_map_point.size());

    bool condition1 = frames_since_last_kf_ >= 20
                      || num_tracked < 100
                      || num_tracked < static_cast<int>(0.25 * ref_kf_tracked);
    bool condition2 = num_tracked < static_cast<int>(0.9 * ref_kf_tracked);

    if (condition1 && condition2) {
        const KeyFrame& kf = map_.createKeyFrame(frame);

        // Associate tracked map points with new keyframe (use local map correspondences)
        map_.associateMapPoints(kf, local_correspondences);
        int num_associated = kf.keypoint_to_map_point.size();

        // Create new map points for unmatched keypoints
        auto keypoints_3d_world = backprojectKeypoints(kf, frame.depth, intrinsic_);
        int num_invalid_depth = kf.keypoints.size() - keypoints_3d_world.size();
        map_.addNewMapPoints(kf, keypoints_3d_world);
        int num_new = kf.keypoint_to_map_point.size() - num_associated;

        // Update covisibility graph
        map_.updateCovisibility(kf);

        // Cull map points with <3 observations after ≥2 keyframes
        int num_culled = map_.cullMapPoints(kf.id);

        // Local Bundle Adjustment
        if (cfg_.local_ba) {
            auto t_ba0 = std::chrono::high_resolution_clock::now();
            runLocalBA(kf.id, map_, intrinsic_, cfg_);
            auto t_ba1 = std::chrono::high_resolution_clock::now();
            std::cout << "[BA] Local BA: "
                      << std::chrono::duration<double, std::milli>(t_ba1 - t_ba0).count()
                      << " ms" << std::endl;

            // Recompute motion model with BA-optimized pose
            Sophus::SE3d ba_pose = map_.getKeyFrame(kf.id).estimated_pose;
            delta_T_ = prev_T_.inverse() * ba_pose;
            prev_T_ = ba_pose;
            frame.estimated_pose = ba_pose;
        }

        if (cfg_.verbose) {
            std::cout << "[KF] Created KF " << kf.id
                      << " | associated: " << num_associated
                      << " | new: " << num_new
                      << " | invalid depth: " << num_invalid_depth
                      << " | culled: " << num_culled
                      << " | covisible: " << kf.covisible_keyframes.size() << std::endl;
            std::cout << "[M] Map size: " << map_.numMapPoints() << " points, "
                      << map_.numKeyFrames() << " keyframes" << std::endl;
        }

        last_kf_id_ = kf.id;
        frames_since_last_kf_ = 0;
    }
    if (cfg_.verbose) std::cout << "--------------------" << std::endl;
    return true;
}

std::vector<std::pair<int, int>>
Tracking::findCorrespondences(const Frame& frame, const Sophus::SE3d& pose, double search_radius, bool use_local_map) {

    // Collect map point IDs to search
    std::unordered_set<int> mp_ids;
    const KeyFrame& last_kf = map_.getKeyFrame(last_kf_id_);

    if (!use_local_map) {
        // Only last keyframe's map points
        for (const auto& [kp_idx, mp_id] : last_kf.keypoint_to_map_point) {
            if (map_.hasMapPoint(mp_id)) mp_ids.insert(mp_id);
        }
    } else {

        // K1: keyframes that share map points with last KF (via covisibility graph)
        std::unordered_set<int> k1_ids = {last_kf_id_};
        for (const auto& [neighbor_id, weight] : last_kf.covisible_keyframes) {
            k1_ids.insert(neighbor_id);
        }
        // Previous approach (exact, but slow — iterates all observations):
        // for (const auto& [kp_idx, mp_id] : last_kf.keypoint_to_map_point) {
        //     if (!map_.hasMapPoint(mp_id)) continue;
        //     for (const auto& [obs_kf_id, _] : map_.getMapPoint(mp_id).getObservations()) {
        //         if (obs_kf_id != last_kf_id_) k1_ids.insert(obs_kf_id);
        //     }
        // }

        // K2: covisible neighbors of K1
        std::unordered_set<int> k2_ids;
        for (int kf_id : k1_ids) {
            if (!map_.hasKeyFrame(kf_id)) continue;
            for (const auto& [neighbor_id, weight] : map_.getKeyFrame(kf_id).covisible_keyframes) {
                if (!k1_ids.count(neighbor_id)) k2_ids.insert(neighbor_id);
            }
        }

        // Collect all map points from K1 ∪ K2
        for (int kf_id : k1_ids) {
            if (!map_.hasKeyFrame(kf_id)) continue;
            for (const auto& [kp_idx, mp_id] : map_.getKeyFrame(kf_id).keypoint_to_map_point) {
                if (map_.hasMapPoint(mp_id)) mp_ids.insert(mp_id);
            }
        }
        for (int kf_id : k2_ids) {
            if (!map_.hasKeyFrame(kf_id)) continue;
            for (const auto& [kp_idx, mp_id] : map_.getKeyFrame(kf_id).keypoint_to_map_point) {
                if (map_.hasMapPoint(mp_id)) mp_ids.insert(mp_id);
            }
        }
    }

    // Build keypoint grid (cell size = search_radius)
    int grid_cols = static_cast<int>(std::ceil(intrinsic_.width / search_radius));
    int grid_rows = static_cast<int>(std::ceil(intrinsic_.height / search_radius));
    std::vector<std::vector<int>> grid(grid_rows * grid_cols);

    for (int i = 0; i < static_cast<int>(frame.keypoints.size()); ++i) {
        int cx = static_cast<int>(frame.keypoints[i].pt.x / search_radius);
        int cy = static_cast<int>(frame.keypoints[i].pt.y / search_radius);
        cx = std::clamp(cx, 0, grid_cols - 1);
        cy = std::clamp(cy, 0, grid_rows - 1);
        grid[cy * grid_cols + cx].push_back(i);
    }

    Sophus::SE3d T_cw = pose.inverse();
    Eigen::Vector3d cam_center = pose.translation();
    std::vector<std::pair<int, int>> correspondences;  // (mp_id, kp_idx)
    std::unordered_set<int> matched_kp_indices;
    double search_radius_sq = search_radius * search_radius;

    for (int mp_id : mp_ids) {
        const MapPoint& mp = map_.getMapPoint(mp_id);
        Eigen::Vector3d p_world = mp.getPosition();

        // Check if point is in front of camera
        Eigen::Vector3d p_cam = T_cw * p_world;
        if (p_cam.z() <= 0) continue;

        // Project into frame
        Eigen::Vector2d projected = project(p_world, T_cw, intrinsic_);
        double u = projected.x();
        double v = projected.y();
        if (u < 0 || u >= intrinsic_.width || v < 0 || v >= intrinsic_.height) continue;

        if (use_local_map) {
            // Viewing angle check: v · n > cos(60°)
            Eigen::Vector3d view_ray = (p_world - cam_center).normalized();
            double cos_angle = view_ray.dot(mp.getViewingDirection());
            if (cos_angle < 0.5) continue;

            // Distance range check
            double d = (p_world - cam_center).norm();
            if (d < mp.getMinDistance() || d > mp.getMaxDistance()) continue;
        }

        // Get map point descriptor
        int rep_kf_id = mp.getRepresentativeKfId();
        int rep_desc_idx = mp.getRepresentativeDescriptorIdx();
        if (rep_kf_id < 0) continue;
        cv::Mat mp_desc = map_.getKeyFrame(rep_kf_id).descriptors.row(rep_desc_idx);

        // Search 3x3 grid neighborhood around projected position
        int cell_x = static_cast<int>(u / search_radius);
        int cell_y = static_cast<int>(v / search_radius);
        double best_dist = std::numeric_limits<double>::max();
        int best_kp_idx = -1;

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cell_x + dx;
                int ny = cell_y + dy;
                if (nx < 0 || nx >= grid_cols || ny < 0 || ny >= grid_rows) continue;

                for (int i : grid[ny * grid_cols + nx]) {
                    if (matched_kp_indices.count(i)) continue;

                    double ddx = frame.keypoints[i].pt.x - u;
                    double ddy = frame.keypoints[i].pt.y - v;
                    if (ddx * ddx + ddy * ddy > search_radius_sq) continue;

                    double hamming = cv::norm(mp_desc, frame.descriptors.row(i), cv::NORM_HAMMING);
                    if (hamming < best_dist) {
                        best_dist = hamming;
                        best_kp_idx = i;
                    }
                }
            }
        }

        if (best_kp_idx >= 0 && best_dist < cfg_.tracking_max_hamming_distance) {
            correspondences.emplace_back(mp_id, best_kp_idx);
            matched_kp_indices.insert(best_kp_idx);
        }
    }

    return correspondences;
}
