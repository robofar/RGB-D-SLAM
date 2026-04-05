#include "TUMDataset.hpp"
#include "ReplicaDataset.hpp"
#include "FeatureManager.hpp"
#include "Tracking.hpp"
#include "Visualizer.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    Config cfg;
    if (argc > 1) cfg.dataset_type = argv[1];
    if (argc > 2) cfg.base_path = argv[2];
    if (argc > 3) cfg.sequence = argv[3];
    if (argc > 4) cfg.local_ba = (std::string(argv[4]) == "1");

    std::unique_ptr<Dataset> dataset;
    if (cfg.dataset_type == "replica") {
        dataset = std::make_unique<ReplicaDataset>(cfg);
    } else {
        dataset = std::make_unique<TUMDataset>(cfg);
    }
    std::cout << "Loaded " << dataset->size() << " associations" << std::endl;
    std::cout << "--------------------" << std::endl;

    FeatureManager feature_manager(cfg);
    Map map(cfg);
    Tracking tracking(cfg, dataset->getIntrinsic(), feature_manager, map);

    std::unique_ptr<Visualizer> visualizer;
    if (cfg.vis_online || cfg.vis_offline) {
        visualizer = std::make_unique<Visualizer>(dataset->getIntrinsic());
    }

    int num_frames = std::min(static_cast<int>(dataset->size()), 10000);
    size_t prev_num_kf = 0;
    std::vector<Eigen::Vector3d> gt_translations, est_translations;

    for (int i = 0; i < num_frames; ++i) {
        Frame frame = dataset->getFrame(i);
        bool success = tracking.processFrame(frame);

        if (success) {
            gt_translations.push_back(frame.gt_pose.translation());
            est_translations.push_back(frame.estimated_pose.translation());
        }

        if (cfg.vis_online && visualizer) {
            visualizer->setFrame(frame.frame_id);
            visualizer->logImage(frame);

            if (map.numKeyFrames() > prev_num_kf) {
                visualizer->logMapPoints(map);
                visualizer->logKeyFrameMapPoints(map, tracking.getLastKeyFrameId());
                visualizer->logKeyFrameFrustums(map);
                visualizer->logCovisibilityGraph(map);
                prev_num_kf = map.numKeyFrames();
            } else {
                visualizer->logFrameFrustum(frame);
            }
        }
    }

    std::cout << "\nMap points: " << map.numMapPoints() << std::endl;
    std::cout << "Keyframes: " << map.numKeyFrames() << std::endl;

    // ATE evaluation — all frames
    if (!gt_translations.empty()) {
        double sum_sq = 0.0;
        for (size_t i = 0; i < gt_translations.size(); ++i) {
            double err = (gt_translations[i] - est_translations[i]).norm();
            sum_sq += err * err;
        }
        double ate_rmse = std::sqrt(sum_sq / gt_translations.size());
        std::cout << "\n[EVAL] ATE RMSE (all frames): " << ate_rmse << " m (" << gt_translations.size() << " frames)" << std::endl;
    }

    // ATE evaluation — keyframes only
    if (map.numKeyFrames() > 0) {
        double sum_sq = 0.0;
        for (const auto& [kf_id, kf] : map.getAllKeyFrames()) {
            double err = (kf.gt_pose.translation() - kf.estimated_pose.translation()).norm();
            sum_sq += err * err;
        }
        double ate_rmse = std::sqrt(sum_sq / map.numKeyFrames());
        std::cout << "[EVAL] ATE RMSE (keyframes): " << ate_rmse << " m (" << map.numKeyFrames() << " keyframes)" << std::endl;
    }

    if (cfg.vis_offline && visualizer) {
        visualizer->logMapPoints(map);
        visualizer->logKeyFrameFrustums(map);
        visualizer->logTrajectoryLines(map);
        visualizer->logCovisibilityGraph(map);
    }

    return 0;
}
