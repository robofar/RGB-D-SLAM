#include "FeatureManager.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

FeatureManager::FeatureManager(const Config& cfg) : cfg_(cfg) {
    orb_ = cv::ORB::create(
        cfg_.orb_num_features * 3,  // large detection pool, grid-binning reduces to orb_num_features
        cfg_.orb_scale_factor,
        cfg_.orb_num_levels,
        cfg_.orb_edge_threshold,
        0,                        // firstLevel
        2,                        // WTA_K
        cv::ORB::HARRIS_SCORE,
        cfg_.orb_patch_size,
        cfg_.orb_fast_threshold
    );
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
}

void FeatureManager::detectFeatures(Frame& frame) {
    cv::Mat gray;
    cv::cvtColor(frame.rgb, gray, cv::COLOR_BGR2GRAY);

    // Detect large candidate pool
    std::vector<cv::KeyPoint> all_kps;
    cv::Mat all_descs;
    orb_->detectAndCompute(gray, cv::noArray(), all_kps, all_descs);

    if (all_kps.empty()) {
        frame.keypoints.clear();
        frame.descriptors = cv::Mat();
        return;
    }

    // Grid-bin: keep top-N by response per cell for spatial uniformity
    const int cell = cfg_.orb_grid_cell_size;
    const int gcols = (gray.cols + cell - 1) / cell;
    const int grows = (gray.rows + cell - 1) / cell;
    const int per_cell = std::max(1, cfg_.orb_num_features / (gcols * grows));

    std::vector<std::vector<int>> cells(gcols * grows);
    for (int i = 0; i < static_cast<int>(all_kps.size()); ++i) {
        int gx = std::min(static_cast<int>(all_kps[i].pt.x / cell), gcols - 1);
        int gy = std::min(static_cast<int>(all_kps[i].pt.y / cell), grows - 1);
        cells[gy * gcols + gx].push_back(i);
    }

    std::vector<int> kept;
    kept.reserve(cfg_.orb_num_features);
    for (auto& c : cells) {
        std::sort(c.begin(), c.end(), [&](int a, int b) {
            return all_kps[a].response > all_kps[b].response;
        });
        int n = std::min(static_cast<int>(c.size()), per_cell);
        for (int i = 0; i < n; ++i) kept.push_back(c[i]);
    }

    frame.keypoints.clear();
    frame.keypoints.reserve(kept.size());
    frame.descriptors = cv::Mat(static_cast<int>(kept.size()), all_descs.cols, all_descs.type());
    for (int i = 0; i < static_cast<int>(kept.size()); ++i) {
        frame.keypoints.push_back(all_kps[kept[i]]);
        all_descs.row(kept[i]).copyTo(frame.descriptors.row(i));
    }
}

std::vector<std::pair<int, int>> FeatureManager::matchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    std::vector<std::pair<int, int>> matches;
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < cfg_.match_ratio_threshold * m[1].distance) {
            matches.emplace_back(m[0].queryIdx, m[0].trainIdx);
        }
    }
    return matches;
}
