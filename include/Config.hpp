#pragma once

#include <string>

struct Config {
    std::string dataset_type = "tum"; // "tum" or "replica"
    std::string base_path = "/media/faris/1B6B-12E9/TUM";
    std::string sequence = "fr3_long_office_household";

    // ORB parameters
    int orb_num_features = 3000;
    float orb_scale_factor = 1.2f;
    int orb_num_levels = 8;
    int orb_edge_threshold = 31;
    int orb_patch_size = 31;
    int orb_fast_threshold = 20;
    int orb_fast_threshold_min = 7;
    int orb_grid_cell_size = 30;

    // Matching parameters
    float match_ratio_threshold = 0.9f;

    // Tracking parameters
    double tracking_search_radius_cv = 20.0;
    double tracking_search_radius_local = 15.0; // 15.0
    double tracking_search_radius_wide = 60.0;
    double tracking_max_hamming_distance = 50.0;
    int tracking_min_matches = 10;
    int motion_ba_iterations = 10;
    bool refine_pnp_pose = true;

    // Covisibility parameters
    int covisibility_threshold = 15;

    // Local Bundle Adjustment
    bool local_ba = true;
    int local_ba_covisibility_threshold = 20;
    int local_ba_max_keyframes = 7;
    double local_ba_noise_sigma = 1.0;       // measurement noise (pixels)
    double local_ba_huber_threshold = 1.345; // Huber transition point
    int local_ba_max_iterations = 10;
    double local_ba_depth_sigma = 0.005;     // depth prior sigma on landmarks (meters)
    double local_ba_max_point_correction = 0.5; // reject corrections larger than this (meters)

    // Visualization
    bool vis_online = false;
    bool vis_offline = true;

    // Logging
    bool verbose = false;
    bool verbose_perf = false;
};
