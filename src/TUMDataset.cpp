#include "TUMDataset.hpp"

#include <fstream>
#include <sstream>
#include <cmath>
#include <set>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>

static size_t findClosest(double timestamp, const std::vector<double>& timestamps) {
    size_t best = 0;
    double best_diff = std::numeric_limits<double>::max();

    for (size_t i = 0; i < timestamps.size(); ++i) {
        double diff = std::abs(timestamps[i] - timestamp);
        if (diff < best_diff) {
            best_diff = diff;
            best = i;
        }
    }
    return best;
}

TUMDataset::TUMDataset(const Config& cfg) : dataset_path_(cfg.base_path + "/" + cfg.sequence) {
    intrinsic_ = {535.4, 539.2, 320.1, 247.6, 640, 480, 5000.0};

    auto depth_entries = parseTimestampFile(dataset_path_ + "/depth.txt");
    auto rgb_entries = parseTimestampFile(dataset_path_ + "/rgb.txt");
    auto gt_entries = parseGroundTruth(dataset_path_ + "/groundtruth.txt");

    std::vector<double> rgb_ts, gt_ts;
    for (const auto& rgb : rgb_entries) rgb_ts.push_back(rgb.first);
    for (const auto& gt : gt_entries) gt_ts.push_back(std::get<0>(gt));

    // Match each depth frame to closest rgb (unique) and closest gt
    std::set<size_t> used_rgb;
    for (const auto& [depth_ts, depth_file] : depth_entries) {
        size_t rgb_idx = findClosest(depth_ts, rgb_ts);
        if (!used_rgb.insert(rgb_idx).second) continue; // if rgb_idx already in set (i.e. prev depth already matched to it)
        size_t gt_idx = findClosest(depth_ts, gt_ts);

        Association assoc;
        assoc.rgb_image_path = dataset_path_ + "/" + rgb_entries[rgb_idx].second;
        assoc.depth_image_path = dataset_path_ + "/" + depth_file;
        assoc.gt_pose = Sophus::SE3d(std::get<2>(gt_entries[gt_idx]), std::get<1>(gt_entries[gt_idx]));
        associations_.push_back(assoc);
    }
}

Frame TUMDataset::getFrame(int idx) {
    if (idx < 0 || idx >= static_cast<int>(associations_.size())) {
        throw std::out_of_range("Frame index out of range");
    }

    const auto& assoc = associations_[idx];
    Frame frame;
    frame.frame_id = idx;
    frame.rgb = cv::imread(assoc.rgb_image_path, cv::IMREAD_COLOR);
    frame.depth = cv::imread(assoc.depth_image_path, cv::IMREAD_UNCHANGED);
    frame.gt_pose = assoc.gt_pose;

    if (frame.rgb.empty()) throw std::runtime_error("Failed to load RGB: " + assoc.rgb_image_path);
    if (frame.depth.empty()) throw std::runtime_error("Failed to load depth: " + assoc.depth_image_path);

    return frame;
}

std::vector<TUMDataset::TimestampedFile> TUMDataset::parseTimestampFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Failed to open: " + filepath);

    std::vector<TimestampedFile> entries;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double ts; 
        std::string fn;
        iss >> ts >> fn;
        entries.emplace_back(ts, fn);
    }
    return entries;
}

std::vector<TUMDataset::GroundTruth> TUMDataset::parseGroundTruth(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Failed to open: " + filepath);

    std::vector<GroundTruth> entries;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double ts, qx, qy, qz, qw;
        Eigen::Vector3d t;
        iss >> ts >> t.x() >> t.y() >> t.z() >> qx >> qy >> qz >> qw;
        entries.emplace_back(ts, t, Eigen::Quaterniond(qw, qx, qy, qz));
    }
    return entries;
}
