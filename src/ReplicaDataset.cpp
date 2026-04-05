#include "ReplicaDataset.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <iostream>

#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>

// Each line in traj.txt: 16 space-separated floats, row-major 4x4 T_world_camera
static std::vector<Sophus::SE3d> parseTraj(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filepath);

    std::vector<Sophus::SE3d> poses;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double v[16];
        for (int i = 0; i < 16; ++i) {
            ss >> v[i];
            if (ss.fail()) throw std::runtime_error("Malformed traj.txt line: " + line);
        }

        Eigen::Matrix4d M;
        M << v[0],  v[1],  v[2],  v[3],
             v[4],  v[5],  v[6],  v[7],
             v[8],  v[9],  v[10], v[11],
             v[12], v[13], v[14], v[15];

        Eigen::Matrix3d R = M.topLeftCorner<3, 3>();
        Eigen::Vector3d t = M.topRightCorner<3, 1>();
        Eigen::Quaterniond q(R);
        q.normalize();
        poses.emplace_back(q, t);
    }
    return poses;
}

static std::string padIndex(size_t i) {
    std::ostringstream ss;
    ss << std::setw(6) << std::setfill('0') << i;
    return ss.str();
}

ReplicaDataset::ReplicaDataset(const Config& cfg)
    : results_dir_(cfg.base_path + "/" + cfg.sequence + "/results")
{
    intrinsic_ = {600.0, 600.0, 599.5, 339.5, 1200, 680, 6553.5};

    auto poses = parseTraj(cfg.base_path + "/" + cfg.sequence + "/traj.txt");

    associations_.reserve(poses.size());
    for (size_t i = 0; i < poses.size(); ++i) {
        std::string id = padIndex(i);
        Association assoc;
        assoc.rgb_image_path = results_dir_ + "/frame" + id + ".jpg";
        assoc.depth_image_path = results_dir_ + "/depth" + id + ".png";
        assoc.gt_pose = poses[i];
        associations_.push_back(assoc);
    }

    std::cout << "ReplicaDataset: " << associations_.size() << " frames"
              << " (sequence=" << cfg.sequence << ")" << std::endl;
}

Frame ReplicaDataset::getFrame(int idx) {
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
