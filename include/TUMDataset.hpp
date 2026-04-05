#pragma once

#include "Dataset.hpp"
#include <string>
#include <utility>
#include <tuple>

class TUMDataset : public Dataset {
public:
    TUMDataset(const Config& cfg);

    Frame getFrame(int idx) override;

private:
    std::string dataset_path_;

    using TimestampedFile = std::pair<double, std::string>;
    using GroundTruth = std::tuple<double, Eigen::Vector3d, Eigen::Quaterniond>;

    std::vector<TimestampedFile> parseTimestampFile(const std::string& filepath);
    std::vector<GroundTruth> parseGroundTruth(const std::string& filepath);
};
