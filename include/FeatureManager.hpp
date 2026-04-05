#pragma once

#include "Types.hpp"
#include "Config.hpp"
#include <opencv2/features2d.hpp>

class FeatureManager {
public:
    FeatureManager(const Config& cfg);

    void detectFeatures(Frame& frame);
    std::vector<std::pair<int, int>> matchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2);

private:
    Config cfg_;
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::BFMatcher> matcher_;
};
