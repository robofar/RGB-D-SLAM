#include "KeyFrameDatabase.hpp"
#include <stdexcept>
#include <iostream>

const KeyFrame& KeyFrameDatabase::createKeyFrame(const Frame& frame) {
    KeyFrame kf;
    kf.id = next_kf_id_++;
    kf.descriptors = frame.descriptors.clone();
    kf.keypoints = frame.keypoints;
    kf.gt_pose = frame.gt_pose;
    kf.estimated_pose = frame.estimated_pose;
    addKeyFrame(kf);
    return keyframes_.at(kf.id);
}

void KeyFrameDatabase::addKeyFrame(const KeyFrame& kf) {
    keyframes_[kf.id] = kf;
}

void KeyFrameDatabase::removeKeyFrame(int kf_id) {
    keyframes_.erase(kf_id);
}

const KeyFrame& KeyFrameDatabase::getKeyFrame(int kf_id) const {
    return keyframes_.at(kf_id);
}

KeyFrame& KeyFrameDatabase::getKeyFrameMutable(int kf_id) {
    return keyframes_.at(kf_id);
}

bool KeyFrameDatabase::hasKeyFrame(int kf_id) const {
    return keyframes_.count(kf_id) > 0;
}
