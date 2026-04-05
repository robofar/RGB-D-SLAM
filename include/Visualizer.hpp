#pragma once

#include "Types.hpp"
#include "Map.hpp"
#include <rerun.hpp>

class Visualizer {
public:
    Visualizer(const Intrinsic& intrinsic);

    void logMapPoints(const Map& map);
    void logKeyFrameMapPoints(const Map& map, int kf_id);
    void logFrameFrustum(const Frame& frame);
    void logKeyFrameFrustums(const Map& map);
    void logTrajectoryLines(const Map& map);
    void logCovisibilityGraph(const Map& map);
    void logImage(const Frame& frame);

    void setFrame(int frame_id);

private:
    rerun::RecordingStream rec_;
    Intrinsic intrinsic_;
};
