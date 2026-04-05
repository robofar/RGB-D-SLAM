#include "Visualizer.hpp"
#include <algorithm>
#include <unordered_set>

Visualizer::Visualizer(const Intrinsic& intrinsic)
    : rec_("fslam"), intrinsic_(intrinsic) {
    rec_.spawn().exit_on_failure();
    rec_.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
}

void Visualizer::setFrame(int frame_id) {
    rec_.set_time_sequence("frame", frame_id);
}

void Visualizer::logMapPoints(const Map& map) {
    std::vector<rerun::Position3D> positions;
    for (const auto& [id, mp] : map.getAllMapPoints()) {
        const auto& p = mp.getPosition();
        positions.push_back({static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())});
    }
    if (!positions.empty()) {
        rec_.log("world/map_points",
            rerun::Points3D(positions)
                .with_colors({rerun::Color(192, 192, 192)})
                .with_radii({0.005f})
        );
    }
}

void Visualizer::logKeyFrameMapPoints(const Map& map, int kf_id) {
    if (!map.hasKeyFrame(kf_id)) return;
    const KeyFrame& kf = map.getKeyFrame(kf_id);

    std::vector<rerun::Position3D> positions;
    for (const auto& [kp_idx, mp_id] : kf.keypoint_to_map_point) {
        if (!map.hasMapPoint(mp_id)) continue;
        const auto& p = map.getMapPoint(mp_id).getPosition();
        positions.push_back({static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())});
    }
    if (!positions.empty()) {
        rec_.log("world/kf_map_points",
            rerun::Points3D(positions)
                .with_colors({rerun::Color(255, 165, 0)})
                .with_radii({0.008f})
        );
    }
}

static rerun::Mat3x3 eigenToMat3x3(const Eigen::Matrix3d& R) {
    Eigen::Matrix3f Rf = R.cast<float>();
    // Eigen is column-major, Rerun Mat3x3 expects column-major
    return rerun::Mat3x3(Rf.data());
}

static rerun::Vec3D eigenToVec3D(const Eigen::Vector3d& t) {
    return rerun::Vec3D(std::array<float, 3>{static_cast<float>(t.x()), static_cast<float>(t.y()), static_cast<float>(t.z())});
}

void Visualizer::logFrameFrustum(const Frame& frame) {
    rec_.log("world/frame",
        rerun::Transform3D(
            eigenToVec3D(frame.estimated_pose.translation()),
            eigenToMat3x3(frame.estimated_pose.rotationMatrix())
        )
    );
    rec_.log("world/frame",
        rerun::Pinhole::from_focal_length_and_resolution(
            {static_cast<float>(intrinsic_.fx), static_cast<float>(intrinsic_.fy)},
            {static_cast<float>(intrinsic_.width), static_cast<float>(intrinsic_.height)}
        ).with_camera_xyz(rerun::components::ViewCoordinates::RDF)
         .with_image_plane_distance(0.1f)
         .with_color(rerun::Color(255, 0, 0))
         .with_line_width(0.003f)
    );
}

void Visualizer::logKeyFrameFrustums(const Map& map) {
    for (const auto& [kf_id, kf] : map.getAllKeyFrames()) {
        std::string entity = "world/keyframes/kf_" + std::to_string(kf_id);
        rec_.log(entity,
            rerun::Transform3D(
                eigenToVec3D(kf.estimated_pose.translation()),
                eigenToMat3x3(kf.estimated_pose.rotationMatrix())
            )
        );
        rec_.log(entity,
            rerun::Pinhole::from_focal_length_and_resolution(
                {static_cast<float>(intrinsic_.fx), static_cast<float>(intrinsic_.fy)},
                {static_cast<float>(intrinsic_.width), static_cast<float>(intrinsic_.height)}
            ).with_camera_xyz(rerun::components::ViewCoordinates::RDF)
             .with_image_plane_distance(0.05f)
             .with_color(rerun::Color(0, 0, 255))
             .with_line_width(0.003f)
        );

        // Ground truth pose (orange)
        std::string gt_entity = "world/keyframes_gt/kf_" + std::to_string(kf_id);
        rec_.log(gt_entity,
            rerun::Transform3D(
                eigenToVec3D(kf.gt_pose.translation()),
                eigenToMat3x3(kf.gt_pose.rotationMatrix())
            )
        );
        rec_.log(gt_entity,
            rerun::Pinhole::from_focal_length_and_resolution(
                {static_cast<float>(intrinsic_.fx), static_cast<float>(intrinsic_.fy)},
                {static_cast<float>(intrinsic_.width), static_cast<float>(intrinsic_.height)}
            ).with_camera_xyz(rerun::components::ViewCoordinates::RDF)
             .with_image_plane_distance(0.05f)
             .with_color(rerun::Color(255, 165, 0))
             .with_line_width(0.003f)
        );
    }
}

void Visualizer::logTrajectoryLines(const Map& map) {
    // Collect KFs sorted by id
    std::vector<std::pair<int, const KeyFrame*>> sorted_kfs;
    for (const auto& [kf_id, kf] : map.getAllKeyFrames())
        sorted_kfs.emplace_back(kf_id, &kf);
    std::sort(sorted_kfs.begin(), sorted_kfs.end());

    if (sorted_kfs.size() < 2) return;

    std::vector<rerun::Position3D> est_positions, gt_positions;
    est_positions.reserve(sorted_kfs.size());
    gt_positions.reserve(sorted_kfs.size());

    for (const auto& [kf_id, kf] : sorted_kfs) {
        const auto& et = kf->estimated_pose.translation();
        est_positions.push_back({static_cast<float>(et.x()), static_cast<float>(et.y()), static_cast<float>(et.z())});
        const auto& gt = kf->gt_pose.translation();
        gt_positions.push_back({static_cast<float>(gt.x()), static_cast<float>(gt.y()), static_cast<float>(gt.z())});
    }

    std::vector<rerun::LineStrip3D> est_line = {rerun::LineStrip3D(est_positions)};
    rec_.log("world/trajectory_est",
        rerun::LineStrips3D(est_line)
            .with_colors({rerun::Color(0, 120, 255)})
            .with_radii({0.006f})
    );
    std::vector<rerun::LineStrip3D> gt_line = {rerun::LineStrip3D(gt_positions)};
    rec_.log("world/trajectory_gt",
        rerun::LineStrips3D(gt_line)
            .with_colors({rerun::Color(255, 165, 0)})
            .with_radii({0.006f})
    );
}

void Visualizer::logCovisibilityGraph(const Map& map) {
    std::vector<rerun::LineStrip3D> lines;
    std::unordered_set<int64_t> logged_edges;

    for (const auto& [kf_id, kf] : map.getAllKeyFrames()) {
        Eigen::Vector3d t1 = kf.estimated_pose.translation();
        for (const auto& [neighbor_id, weight] : kf.covisible_keyframes) {
            // Avoid duplicate edges
            int64_t edge_key = std::min(kf_id, neighbor_id) * 100000L + std::max(kf_id, neighbor_id);
            if (logged_edges.count(edge_key)) continue;
            logged_edges.insert(edge_key);

            if (!map.hasKeyFrame(neighbor_id)) continue;
            Eigen::Vector3d t2 = map.getKeyFrame(neighbor_id).estimated_pose.translation();
            lines.push_back(rerun::LineStrip3D({
                rerun::Position3D{static_cast<float>(t1.x()), static_cast<float>(t1.y()), static_cast<float>(t1.z())},
                rerun::Position3D{static_cast<float>(t2.x()), static_cast<float>(t2.y()), static_cast<float>(t2.z())}
            }));
        }
    }
    if (!lines.empty()) {
        rec_.log("world/covisibility",
            rerun::LineStrips3D(lines)
                .with_colors({rerun::Color(0, 255, 0)})
                .with_radii({0.002f})
        );
    }
}

void Visualizer::logImage(const Frame& frame) {
    rec_.log("image/rgb",
        rerun::Image(
            reinterpret_cast<const uint8_t*>(frame.rgb.data),
            rerun::WidthHeight(static_cast<uint32_t>(frame.rgb.cols), static_cast<uint32_t>(frame.rgb.rows)),
            rerun::ColorModel::BGR
        )
    );

    std::vector<rerun::Position2D> kp_positions;
    kp_positions.reserve(frame.keypoints.size());
    for (const auto& kp : frame.keypoints) {
        kp_positions.push_back({kp.pt.x, kp.pt.y});
    }
    rec_.log("image/rgb/keypoints",
        rerun::Points2D(kp_positions)
            .with_colors({rerun::Color(0, 255, 0)})
            .with_radii({2.0f})
    );
}
