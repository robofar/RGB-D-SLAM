#include "PoseEstimator.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>

PnPResult estimatePosePnP(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    const Intrinsic& intrinsic) {

    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        intrinsic.fx, 0, intrinsic.cx,
        0, intrinsic.fy, intrinsic.cy,
        0, 0, 1);

    cv::Mat rvec, tvec, inliers;
    bool success = cv::solvePnPRansac(points_3d, points_2d, camera_matrix, cv::noArray(), rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_AP3P);

    if (!success) return {std::nullopt, {}};

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    Eigen::Matrix3d R_eigen;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R_eigen(i, j) = R.at<double>(i, j);

    Eigen::Vector3d t_eigen(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // PnP gives T_cw (camera-from-world), return T_wc (world-from-camera)
    Sophus::SE3d T_cw(R_eigen, t_eigen);

    std::vector<int> inlier_indices;
    inlier_indices.reserve(inliers.rows);
    for (int i = 0; i < inliers.rows; ++i) {
        inlier_indices.push_back(inliers.at<int>(i));
    }

    return {T_cw.inverse(), inlier_indices};
}

Sophus::SE3d motionBA(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    const Intrinsic& intrinsic,
    const Sophus::SE3d& T_init,
    int num_iterations,
    std::vector<bool>* inlier_mask) {

    const int n = static_cast<int>(points_3d.size());
    const double chi2_threshold = 5.991;

    // Work in T_cw space (camera-from-world)
    Sophus::SE3d T_cw = T_init.inverse();

    // Per-point outlier status (can be re-classified each pass)
    std::vector<bool> is_outlier(n, false);

    // 4 passes: passes 0-1 use Huber kernel, passes 2-3 do not
    for (int pass = 0; pass < 4; ++pass) {
        bool use_huber = (pass < 2);

        for (int iter = 0; iter < num_iterations; ++iter) {
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

            for (int i = 0; i < n; ++i) {
                if (is_outlier[i]) continue;

                Eigen::Vector3d p_w(points_3d[i].x, points_3d[i].y, points_3d[i].z);
                Eigen::Vector3d p_c = T_cw * p_w;

                double x = p_c.x(), y = p_c.y(), z = p_c.z();
                if (z <= 0) { is_outlier[i] = true; continue; }
                double z_inv = 1.0 / z;
                double z_inv2 = z_inv * z_inv;

                double u = intrinsic.fx * x * z_inv + intrinsic.cx;
                double v = intrinsic.fy * y * z_inv + intrinsic.cy;

                Eigen::Vector2d e(u - points_2d[i].x, v - points_2d[i].y);
                double chi2 = e.squaredNorm();

                // Huber weight
                double w = 1.0;
                if (use_huber) {
                    double e_norm = e.norm();
                    if (e_norm > std::sqrt(chi2_threshold)) {
                        w = std::sqrt(chi2_threshold) / e_norm;
                    }
                }

                // Jacobian of projection w.r.t. left perturbation on T_cw
                Eigen::Matrix<double, 2, 6> J;
                J(0, 0) = intrinsic.fx * z_inv;
                J(0, 1) = 0.0;
                J(0, 2) = -intrinsic.fx * x * z_inv2;
                J(0, 3) = -intrinsic.fx * x * y * z_inv2;
                J(0, 4) = intrinsic.fx * (1.0 + x * x * z_inv2);
                J(0, 5) = -intrinsic.fx * y * z_inv;

                J(1, 0) = 0.0;
                J(1, 1) = intrinsic.fy * z_inv;
                J(1, 2) = -intrinsic.fy * y * z_inv2;
                J(1, 3) = -intrinsic.fy * (1.0 + y * y * z_inv2);
                J(1, 4) = intrinsic.fy * x * y * z_inv2;
                J(1, 5) = intrinsic.fy * x * z_inv;

                H += w * J.transpose() * J;
                b += -w * J.transpose() * e;
            }

            Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
            T_cw = Sophus::SE3d::exp(dx) * T_cw;

            if (dx.norm() < 1e-6) break;
        }

        // After each pass: re-classify outliers based on chi2
        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d p_w(points_3d[i].x, points_3d[i].y, points_3d[i].z);
            Eigen::Vector3d p_c = T_cw * p_w;
            if (p_c.z() <= 0) { is_outlier[i] = true; continue; }

            double u = intrinsic.fx * (p_c.x() / p_c.z()) + intrinsic.cx;
            double v = intrinsic.fy * (p_c.y() / p_c.z()) + intrinsic.cy;
            double chi2 = (u - points_2d[i].x) * (u - points_2d[i].x)
                        + (v - points_2d[i].y) * (v - points_2d[i].y);

            is_outlier[i] = (chi2 > chi2_threshold);
        }
    }

    if (inlier_mask) {
        inlier_mask->resize(n);
        for (int i = 0; i < n; ++i)
            (*inlier_mask)[i] = !is_outlier[i];
    }

    return T_cw.inverse();
}
