#include "loop.h"

#include "residual.h"

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

static gtsam::Pose3 p(LMTransform tr) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(tr.yaw, tr.pitch, tr.roll),
                        gtsam::Point3(tr.x, tr.y, tr.z));
}

static gtsam::Pose3 p(const Eigen::Matrix4d& tr) {
    return gtsam::Pose3(gtsam::Rot3(tr.block<3, 3>(0, 0)),
                        gtsam::Point3(tr(0, 3), tr(1, 3), tr(2, 3)));
}

static Eigen::Matrix4d to_eigen(const gtsam::Pose3& tr) {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m.block<3, 3>(0, 0) = tr.rotation().matrix();
    m(0, 3) = tr.x();
    m(1, 3) = tr.y();
    m(2, 3) = tr.z();
    return m;
}

static void dump_features(const feature_objects& f, const char* filename) {
    pcl::PointCloud<XYZIRT>::Ptr cloud(new pcl::PointCloud<XYZIRT>);
    concat(cloud, f.line_features);
    concat(cloud, f.plane_features);
    concat(cloud, f.non_features);

    pcl::io::savePCDFileBinary(filename, *cloud);
}

loop_var::loop_var(): loop_counter(loop_reset) {
}

size_t loop_var::loop_detection(const pcl::PointCloud<XYZIRT>::Ptr& cloud,
                                const feature_objects& frame, const Eigen::Matrix4d& transform) {
    sc_manager.makeAndSaveScancontextAndKeys(*cloud);
    frames.push_back({ frame, transform });

    if(loop_counter > 0) {
        loop_counter--;
        return 0;
    }

    loop_counter = loop_reset;

    auto [id, yaw] = sc_manager.detectLoopClosureID();

    if(id == -1) {
        return 0;
    }

    // build local map;
    int start_index = id - 0;

    if(start_index < 0) {
        start_index = 0;
    }

    int end_index = id + 0;
    if(end_index >= frames.size()) {
        end_index = frames.size() - 1;
    }

    feature_objects local_map;
    feature_objects transformed;

    Eigen::Matrix4d tr = frames[id].transform.inverse();

    for(int i = start_index; i <= end_index; i++) {
        auto& frame = frames[i];
        Eigen::Matrix4d this_tr = tr * frame.transform;
        transform_cloud(frame.velodyne_features, transformed, this_tr);
        concat(local_map, transformed);
    }
    if(yaw > M_PI)
        yaw -= M_PI * 2.0;

    LMTransform initial_guess = { 0, 0, 0, 0, 0, yaw };

    float loss = 0.0f;
    auto final_tr = LM(frame, local_map, initial_guess, &loss);

    gtsam::Pose3 from = p(frames[id].transform);
    gtsam::Pose3 to = p(frames[id].transform * to_eigen(final_tr));
    printf("loss: %f\n", loss);
    if(loss > 0.05) {
        return 0;
    }
    loop_result r = {
        (size_t)id,
        frames.size() - 1,
        from.between(to),
    };
    loop.push_back(r);

    optimization(id);
    printf("loop detected: %d %zd\n", id, frames.size() - 1);
    return id;
}

void loop_var::optimization(size_t from_id) {

    gtsam::ISAM2 isam;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    gtsam::noiseModel::Diagonal::shared_ptr fixed_noise =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.0));

    gtsam::noiseModel::Diagonal::shared_ptr prior_noise =
        gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector6::Constant(0.1)));

    gtsam::Pose3 X0 = p(frames[from_id].transform);
    gtsam::PriorFactor F0 = gtsam::PriorFactor<gtsam::Pose3>(from_id, X0, fixed_noise);
    graph.push_back(F0);
    initial.insert(from_id, X0);

    for(size_t i = from_id + 1; i < frames.size(); i++) {
        auto& frame = frames[i];
        gtsam::Pose3 Xi = p(frame.transform);

        gtsam::BetweenFactor<gtsam::Pose3> odometry_factor(i - 1, i, X0.between(Xi), prior_noise);

        graph.push_back(odometry_factor);
        initial.insert(i, Xi);
        X0 = Xi;
    }

    for(auto&& [from, to, tr]: loop) {
        if(from < from_id || to < from_id)
            continue;
        gtsam::BetweenFactor<gtsam::Pose3> loop_factor(from, to, tr, prior_noise);
        graph.push_back(loop_factor);
    }

    isam.update(graph, initial);
    auto result = isam.calculateEstimate();

    for(auto&& value: result) {
        frames[value.key].transform = to_eigen(value.value.cast<gtsam::Pose3>());
    }
}
