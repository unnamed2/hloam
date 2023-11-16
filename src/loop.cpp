#include "loop.h"

#include "residual.h"

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/registration/icp.h>

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

static void downsample_surf2(const pcl::PointCloud<PointType>::Ptr& surface_points,
                             pcl::PointCloud<PointType>& downsampled_surface_points) {
    static pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surface_points);
    downSizeFilter.setLeafSize(0.4, 0.4, 0.4);
    downSizeFilter.filter(downsampled_surface_points);
}

inline feature_objects downsample(const feature_objects& input) {
    feature_objects result;
    if(input.line_features != nullptr) {
        result.line_features.reset(new pcl::PointCloud<PointType>());
        downsample_surf2(input.line_features, *result.line_features);
    }
    if(input.plane_features != nullptr) {
        result.plane_features.reset(new pcl::PointCloud<PointType>());
        downsample_surf2(input.plane_features, *result.plane_features);
    }
    if(input.non_features != nullptr) {
        result.non_features.reset(new pcl::PointCloud<PointType>());
        downsample_surf2(input.non_features, *result.non_features);
    }
    return result;
}

inline feature_frame downsample(const feature_frame& input) {
    feature_frame result;
    result.velodyne_feature = downsample(input.velodyne_feature);
    result.livox_feature = downsample(input.livox_feature);
    return result;
}

static void dump_features(const feature_objects& f, const char* filename) {
    char filename2[256];
    pcl::PointCloud<XYZIRT>::Ptr cloud(new pcl::PointCloud<XYZIRT>);
    if(f.line_features != nullptr) {
        sprintf(filename2, "%s_line.pcd", filename);
        pcl::io::savePCDFileBinary(filename2, *f.line_features);
    }

    if(f.plane_features != nullptr) {
        sprintf(filename2, "%s_plane.pcd", filename);
        pcl::io::savePCDFileBinary(filename2, *f.plane_features);
    }

    if(f.non_features != nullptr) {
        sprintf(filename2, "%s_non.pcd", filename);
        pcl::io::savePCDFileBinary(filename2, *f.non_features);
    }
}

loop_var::loop_var(): loop_counter(loop_reset) {
}

size_t loop_var::loop_detection(const pcl::PointCloud<XYZIRT>::Ptr& cloud,
                                const feature_objects& frame, const Eigen::Matrix4d& transform) {
    sc_manager.makeAndSaveScancontextAndKeys(*cloud);
    frames.push_back({ cloud, transform });

    if(loop_counter > 0) {
        loop_counter--;
        return NO_LOOP;
    }

    loop_counter = loop_reset;

    auto [id, yaw] = sc_manager.detectLoopClosureID();

    if(id == -1) {
        return NO_LOOP;
    }

    // build local map;
    int start_index = id - 10;

    if(start_index < 0) {
        start_index = 0;
    }

    int end_index = start_index + 20;
    if(end_index >= frames.size()) {
        end_index = frames.size() - 1;
    }

    pcl::PointCloud<XYZIRT>::Ptr local_map(new pcl::PointCloud<XYZIRT>);
    pcl::PointCloud<XYZIRT>::Ptr transformed(new pcl::PointCloud<XYZIRT>);

    Eigen::Matrix4d tr = frames[id].transform.inverse();

    for(int i = start_index; i <= end_index; i++) {
        auto& frame = frames[i];
        Eigen::Matrix4d this_tr = tr * frame.transform;
        pcl::transformPointCloud(*frame.velodyne_cloud, *transformed, this_tr);
        *local_map += *transformed;
    }

    downsample_surf2(local_map, *local_map);
    downsample_surf2(cloud, *transformed);

    if(yaw > M_PI)
        yaw -= M_PI * 2.0;

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setInputSource(transformed);
    icp.setInputTarget(local_map);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setMaxCorrespondenceDistance(0.5);

    Eigen::Matrix4f init_tr = Eigen::Matrix4f::Identity();
    init_tr.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(-yaw, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    pcl::PointCloud<PointType> final;
    icp.align(final, init_tr);

    Eigen::Matrix4d final_tr = icp.getFinalTransformation().cast<double>();
    float loss = icp.getFitnessScore();

    gtsam::Pose3 from = p(frames[id].transform);
    gtsam::Pose3 to = p(frames[id].transform * final_tr);
    printf("loss: %f\n", loss);
    if(loss > 0.5f) {
        return NO_LOOP;
    }
    loop_result r = {
        (size_t)id,
        frames.size() - 1,
        from.between(to),
    };
    loop.push_back(r);

    if(id < min_constriant_node)
        min_constriant_node = id;

    optimization(min_constriant_node);
    printf("loop detected: %d %zd\n", id, frames.size() - 1);
    return min_constriant_node;
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

void loop_var::pop_back() {
    if(!loop.empty() && loop.back().target_frame_id == frames.size() - 1)
        loop.pop_back();

    sc_manager.pop_back();
    frames.pop_back();
}

Eigen::Matrix4d solve_GTSAM(const Eigen::Matrix4d& M1, const Eigen::Matrix4d& M2, float loss_M1,
                            float loss_M2) {
    gtsam::ISAM2 isam;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    gtsam::noiseModel::Diagonal::shared_ptr fixed_noise =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.0));
    gtsam::noiseModel::Diagonal::shared_ptr loss_M1_noise =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(loss_M1));

    gtsam::noiseModel::Diagonal::shared_ptr loss_M2_noise =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(loss_M2));

    gtsam::Pose3 X0 = gtsam::Pose3::Identity();
    gtsam::PriorFactor F0 = gtsam::PriorFactor<gtsam::Pose3>(0, X0, fixed_noise);
    graph.push_back(F0);

    gtsam::Pose3 X1 = p(M1);
    gtsam::Pose3 X2 = p(M2);

    gtsam::BetweenFactor B0 =
        gtsam::BetweenFactor<gtsam::Pose3>(0, 1, X0.between(X1), loss_M1_noise);
    gtsam::BetweenFactor B1 =
        gtsam::BetweenFactor<gtsam::Pose3>(0, 1, X0.between(X2), loss_M2_noise);

    graph.push_back(B0);
    graph.push_back(B1);

    initial.insert(0, X0);
    initial.insert(1, loss_M1 < loss_M2 ? X1 : X2);

    isam.update(graph, initial);

    auto result = isam.calculateEstimate();
    return to_eigen(result.at<gtsam::Pose3>(1));
}

#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/registration/impl/icp.hpp>
#include <pcl/search/impl/kdtree.hpp>