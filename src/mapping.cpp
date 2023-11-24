#include "comm.h"
#include "loop.h"
#include "residual.h"

#include <algorithm>
#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <result_of>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

static void dump_feature_frame(const feature_frame& f, const char* tag) {
    const char* base_ptr = "/home/jlurobot/桌面/LM测试/";
    char filename[256];

    if(f.velodyne_feature.line_features != nullptr) {
        sprintf(filename, "%s/%s_vl.pcd", base_ptr, tag);
        pcl::io::savePCDFileBinary(filename, *f.velodyne_feature.line_features);
    }

    if(f.velodyne_feature.plane_features != nullptr) {
        sprintf(filename, "%s/%s_vp.pcd", base_ptr, tag);
        pcl::io::savePCDFileBinary(filename, *f.velodyne_feature.plane_features);
    }

    if(f.livox_feature.plane_features != nullptr) {
        sprintf(filename, "%s/%s_lp.pcd", base_ptr, tag);
        pcl::io::savePCDFileBinary(filename, *f.livox_feature.plane_features);
    }

    if(f.livox_feature.non_features != nullptr) {
        sprintf(filename, "%s/%s_ln.pcd", base_ptr, tag);
        pcl::io::savePCDFileBinary(filename, *f.livox_feature.non_features);
    }
}

static void remove_degenerate(Eigen::Matrix<double, 6, 6>& ATA, double threshold) {
    auto eigen = ATA.eigenvalues();
    bool is_degenerate = false;
    for(int i = 0; i < 6; i++) {
        if(eigen(i).real() < threshold) {
            ROS_INFO("degenerate: %f", eigen(i).real());
            is_degenerate = true;
            break;
        }
    }

    if(is_degenerate) {
        for(int i = 0; i < 6; i++) {
            ATA(i, i) += 0.5;
        }
    }
}

static void downsample_surf2(const pcl::PointCloud<PointType>::Ptr& surface_points,
                             pcl::PointCloud<PointType>::Ptr& downsampled_surface_points) {
    static pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surface_points);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(*downsampled_surface_points);
}

inline feature_objects downsample(const feature_objects& input) {
    feature_objects result;
    if(input.line_features != nullptr) {
        result.line_features.reset(new pcl::PointCloud<PointType>());
        downsample_surf2(input.line_features, result.line_features);
    }
    if(input.plane_features != nullptr) {
        result.plane_features.reset(new pcl::PointCloud<PointType>());
        downsample_surf2(input.plane_features, result.plane_features);
    }
    if(input.non_features != nullptr) {
        result.non_features.reset(new pcl::PointCloud<PointType>());
        downsample_surf2(input.non_features, result.non_features);
    }
    return result;
}

inline feature_frame downsample(const feature_frame& input) {
    feature_frame result;
    result.velodyne_feature = downsample(input.velodyne_feature);
    result.livox_feature = downsample(input.livox_feature);
    return result;
}

Transform LM2(const feature_frame& this_features, const feature_frame& local_maps,
              float degenerate_threshold, Transform initial = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
              float* loss = nullptr) {

    feature_adapter adap_velodyne(local_maps.velodyne_feature);
    feature_adapter adap_livox(local_maps.livox_feature);

    if(loss != nullptr) {
        *loss = 0.0f;
    }

    for(int i = 0; i < 30; i++) {

        float __loss = 0.0f;
        auto N = Ab({ { this_features.velodyne_feature, adap_velodyne },
                      { this_features.livox_feature, adap_livox } },
                    initial, &__loss);
        if(loss != nullptr) {
            *loss = __loss;
        }

        if(N.top < 5) {
            if(loss != nullptr) {
                *loss = 10000.0f;
            }
            return initial;
        }

        Eigen::Matrix<double, 6, 6> ATA = N.A.topRows(N.top).transpose() * N.A.topRows(N.top);

        if(i == 0) {
            remove_degenerate(ATA, degenerate_threshold);
        }

        Eigen::Matrix<double, 6, 1> ATb = N.A.topRows(N.top).transpose() * N.b.topRows(N.top);

        Eigen::Matrix<double, 6, 1> delta = ATA.householderQr().solve(ATb);

        Transform tr = initial;
        tr.x += delta(0);
        tr.y += delta(1);
        tr.z += delta(2);
        tr.roll += delta(3);
        tr.pitch += delta(4);
        tr.yaw += delta(5);

        double delta_xyz = p2(delta(0)) + p2(delta(1)) + p2(delta(2));
        double delta_rpy = p2(delta(3)) + p2(delta(4)) + p2(delta(5));

        initial = tr;

        if(delta_xyz < 1e-7 && delta_rpy < 1e-7) {
            tr.pitch = tr.roll = 0.0;
            return tr;
        }
    }

    return initial;
}

static bool feature_ok(const feature_objects& object) {
    if(object.line_features != nullptr && object.line_features->size() < 10)
        return false;

    if(object.plane_features != nullptr && object.plane_features->size() < 100)
        return false;

    if(object.non_features != nullptr && object.non_features->size() < 100)
        return false;

    return true;
}

struct local_map {
    constexpr static size_t previous_frame_count = 10;
    feature_frame prev_frames[previous_frame_count];
    Eigen::Matrix4d prev_frame_location[previous_frame_count];

    size_t head = previous_frame_count - 1, counters = 0;

    feature_frame local_map;
    bool local_map_dirty = true;
    const feature_frame& get_local_map() {
        if(local_map_dirty) {
            local_map = update_local_map();
            local_map_dirty = false;
        }
        return local_map;
    }

    feature_frame update_local_map() const {
        assert(counters > 0);

        feature_frame result;
        concat(result.velodyne_feature, prev_frames[head].velodyne_feature);

        concat(result.livox_feature, prev_frames[head].livox_feature);
        Eigen::Matrix4d transform = prev_frame_location[head].inverse();

        for(size_t i = 0; i < counters; i++) {
            Eigen::Matrix4d this_transform = transform * prev_frame_location[i];
            if(prev_frames[i].velodyne_feature.line_features) {
                transform_cloud(prev_frames[i].velodyne_feature.line_features->points,
                                std::back_inserter(result.velodyne_feature.line_features->points),
                                this_transform);
            }

            if(prev_frames[i].velodyne_feature.plane_features) {
                transform_cloud(prev_frames[i].velodyne_feature.plane_features->points,
                                std::back_inserter(result.velodyne_feature.plane_features->points),
                                this_transform);
            }

            if(prev_frames[i].livox_feature.plane_features) {
                transform_cloud(prev_frames[i].livox_feature.plane_features->points,
                                std::back_inserter(result.livox_feature.plane_features->points),
                                this_transform);
            }

            if(prev_frames[i].livox_feature.non_features) {
                transform_cloud(prev_frames[i].livox_feature.non_features->points,
                                std::back_inserter(result.livox_feature.non_features->points),
                                this_transform);
            }
        }
        if(result.velodyne_feature.line_features)
            result.velodyne_feature.line_features->width =
                result.velodyne_feature.line_features->points.size();
        if(result.velodyne_feature.plane_features)
            result.velodyne_feature.plane_features->width =
                result.velodyne_feature.plane_features->points.size();

        if(result.livox_feature.plane_features)
            result.livox_feature.plane_features->width =
                result.livox_feature.plane_features->points.size();

        if(result.livox_feature.non_features)
            result.livox_feature.non_features->width =
                result.livox_feature.non_features->points.size();
        return downsample(result);
    }

    template<typename _Range, typename OutputIter, typename MatrixType>
    static void transform_cloud(_Range&& range, OutputIter iter, MatrixType&& matrix) {

        for(auto&& point: range) {
            Eigen::Vector4d p(point.x, point.y, point.z, 1);

            auto transformed = matrix * p;
            auto new_point = point;
            new_point.x = transformed.x();
            new_point.y = transformed.y();
            new_point.z = transformed.z();
            *iter++ = new_point;
        }
    }

    void push(const feature_frame& frame, const Eigen::Matrix4d& transform) {

        head = (head + 1) % previous_frame_count;

        if(counters < previous_frame_count)
            counters++;
        prev_frames[head] = frame;
        prev_frame_location[head] = transform;
        local_map_dirty = true;
    }

    bool empty() const {
        return counters == 0;
    }

    size_t size() const {
        return counters;
    }

    const Eigen::Matrix4d& tr() const {
        assert(counters > 0);
        return prev_frame_location[head];
    }

    void set(size_t back_index, const Eigen::Matrix4d& transform) {
        assert(back_index <= counters);
        if(back_index <= head + 1) {
            prev_frame_location[head + 1 - back_index] = transform;
        } else {
            prev_frame_location[counters + head + 1 - back_index] = transform;
        }
    }
};

geometry_msgs::Pose to_ros_pose(const Eigen::Matrix4d& transform) {
    geometry_msgs::Pose pose;
    pose.position.x = transform(0, 3);
    pose.position.y = transform(1, 3);
    pose.position.z = transform(2, 3);

    Eigen::Matrix3d rot = transform.block<3, 3>(0, 0);
    Eigen::Quaterniond qd(rot);

    pose.orientation.x = qd.x();
    pose.orientation.y = qd.y();
    pose.orientation.z = qd.z();
    pose.orientation.w = qd.w();

    return pose;
}

struct visual_odom_v2_config {
    int method = 0;
    float degenerate_threshold = 10.0f;

    double key_frame_distance_x = 0.5;
    double key_frame_distance_y = 0.5;
    double key_frame_distance_z = 0.1;

    double key_frame_distance_roll = 0.02;
    double key_frame_distance_pitch = 0.02;
    double key_frame_distance_yaw = 0.02;

    double loop_loss = 0.05f;

    int loop_reset = 5;
    int loop_initial_load = 100;

    bool enable_loop = true;
};

visual_odom_v2_config get_odom_config(ros::NodeHandle* handle) {
    visual_odom_v2_config config;
    handle->param<float>("/hloam/LM/degenerate_threshold", config.degenerate_threshold, 10.0f);
    handle->param<int>("/hloam/LM/method", config.method, 0);

    handle->param<double>("/hloam/key_frame/x", config.key_frame_distance_x, 0.5);
    handle->param<double>("/hloam/key_frame/y", config.key_frame_distance_y, 0.5);
    handle->param<double>("/hloam/key_frame/z", config.key_frame_distance_z, 0.1);

    handle->param<double>("/hloam/key_frame/roll", config.key_frame_distance_roll, 0.02);
    handle->param<double>("/hloam/key_frame/pitch", config.key_frame_distance_pitch, 0.02);
    handle->param<double>("/hloam/key_frame/yaw", config.key_frame_distance_yaw, 0.02);

    handle->param<double>("/hloam/loop/max_loss", config.loop_loss, 0.05);

    handle->param<int>("/hloam/loop/reset", config.loop_reset, 5);
    handle->param<int>("/hloam/loop/initial_load", config.loop_initial_load, 100);

    handle->param<bool>("/hloam/loop/enable", config.enable_loop, true);

    return config;
}

struct visual_odom_v2 {
    local_map local_maps;

    Transform next_initial_guess;
    Eigen::Matrix4d prev_transform = Eigen::Matrix4d::Identity();

    float degenerate_threshold = 10.0f;

    loop_var loop;

    visual_odom_v2_config config;

    visual_odom_v2(ros::NodeHandle* nh) {

        config = get_odom_config(nh);

        loop.loop_counter = config.loop_initial_load;
        loop.loop_reset = config.loop_reset;
        loop.loop_max_loss = config.loop_loss;

        final_path.header.frame_id = "map";
        loop_markers.header.frame_id = "map";

        loop_markers.type = visualization_msgs::Marker::LINE_LIST;
        loop_markers.action = visualization_msgs::Marker::ADD;
        loop_markers.ns = "loop_marker";
        loop_markers.id = 0;
        loop_markers.pose.orientation.w = 1.0;
        loop_markers.color.r = 1.0;
        loop_markers.color.g = 1.0;
        loop_markers.color.b = 0.0;
        loop_markers.color.a = 1.0;

        loop_markers.scale.x = 0.1;
        loop_markers.scale.y = 0.1;
        loop_markers.scale.z = 0.1;

        memset(&next_initial_guess, 0, sizeof(next_initial_guess));
    }

    result_of<Transform, std::string> update_current_frame_LM2(const feature_frame& this_features,
                                                               const feature_frame& M) {
        constexpr float loss_threshold = 0.03f;
        float loss = 0.0f;
        Transform Tr = LM2(this_features, M, degenerate_threshold, next_initial_guess, &loss);
        /*if(loss > loss_threshold) {
            // reset initial guess and try again
            memset(&next_initial_guess, 0, sizeof(next_initial_guess));
            Tr = LM2(this_features, M, degenerate_threshold, next_initial_guess, &loss);
        }

        if(loss > loss_threshold) {
            char buffer[256];
            sprintf(buffer, "LM loss too large, %f", loss);
            return fail(buffer);
        }*/

        next_initial_guess = Tr;
        return ok(Tr);
    }

    result_of<Transform, std::string> update_current_frame_GTSAM(const feature_frame& this_features,
                                                                 const feature_frame& M) {
        if(this_features.velodyne_feature.plane_features == nullptr ||
           this_features.livox_feature.plane_features == nullptr) {
            ROS_WARN_ONCE("GTSAM-Method not available, using LM2-Method");
            return update_current_frame_LM2(this_features, M);
        }

        ROS_INFO_ONCE("GTSAM-Method enabled");

        float loss_M1 = 0.0f, loss_M2 = 0.0f;
        Transform tr_livox =
            LM(this_features.livox_feature, M.livox_feature, next_initial_guess, &loss_M1);
        Transform tr_v =
            LM(this_features.velodyne_feature, M.velodyne_feature, next_initial_guess, &loss_M2);

        if(loss_M1 > 1.0f && loss_M2 < 1.0f) {
            next_initial_guess = tr_v;
            return ok(tr_v);
        } else if(loss_M1 < 1.0f && loss_M2 > 1.0f) {
            next_initial_guess = tr_livox;
            return ok(tr_livox);
        } else if(loss_M1 < 1.0f && loss_M2 < 1.0f) {
            auto M = solve_GTSAM(to_eigen(tr_livox), to_eigen(tr_v), loss_M1, loss_M2);
            next_initial_guess = from_eigen(M);
            return ok(next_initial_guess);
        }
        return fail("LM loss too large");
    }

    result_of<Transform, std::string> update_current_frame(const feature_frame& this_features) {

        if(!feature_ok(this_features.velodyne_feature))
            return fail("velodyne not enough features");

        if(!feature_ok(this_features.livox_feature))
            return fail("livox not enough features");

        auto f_ds = downsample(this_features);

        if(local_maps.empty()) {
            Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
            local_maps.push(this_features, identity);
            return ok(Transform{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
        }

        auto M = local_maps.get_local_map();

        static size_t frame_id = 0;
        if(frame_id == 200) {
            dump_feature_frame(this_features, "T");
            dump_feature_frame(M, "M");
        }

        frame_id++;

        if(config.method == 0)
            return update_current_frame_LM2(f_ds, M);
        else
            return update_current_frame_GTSAM(f_ds, M);
    }

    Eigen::Matrix4d loop_detection(const pcl::PointCloud<PointType>::Ptr& cloud,
                                   const feature_objects& frame, const Eigen::Matrix4d& transform,
                                   bool* has_loop) {
        size_t result = loop.loop_detection(cloud, frame, transform);
        if(result == NO_LOOP) {
            if(has_loop != nullptr)
                *has_loop = false;
            return transform;
        }

        if(has_loop != nullptr)
            *has_loop = true;

        for(size_t i = 1; i <= local_maps.size(); i++) {
            local_maps.set(i, loop.btr(i));
        }

        for(size_t i = result; i < final_path.poses.size(); i++) {
            auto pose = to_ros_pose(loop.tr(i));
            final_path.poses[i].pose = pose;
        }

        loop_markers.points.clear();
        for(auto&& r: loop.loop) {
            geometry_msgs::Point p1, p2;
            p1.x = loop.tr(r.source_frame_id)(0, 3);
            p1.y = loop.tr(r.source_frame_id)(1, 3);
            p1.z = loop.tr(r.source_frame_id)(2, 3);

            p2.x = loop.tr(r.target_frame_id)(0, 3);
            p2.y = loop.tr(r.target_frame_id)(1, 3);
            p2.z = loop.tr(r.target_frame_id)(2, 3);

            loop_markers.points.push_back(p1);
            loop_markers.points.push_back(p2);
        }

        return loop.btr(1);
    }

    visualization_msgs::Marker loop_markers;
    nav_msgs::Path final_path;

    std::optional<Eigen::Matrix4d> mapping(const pcl::PointCloud<PointType>::Ptr& velodyne_cloud,
                                           const feature_frame& frame, ros::Time time) {

        auto Mr = update_current_frame(frame);
        if(!Mr.ok()) {
            ROS_INFO("Frame dropped : %s", Mr.error().c_str());
            return std::nullopt;
        }

        auto Tr = Mr.value();
        Eigen::Matrix4d M = local_maps.tr() * to_eigen(Tr);
        Eigen::Matrix4d X = prev_transform.inverse() * M;
        prev_transform = M;

        bool has_loop = false;
        if(config.enable_loop) {
            M = loop_detection(velodyne_cloud, frame.velodyne_feature, M, &has_loop);
        }

        // if transformation and rotation is too small, drop this frame
        if(!has_loop && !local_maps.empty() && std::abs(Tr.x) < config.key_frame_distance_x &&
           std::abs(Tr.y) < config.key_frame_distance_y &&
           std::abs(Tr.z) < config.key_frame_distance_z &&
           std::abs(Tr.roll) < config.key_frame_distance_roll &&
           std::abs(Tr.pitch) < config.key_frame_distance_pitch &&
           std::abs(Tr.yaw) < config.key_frame_distance_yaw) {
            loop.pop_back();
            return M;
        }

        Transform M_tr = from_eigen(M);
        ROS_INFO("Mapping: %lf %lf %lf %lf %lf %lf", M_tr.x, M_tr.y, M_tr.z, M_tr.roll, M_tr.pitch,
                 M_tr.yaw);
        next_initial_guess = from_eigen(X);

        local_maps.push(frame, M);

        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp = time;
        pose.pose = to_ros_pose(M);

        final_path.poses.push_back(pose);
        final_path.header.stamp = time;

        loop_markers.header.stamp = time;

        return M;
    }
};

struct calculate_val {
    synced_message msg;
    feature_frame frame;
};

static void save_traces(const nav_msgs::Path& traces, std::string save_path) {
    char filename[256];
    sprintf(filename, "%s/%ld.txt", save_path.c_str(), time(nullptr));
    FILE* fp = fopen(filename, "w");
    for(auto&& tr: traces.poses) {
        // TUM
        fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf\r\n", tr.header.stamp.toSec(),
                tr.pose.position.x, tr.pose.position.y, tr.pose.position.z, tr.pose.orientation.x,
                tr.pose.orientation.y, tr.pose.orientation.z, tr.pose.orientation.w);
    }
    fclose(fp);
}

struct mapping_thread {
    struct calculate_val {
        synced_message msg;
        feature_frame frame;
    };

    synced_queue<calculate_val> q;
    std::thread thread;

    ros::Publisher pub_path;
    ros::Publisher pub_local_map;
    ros::Publisher pub_loop_marker;
    ros::Publisher pub_velodyne;
    ros::Publisher pub_livox;

    tf::TransformBroadcaster tf_broadcaster;

    Eigen::Matrix4d livox_transform;

    volatile bool should_stop = false;

    float degenerate_threshold;

    mapping_thread(ros::NodeHandle* nh);
    ~mapping_thread();

    void publish_transform(const Eigen::Matrix4d& transform, const ros::Time& time) {
        tf::Transform tf;
        tf.setOrigin(tf::Vector3(transform(0, 3), transform(1, 3), transform(2, 3)));

        Eigen::Matrix3d rot = transform.block<3, 3>(0, 0);
        Eigen::Quaterniond qd(rot);

        tf.setRotation(tf::Quaternion(qd.x(), qd.y(), qd.z(), qd.w()));
        tf_broadcaster.sendTransform(tf::StampedTransform(tf, time, "map", "velodyne16"));
    }

    void publish_map(const pcl::PointCloud<XYZIRT>& velodyne, const pcl::PointCloud<XYZIRT>& livox,
                     const ros::Time& time) {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(velodyne, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = time;
        pub_velodyne.publish(msg);

        pcl::toROSMsg(livox, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = time;
        pub_livox.publish(msg);
    }

private:
    void __mapping_thread(ros::NodeHandle* nh);
    static void __mapping_thread_entry(mapping_thread* self, ros::NodeHandle* nh);
};

void mapping_thread::__mapping_thread(ros::NodeHandle* nh) {
    visual_odom_v2 mapping_v2(nh);

    std::string save_path;
    nh->param<std::string>("/hloam/mapping_save_path", save_path, "");
    printf("Mapping save path: %s\r\n", save_path.c_str());

    mapping_v2.degenerate_threshold = degenerate_threshold;

    printf("Mapping thread started\r\n");
    while(true) {
        auto pq = q.acquire([this]() { return this->should_stop; });

        if(pq.empty())
            break;

        while(!pq.empty() && !this->should_stop) {
            auto& s = pq.front().msg;
            auto Mr = mapping_v2.mapping(s.velodyne, pq.front().frame, s.time);
            if(!Mr.has_value()) {
                pq.pop();
                continue;
            }

            auto M = Mr.value();
            pcl::PointCloud<PointType> final_cloud_velodyne, final_cloud_livox;

            Eigen::Matrix4d LX = M * livox_transform;
            pcl::transformPointCloud(*s.velodyne, final_cloud_velodyne, M);
            pcl::transformPointCloud(*s.livox, final_cloud_livox, LX);

            publish_map(final_cloud_velodyne, final_cloud_livox, s.time);
            publish_transform(M, s.time);

            pub_path.publish(mapping_v2.final_path);

            if(!mapping_v2.loop_markers.points.empty())
                pub_loop_marker.publish(mapping_v2.loop_markers);

            pq.pop();
        }
    }

    auto path = mapping_v2.final_path;
    if(!save_path.empty()) {
        if(path.poses.empty()) {
            printf("No trace to save!\r\n");
        } else {
            save_traces(path, save_path);
            printf("Saved %zd traces to %s\r\n", path.poses.size(), save_path.c_str());
        }
    }

    printf("Mapping thread stopped\r\n");
}

void mapping_thread::__mapping_thread_entry(mapping_thread* self, ros::NodeHandle* nh) {
    self->__mapping_thread(nh);
}

mapping_thread::mapping_thread(ros::NodeHandle* nh) {

    pub_path = nh->advertise<nav_msgs::Path>("/paths", 1000);
    pub_local_map = nh->advertise<sensor_msgs::PointCloud2>("/local_map", 1000);
    pub_loop_marker = nh->advertise<visualization_msgs::Marker>("/loop_marker", 1000);
    pub_velodyne = nh->advertise<sensor_msgs::PointCloud2>("/g_velodyne", 1000);
    pub_livox = nh->advertise<sensor_msgs::PointCloud2>("/g_livox", 1000);

    thread = std::thread(&mapping_thread::__mapping_thread_entry, this, nh);

    feature_frame_delegate.append([this](const synced_message& msg, const feature_frame& frame) {
        q.push({ msg, frame });
    });

    std::vector<float> livox_cab;
    nh->param<std::vector<float>>("/hloam/livox_transform", livox_cab, { 0, 0, 0, 0, 0, 0 });

    if(livox_cab.size() != 6) {
        ROS_FATAL("livox_transform must have 6 elements, %zd got", livox_cab.size());
    }

    Transform tr;
    tr.x = livox_cab[0];
    tr.y = livox_cab[1];
    tr.z = livox_cab[2];
    tr.roll = livox_cab[3];
    tr.pitch = livox_cab[4];
    tr.yaw = livox_cab[5];

    ROS_INFO("livox_transform: %f %f %f %f %f %f", tr.x, tr.y, tr.z, tr.roll, tr.pitch, tr.yaw);

    livox_transform = to_eigen(tr).inverse();

    nh->param<float>("/hloam/degenerate_threshold", degenerate_threshold, 10.0f);
    if(degenerate_threshold < 5.0f) {
        ROS_WARN("degenerate_threshold is too small, %f", degenerate_threshold);
    }
}

mapping_thread::~mapping_thread() {
    should_stop = true;
    q.notify();
    thread.join();
}

std::shared_ptr<mapping_thread> create_mapping_thread(ros::NodeHandle* nh) {
    return std::make_shared<mapping_thread>(nh);
}

#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/impl/pcl_base.hpp>
