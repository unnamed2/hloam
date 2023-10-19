#include "mapping.h"

#include <J.h>
#include <comm.h>
#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <result_of>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>

struct mapping_thread {
    struct calculate_val {
        synced_message msg;
        feature_frame frame;
    };

    synced_queue<calculate_val> q;
    std::thread thread;

    ros::Publisher pub_path;
    ros::Publisher pub_local_map;

    nav_msgs::Path path;
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

        path.header.stamp = time;

        geometry_msgs::PoseStamped pose;
        pose.header = path.header;
        pose.pose.position.x = transform(0, 3);
        pose.pose.position.y = transform(1, 3);
        pose.pose.position.z = transform(2, 3);

        pose.pose.orientation.x = qd.x();
        pose.pose.orientation.y = qd.y();
        pose.pose.orientation.z = qd.z();
        pose.pose.orientation.w = qd.w();

        path.poses.push_back(pose);
        pub_path.publish(path);
    }

private:
    void __mapping_thread(const std::string& save_path);
    static void __mapping_thread_entry(mapping_thread* self, const std::string& save_path);
};

template<typename Scalar>
static void remove_degenerate(Eigen::Matrix<Scalar, 6, 6>& ATA, Scalar threshold) {
    auto eigen = ATA.eigenvalues();
    bool is_degenerate = false;
    for(int i = 0; i < 6; i++) {
        if(eigen(i).real() < threshold) {
            is_degenerate = true;
            break;
        }
    }

    if(is_degenerate) {
        for(int i = 0; i < 6; i++) {
            ATA(i, i) += threshold;
        }
    }
}

Transform LM2(const feature_frame& this_features, const feature_frame& local_maps,
              float degenerate_threshold) {
    Transform initial = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    feature_adapter adap_velodyne(local_maps.velodyne_feature);
    feature_adapter adap_livox(local_maps.livox_feature);
    for(int i = 0; i < 30; i++) {
        auto N = Ab({ { this_features.velodyne_feature, adap_velodyne },
                      { this_features.livox_feature, adap_livox } },
                    initial);

        if(N.top == 0) {
            printf("No feature found!\r\n");
            return initial;
        }

        Eigen::Matrix<float, 6, 6> ATA = N.A.topRows(N.top).transpose() * N.A.topRows(N.top);

        if(i == 0) {
            remove_degenerate(ATA, degenerate_threshold);
        }

        Eigen::Matrix<float, 6, 1> ATb = N.A.topRows(N.top).transpose() * N.b.topRows(N.top);

        Eigen::Matrix<float, 6, 1> delta = ATA.householderQr().solve(ATb);

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
            return tr;
        }
    }

    return initial;
}

struct visual_odom_v2 {
    constexpr static size_t previous_frame_count = 20;
    feature_frame prev_frames[previous_frame_count];
    Eigen::Matrix4d prev_frame_location[previous_frame_count];
    size_t head = 0, counters = 0;

    float degenerate_threshold = 10.0f;

    bool use_livox = true;
    bool use_velodyne = true;

    loop_var loop;

    result_of<Eigen::Matrix4d, std::string>
        update_current_frame(const feature_frame& this_features) {
        if(counters == 0) {
            prev_frames[0] = std::move(this_features);
            prev_frame_location[0] = Eigen::Matrix4d::Identity();
            counters++;
            return ok(prev_frame_location[0]);
        }

        auto local_maps = local_map();
        if(use_velodyne && this_features.velodyne_feature.line_features->size() < 10 ||
           this_features.velodyne_feature.plane_features->size() < 100)
            return fail("velodyne not enough features");

        if(use_livox && this_features.livox_feature.plane_features->size() < 10 ||
           this_features.livox_feature.non_features->size() < 100)
            return fail("livox not enough features");

        Transform Tr = LM2(this_features, local_maps, degenerate_threshold);
        Eigen::Matrix4d this_frame_location = prev_frame_location[head] * to_eigen(Tr);
        // if transformation and rotation is too small, drop this frame

        if(std::abs(Tr.x) < 0.1 && std::abs(Tr.y) < 0.1 && std::abs(Tr.z) < 0.1 &&
           std::abs(Tr.roll) < 0.01 && std::abs(Tr.pitch) < 0.01 && std::abs(Tr.yaw) < 0.01) {
            return ok(this_frame_location);
        }

        head = (head + 1) % previous_frame_count;
        if(counters < previous_frame_count)
            counters++;
        prev_frames[head] = this_features;
        prev_frame_location[head] = this_frame_location;
        return ok(this_frame_location);
    }

    feature_frame local_map() {
        assert(counters > 0);

        feature_frame result;
        concat(result.velodyne_feature, prev_frames[head].velodyne_feature);

        concat(result.livox_feature, prev_frames[head].livox_feature);
        Eigen::Matrix4d transform = prev_frame_location[head].inverse();

        for(size_t i = 0; i < counters; i++) {
            Eigen::Matrix4d this_transform = transform * prev_frame_location[i];
            transform_cloud(prev_frames[i].velodyne_feature.line_features->points,
                            std::back_inserter(result.velodyne_feature.line_features->points),
                            this_transform);

            transform_cloud(prev_frames[i].velodyne_feature.plane_features->points,
                            std::back_inserter(result.velodyne_feature.plane_features->points),
                            this_transform);

            transform_cloud(prev_frames[i].livox_feature.plane_features->points,
                            std::back_inserter(result.livox_feature.plane_features->points),
                            this_transform);

            transform_cloud(prev_frames[i].livox_feature.non_features->points,
                            std::back_inserter(result.livox_feature.non_features->points),
                            this_transform);
        }

        result.velodyne_feature.line_features->width =
            result.velodyne_feature.line_features->points.size();

        result.velodyne_feature.plane_features->width =
            result.velodyne_feature.plane_features->points.size();

        result.livox_feature.plane_features->width =
            result.livox_feature.plane_features->points.size();

        result.livox_feature.non_features->width = result.livox_feature.non_features->points.size();
        return result;
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

    void loop_detection(const pcl::PointCloud<PointType>::Ptr& cloud, const feature_objects& frame,
                        const Eigen::Matrix4d& transform) {
        size_t result = loop.loop_detection(cloud, frame, transform);
        if(result == 0)
            return;

        for(size_t i = head;; i--) {
            prev_frame_location[i] = loop.btr(head - i + 1);
            if(i == 0)
                break;
        }

        for(size_t i = counters - 1; i > head; i--) {
            prev_frame_location[i] = loop.btr(counters - i + head + 2);
        }
    }
};

static Eigen::Matrix4d mapping_for_V2(const feature_frame& frame,
                                      visual_odom_v2& mapping_velodyne) {
    auto r = mapping_velodyne.update_current_frame(frame);
    if(!r.ok()) {
        ROS_INFO("Mapping failed: %s", r.error().c_str());
        return Eigen::Matrix4d::Identity();
    }
    return r.value();
}

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

void mapping_thread::__mapping_thread(const std::string& save_path) {

    visual_odom_v2 mapping_v2;
    mapping_v2.degenerate_threshold = degenerate_threshold;

    printf("Mapping thread started\r\n");
    while(true) {
        auto pq = q.acquire([this]() { return this->should_stop; });

        if(pq.empty())
            break;

        while(!pq.empty() && !this->should_stop) {
            auto M = mapping_for_V2(pq.front().frame, mapping_v2);
            mapping_v2.loop_detection(pq.front().msg.velodyne, pq.front().frame.velodyne_feature,
                                      M);

            pcl::PointCloud<PointType> final_cloud_velodyne, final_cloud_livox;

            Eigen::Matrix4d LX = M * livox_transform;
            pcl::transformPointCloud(*pq.front().msg.velodyne, final_cloud_velodyne, M);
            pcl::transformPointCloud(*pq.front().msg.livox, final_cloud_livox, LX);

            publish_delegate(final_cloud_velodyne, final_cloud_livox, pq.front().msg.time);
            publish_transform(M, pq.front().msg.time);
            pq.pop();
        }
    }

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

void mapping_thread::__mapping_thread_entry(mapping_thread* self, const std::string& save_path) {
    self->__mapping_thread(save_path);
}

mapping_thread::mapping_thread(ros::NodeHandle* nh) {
    std::string save_path;
    nh->param<std::string>("/tailor/mapping_save_path", save_path, "");
    printf("Mapping save path: %s\r\n", save_path.c_str());

    pub_path = nh->advertise<nav_msgs::Path>("/paths", 1000);
    pub_local_map = nh->advertise<sensor_msgs::PointCloud2>("/local_map", 1000);

    thread = std::thread(&mapping_thread::__mapping_thread_entry, this, save_path);

    feature_frame_delegate.append([this](const synced_message& msg, const feature_frame& frame) {
        q.push({ msg, frame });
    });

    path.header.frame_id = "map";

    std::vector<float> livox_cab;
    nh->param<std::vector<float>>("/tailor/livox_transform", livox_cab, { 0, 0, 0, 0, 0, 0 });

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

    nh->param<float>("/tailor/degenerate_threshold", degenerate_threshold, 10.0f);
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
