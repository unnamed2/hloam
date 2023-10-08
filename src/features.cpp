#include <comm.h>
#include <condition_variable>
#include <mutex>
#include <pcl/common/transforms.h>
#include <queue>
#include <thread>

struct feature_thread {
    synced_queue<synced_message> q;
    std::thread thread;

    volatile bool should_stop = false;

    feature_thread(ros::NodeHandle* nh);
    ~feature_thread();

private:
    Eigen::Matrix4d livox_transform;
    void __features_thread();
    static void __features_thread_entry(feature_thread* self);
};

void feature_thread::__features_thread() {
    printf("Features thread started\r\n");
    while(true) {
        auto pq = q.acquire([this]() { return this->should_stop; });

        if(pq.empty()) {
            break;
        }

        for(; !pq.empty() && !this->should_stop; pq.pop()) {
            feature_frame frame;
            feature_livox(pq.front().livox, frame.livox_feature);
            if(frame.livox_feature.plane_features->empty() ||
               frame.livox_feature.non_features->empty()) {
                printf("livox feature empty!\r\n");
                continue;
            }
            feature_velodyne(pq.front().velodyne, frame.velodyne_feature);
            if(frame.velodyne_feature.line_features->size() < 20 ||
               frame.velodyne_feature.plane_features->size() < 100) {
                printf("velodyne feature not enough!\r\n");
                continue;
            }

            pcl::transformPointCloud(*frame.livox_feature.plane_features,
                                     *frame.livox_feature.plane_features, livox_transform);

            pcl::transformPointCloud(*frame.livox_feature.non_features,
                                     *frame.livox_feature.non_features, livox_transform);

            feature_frame_delegate(pq.front(), frame);
        }
    }
    printf("Features thread stopped\r\n");
}

feature_thread::feature_thread(ros::NodeHandle* nh) {
    // X,Y,Z,R,P,Y
    std::vector<float> livox_cab;
    nh->param<std::vector<float>>("/livox_transform", livox_cab, { 0, 0, 0, 0, 0, 0 });

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

    sync_frame_delegate.append([this](const synced_message& msg) { q.push(msg); });
    thread = std::thread(&feature_thread::__features_thread_entry, this);
}

void feature_thread::__features_thread_entry(feature_thread* self) {
    self->__features_thread();
}

feature_thread::~feature_thread() {
    should_stop = true;
    q.notify();
    thread.join();
}

std::shared_ptr<feature_thread> create_feature_thread(ros::NodeHandle* nh) {
    return std::make_shared<feature_thread>(nh);
}
