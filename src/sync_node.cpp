#include "comm.h"

#include <mutex>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <signal.h>

struct stamped_velodyne {
    pcl::PointCloud<PointType>::Ptr cloud;
    double time;
};

static auto minmax_time(const stamped_velodyne& cloud) {
    auto var =
        std::minmax_element(cloud.cloud->begin(), cloud.cloud->end(),
                            [](const PointType& a, const PointType& b) { return a.time < b.time; });

    double offset = 0.0f;

    return std::make_pair(var.first->time + offset, var.second->time + offset);
}

class Junk: public ros::NodeHandle {
    std::vector<PointType> livox_sequences;
    std::queue<stamped_velodyne> velodyne_sequences;

    Eigen::Matrix4d livox_transform;

    size_t livox_index = 0;

    ros::Subscriber sub_livox;
    ros::Subscriber sub_velodyne;

    std::string livox_frame_id, velodyne_frame_id;

    std::mutex mtx;

    ros::Publisher pub;

    ros::Publisher publish_combined;

public:
    Junk(): ros::NodeHandle("Junk") {
        std::string livox_topic, velodyne_topic;

        param<std::string>("/livox_topic", livox_topic, "/livox_hap");
        param<std::string>("/velodyne_topic", velodyne_topic, "/u2102");
        // X,Y,Z,R,P,Y
        std::vector<float> livox_cab;
        param<std::vector<float>>("/livox_transform", livox_cab, { 0, 0, 0, 0, 0, 0 });

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

        ROS_INFO("Subscribing to %s and %s", livox_topic.c_str(), velodyne_topic.c_str());
        sub_livox = subscribe(livox_topic, 100, &Junk::livox_callback, this);
        sub_velodyne = subscribe(velodyne_topic, 100, &Junk::velodyne_callback, this);
        pub = advertise<sensor_msgs::PointCloud2>("/global_map", 100);
        publish_combined = advertise<sensor_msgs::PointCloud2>("/combined_cloud", 100);

        publish_delegate.append([this](pcl::PointCloud<PointType>& cloud_velodyne,
                                       pcl::PointCloud<PointType>& cloud_livox, ros::Time time) {
            publish_message(cloud_velodyne, cloud_livox, time);
        });

        sync_frame_delegate.append([this](const synced_message& msg) {
            pcl::PointCloud<PointType> cloud;
            cloud.reserve(msg.livox->size() + msg.velodyne->size());
            cloud.insert(cloud.end(), msg.livox->begin(), msg.livox->end());
            cloud.insert(cloud.end(), msg.velodyne->begin(), msg.velodyne->end());

            sensor_msgs::PointCloud2 msg2;
            pcl::toROSMsg(cloud, msg2);
            msg2.header.frame_id = "velodyne16";
            msg2.header.stamp = msg.time;
            publish_combined.publish(msg2);
        });
    }

    void livox_callback(const sensor_msgs::PointCloud2ConstPtr& msg) {

        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
        pcl::fromROSMsg(*msg, *cloud);

        if(cloud->empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mtx);
        livox_frame_id = msg->header.frame_id;
        livox_sequences.insert(livox_sequences.end(), cloud->points.begin(), cloud->points.end());
        try_combine_clouds();
    }

    void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
        pcl::fromROSMsg(*msg, *cloud);
        if(cloud->empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mtx);
        velodyne_frame_id = msg->header.frame_id;
        velodyne_sequences.push({ cloud, msg->header.stamp.toSec() });
        try_combine_clouds();
    }

    bool __try_combine_clouds() {
        if(livox_sequences.empty() || velodyne_sequences.empty()) {
            return false;
        }

        auto [velodyne_frame_start, velodyne_frame_end] = minmax_time(velodyne_sequences.front());
        double livox_frame_end = livox_sequences.back().time;
        double livox_frame_start = livox_sequences.front().time;

        if(livox_frame_end < velodyne_frame_end) {
            return false;
        }

        if(velodyne_frame_start < livox_frame_start) {
            velodyne_sequences.pop();
            return true;
        };

        pcl::PointCloud<PointType>::Ptr livox_cloud(new pcl::PointCloud<PointType>);

        size_t start_index = livox_index;

        while(start_index < livox_sequences.size()) {
            double livox_time = livox_sequences[start_index].time;
            if(livox_time > velodyne_frame_start) {
                break;
            }
            start_index++;
        }

        size_t end_index = start_index;
        while(end_index < livox_sequences.size()) {
            double livox_time = livox_sequences[end_index].time;
            if(livox_time > velodyne_frame_end) {
                break;
            }
            end_index++;
        }

        livox_cloud->points.assign(livox_sequences.begin() + start_index,
                                   livox_sequences.begin() + end_index);
        livox_index = end_index;

        call_features(livox_cloud, velodyne_sequences.front().cloud,
                      ros::Time(velodyne_frame_start));

        velodyne_sequences.pop();
        return true;
    }

    void try_combine_clouds() {
        while(ros::ok() && __try_combine_clouds())
            ;
    }

    void call_features(pcl::PointCloud<PointType>::Ptr livox_cloud,
                       pcl::PointCloud<PointType>::Ptr velodyne_cloud, ros::Time time) {

        if(livox_cloud->size() < 100 || velodyne_cloud->size() < 100) {
            ROS_INFO("livox_cloud->size(%zd) < 100 || velodyne_cloud->size(%zd) < 100",
                     livox_cloud->size(), velodyne_cloud->size());
            return;
        }
        synced_message msg;
        msg.livox = livox_cloud;
        msg.velodyne = velodyne_cloud;
        msg.time = time;

        sync_frame_delegate(msg);
    }

    void publish_message(pcl::PointCloud<PointType>& cloud_velodyne,
                         pcl::PointCloud<PointType>& cloud_livox, ros::Time time) {

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud_velodyne, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = time;
        pub.publish(msg);

        pcl::PointCloud<PointType> transform;
        pcl::transformPointCloud(cloud_livox, transform, livox_transform);
        pcl::toROSMsg(transform, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = time;

        pub.publish(msg);
    }
};

int main(int argc, char** argv) {
    for(int i = 0; i < argc; i++) {
        ROS_INFO("argv[%d] : %s", i, argv[i]);
    }

    ros::init(argc, argv, "sync_node");
    ros::NodeHandle nh;
    Junk junk;

    auto feature_thrd = create_feature_thread(&junk);
    auto mapping_Thrd = create_mapping_thread(&junk);

    ros::spin();

    printf("Exiting...");
    return 0;
}