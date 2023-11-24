#include "comm.h"

#include <filesystem>
#include <mutex>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <signal.h>
#include <tf/transform_broadcaster.h>

struct XYZRPYT {
    PCL_ADD_POINT4D;
    double roll, pitch, yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(XYZRPYT,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (double, roll, roll)
                                  (double, pitch, pitch)
                                  (double, yaw, yaw)
                                  (double, time, time)
                        );

// clang format on
struct LMCacheItem {
    Eigen::Matrix4d pose;
    double time;
};

struct LMCacheTable {
    std::vector<LMCacheItem> items;
    size_t next_index;

    LMCacheTable(const char* filename) {
        FILE* fp = fopen(filename, "r");

        double time, x, y, z, qx, qy, qz, qw;
        while(fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf", &time, &x, &y, &z, &qx, &qy, &qz,
                     &qw) == 8) {
            LMCacheItem item;
            item.time = time;
            Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
            pose.block<3, 3>(0, 0) = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
            pose(0, 3) = x;
            pose(1, 3) = y;
            pose(2, 3) = z;
            item.pose = pose;
            items.push_back(item);
        }

        fclose(fp);

        next_index = 0;
        printf("LMCacheTable loaded %zd items\r\n", items.size());
    }

    bool test_for(double key) {
        if(items.empty()) {
            return false;
        }

        if(fabsf(items[next_index].time - key) < 0.05) {
            return true;
        }

        return false;
    }

    Eigen::Matrix4d next() {
        if(items.empty()) {
            return Eigen::Matrix4d::Identity();
        }

        Eigen::Matrix4d ret = items[next_index].pose;
        next_index++;
        if(next_index >= items.size()) {
            ROS_INFO("next_index >= items.size(), All done!");
            next_index = 0;
            items.clear();
        }
        return ret;
    }

    bool should_drop(double time) const {
        if(items.empty()) {
            return false;
        }

        return time > items[next_index].time + 0.05;
    }

    void drop_to(double sec) {
        while(should_drop(sec)) {
            next();
        }
    }
};

struct stamped_velodyne {
    pcl::PointCloud<PointType>::Ptr cloud;
    double time;
};

static auto minmax_time(const stamped_velodyne& cloud) {
    auto var =
        std::minmax_element(cloud.cloud->begin(), cloud.cloud->end(),
                            [](const PointType& a, const PointType& b) { return a.time < b.time; });

    double offset = 0.0f;
    if(var.first->time < 1.0f) {
        offset = cloud.time;
    }

    return std::make_pair(var.first->time + offset, var.second->time + offset);
}

template<typename T>
static auto downsample(pcl::PointCloud<T>& input, double rate) {
    pcl::PointCloud<T> out;
    for(size_t i = 0; i < input.size(); i++) {
        double val = rand() / (RAND_MAX + 1.0);
        if(val < rate) {
            out.push_back(input[i]);
        }
    }
    return out;
}

class Gen: public ros::NodeHandle {
    std::vector<PointType> livox_sequences;
    std::queue<stamped_velodyne> velodyne_sequences;

    size_t livox_index = 0;

    ros::Subscriber sub_livox;
    ros::Subscriber sub_velodyne;

    std::string livox_frame_id, velodyne_frame_id;

    std::mutex mtx;

    LMCacheTable* tbl = nullptr;
    Eigen::Matrix4d livox_transform;

    ros::Publisher pub_global_velodyne;
    ros::Publisher pub_global_livox;

    ros::Publisher pub_feature_velodyne_line;
    ros::Publisher pub_feature_velodyne_plane;
    ros::Publisher pub_feature_livox_plane;
    ros::Publisher pub_feature_livox_non;

    pcl::PointCloud<XYZIRT> global_map_velodyne;
    pcl::PointCloud<XYZIRT> global_map_livox;

    feature_frame global_features;

    std::string global_map_filename;

    tf::TransformBroadcaster br;

    double downsample_rate = 0.01;

    std::vector<LMCacheItem> traces;

    bool lego = false;

public:
    Gen(): ros::NodeHandle("hloam") {
        std::string livox_topic, velodyne_topic;

        std::string lmcache_filename;
        param<std::string>("/gen/pose", lmcache_filename, "");
        if(lmcache_filename.empty()) {
            ROS_FATAL("pose file not specified");
            exit(1);
        }

        tbl = new LMCacheTable(lmcache_filename.c_str());
        if(tbl->items.empty()) {
            ROS_FATAL("pose file empty");
            exit(1);
        }

        param<std::string>("/gen/livox_topic", livox_topic, "/livox_hap");
        param<std::string>("/gen/velodyne_topic", velodyne_topic, "/u2102");
        param<std::string>("/gen/global_map", global_map_filename, "/tmp/global_map.pcd");
        param<double>("/gen/downsample_rate", downsample_rate, 0.01);
        param<bool>("/gen/lego", lego, false);
        // X,Y,Z,R,P,Y
        std::vector<float> livox_cab;
        param<std::vector<float>>("/gen/livox_transform", livox_cab, { 0, 0, 0, 0, 0, 0 });
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

        livox_transform = to_eigen(tr).inverse();

        ROS_INFO("livox_transform: %f %f %f %f %f %f", tr.x, tr.y, tr.z, tr.roll, tr.pitch, tr.yaw);

        ROS_INFO("Subscribing to %s and %s", livox_topic.c_str(), velodyne_topic.c_str());
        sub_livox = subscribe(livox_topic, 100, &Gen::livox_callback, this);
        sub_velodyne = subscribe(velodyne_topic, 100, &Gen::velodyne_callback, this);

        pub_global_velodyne = advertise<sensor_msgs::PointCloud2>("/global_map/velodyne", 100);
        pub_global_livox = advertise<sensor_msgs::PointCloud2>("/global_map/livox", 100);
        pub_feature_velodyne_line =
            advertise<sensor_msgs::PointCloud2>("/features/velodyne_line", 100);
        pub_feature_velodyne_plane =
            advertise<sensor_msgs::PointCloud2>("/features/velodyne_plane", 100);

        pub_feature_livox_plane = advertise<sensor_msgs::PointCloud2>("/features/livox_plane", 100);
        pub_feature_livox_non = advertise<sensor_msgs::PointCloud2>("/features/livox_non", 100);
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
        std::sort(livox_sequences.begin(), livox_sequences.end(),
                  [](const PointType& a, const PointType& b) { return a.time < b.time; });

        auto start_livox_index = livox_index;
        while(ros::ok() && __try_combine_clouds())
            ;
        if(start_livox_index != livox_index) {
            livox_sequences.erase(livox_sequences.begin(),
                                  livox_sequences.begin() + start_livox_index);
            livox_index -= start_livox_index;
        }
    }

    void call_features(pcl::PointCloud<PointType>::Ptr livox_cloud,
                       pcl::PointCloud<PointType>::Ptr velodyne_cloud, ros::Time time) {

        if(livox_cloud->size() < 100 || velodyne_cloud->size() < 100) {
            ROS_INFO("livox_cloud->size(%zd) < 100 || velodyne_cloud->size(%zd) < 100",
                     livox_cloud->size(), velodyne_cloud->size());
            return;
        }

        double sec = time.toSec();
        tbl->drop_to(sec);

        if(!tbl->test_for(sec)) {
            ROS_INFO("Not ready for pose %lf", sec);
            return;
        }

        feature_objects livox_features;
        feature_livox(livox_cloud, livox_features);

        feature_objects velodyne_features;
        feature_velodyne(velodyne_cloud, velodyne_features);

        Eigen::Matrix4d pose = tbl->next();
        if(lego) {
            Transform t = { 0, 0, 0, M_PI_2, 0, M_PI_2 };
            pose = to_eigen(t) * pose;
        }

        traces.push_back({ pose, sec });

        pcl::PointCloud<PointType> livox_tr;
        pcl::PointCloud<PointType> velodyne_tr;

        pcl::transformPointCloud(*livox_cloud, livox_tr,
                                 static_cast<Eigen::Matrix4d>(pose * livox_transform));

        pcl::transformPointCloud(*livox_features.non_features, *livox_features.non_features,
                                 static_cast<Eigen::Matrix4d>(pose * livox_transform));

        pcl::transformPointCloud(*livox_features.plane_features, *livox_features.plane_features,
                                 static_cast<Eigen::Matrix4d>(pose * livox_transform));

        pcl::transformPointCloud(*velodyne_features.line_features, *velodyne_features.line_features,
                                 static_cast<Eigen::Matrix4d>(pose));

        pcl::transformPointCloud(*velodyne_features.plane_features,
                                 *velodyne_features.plane_features,
                                 static_cast<Eigen::Matrix4d>(pose));

        pcl::transformPointCloud(*velodyne_cloud, velodyne_tr, pose);

        concat(global_features.livox_feature, livox_features);
        concat(global_features.velodyne_feature, velodyne_features);

        auto velodyne_ds = downsample(velodyne_tr, downsample_rate);
        auto livox_ds = downsample(livox_tr, downsample_rate);

        global_map_livox += livox_ds;
        global_map_velodyne += velodyne_ds;

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(velodyne_ds, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = time;
        pub_global_velodyne.publish(msg);

        pcl::toROSMsg(livox_ds, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = time;
        pub_global_livox.publish(msg);

        tf::StampedTransform st;
        st.frame_id_ = "map";
        st.child_frame_id_ = "velodyne16";
        st.stamp_ = time;
        st.setOrigin(tf::Vector3(pose(0, 3), pose(1, 3), pose(2, 3)));

        Eigen::Quaterniond q(pose.block<3, 3>(0, 0));
        st.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
        br.sendTransform(st);

        pcl::toROSMsg(*livox_features.non_features, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = time;
        pub_feature_livox_non.publish(msg);

        pcl::toROSMsg(*livox_features.plane_features, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = time;
        pub_feature_livox_plane.publish(msg);

        pcl::toROSMsg(*velodyne_features.line_features, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = time;
        pub_feature_velodyne_line.publish(msg);

        pcl::toROSMsg(*velodyne_features.plane_features, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = time;
        pub_feature_velodyne_plane.publish(msg);
    }

    void save_global_map() {
        if(global_map_filename.empty())
            return;
        return;
        if(std::filesystem::exists(global_map_filename)) {
            if(!std::filesystem::is_directory(global_map_filename)) {
                std::filesystem::remove(global_map_filename);
                std::filesystem::create_directories(global_map_filename);
            }

        } else {
            std::filesystem::create_directories(global_map_filename);
        }

        char filenames[1024];
        sprintf(filenames, "%s/velodyne_global.pcd", global_map_filename.c_str());
        pcl::io::savePCDFileBinary(filenames, global_map_velodyne);

        sprintf(filenames, "%s/livox_global.pcd", global_map_filename.c_str());
        pcl::io::savePCDFileBinary(filenames, global_map_livox);

        pcl::PointCloud<PointType> global_map;
        global_map += global_map_velodyne;
        global_map += global_map_livox;

        sprintf(filenames, "%s/global.pcd", global_map_filename.c_str());
        pcl::io::savePCDFileBinary(filenames, global_map);

        sprintf(filenames, "%s/livox_plane_features.pcd", global_map_filename.c_str());
        pcl::io::savePCDFileBinary(filenames, *global_features.livox_feature.plane_features);

        sprintf(filenames, "%s/livox_non_features.pcd", global_map_filename.c_str());
        pcl::io::savePCDFileBinary(filenames, *global_features.livox_feature.non_features);

        sprintf(filenames, "%s/velodyne_plane_features.pcd", global_map_filename.c_str());
        pcl::io::savePCDFileBinary(filenames, *global_features.velodyne_feature.plane_features);

        sprintf(filenames, "%s/velodyne_line_features.pcd", global_map_filename.c_str());
        pcl::io::savePCDFileBinary(filenames, *global_features.velodyne_feature.line_features);

        pcl::PointCloud<XYZRPYT> Tr;
        for(auto& trace: traces) {
            Transform tr = from_eigen(trace.pose);
            XYZRPYT pt;
            pt.x = tr.x;
            pt.y = tr.y;
            pt.z = tr.z;
            pt.roll = tr.roll;
            pt.pitch = tr.pitch;
            pt.yaw = tr.yaw;
            pt.time = trace.time;
            Tr.push_back(pt);
        }

        sprintf(filenames, "%s/trace.pcd", global_map_filename.c_str());
        pcl::io::savePCDFileBinary(filenames, Tr);
    }
};

int main(int argc, char** argv) {
    for(int i = 0; i < argc; i++) {
        ROS_INFO("argv[%d] : %s", i, argv[i]);
    }

    ros::init(argc, argv, "gen_node");
    Gen gen;
    ros::spin();
    printf("Saving global map...\r\n");
    gen.save_global_map();
    printf("Exiting...\r\n");
    return 0;
}