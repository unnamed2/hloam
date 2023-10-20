#include "loop.h"

#include <filesystem>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>

struct LMCacheItem {
    Eigen::Matrix4d pose;
    double time;
    Transform LM_result;
};

struct LMCacheTable {
    std::vector<LMCacheItem> items;
    size_t next_index;

    LMCacheTable(const char* filename) {
        FILE* fp = fopen(filename, "r");
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
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

            Eigen::Matrix4d X = pose * transform.inverse();
            item.LM_result = from_eigen(X);

            transform = pose;

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

        return fabsf(items[next_index].time - key) < 0.05;
    }

    Eigen::Matrix4d next() {
        if(items.empty()) {
            return Eigen::Matrix4d::Identity();
        }

        Eigen::Matrix4d ret = items[next_index].pose;
        next_index++;
        if(next_index >= items.size()) {
            next_index = 0;
        }
        return ret;
    }
};

std::vector<std::filesystem::path> list_all_files(const char* dir) {
    std::vector<std::filesystem::path> files;
    for(auto& p: std::filesystem::directory_iterator(dir)) {
        if(p.is_regular_file()) {
            files.push_back(p.path());
        }
    }
    return files;
}

static bool sort_cmp(const std::filesystem::path& a, const std::filesystem::path& b) {
    size_t fid = atoi(a.filename().string().c_str());
    size_t tid = atoi(b.filename().string().c_str());

    return fid < tid;
}

int run_loop(LMCacheTable& cache, std::vector<std::filesystem::path>& clouds) {

    ros::NodeHandle nh;
    ros::Publisher global_map = nh.advertise<sensor_msgs::PointCloud2>("/global_map", 100);

    tf::TransformBroadcaster br;

    loop_var vars;
    pcl::PointCloud<XYZIRT>::Ptr cloud(new pcl::PointCloud<XYZIRT>);
    feature_objects features;

    std::vector<Eigen::Matrix4d> final_traces;

    Eigen::Matrix4d corr = Eigen::Matrix4d::Identity();
    size_t index = 0;
    for(auto var: clouds) {
        if(++index % 10 == 0) {
            printf("running %s\r\n", var.string().c_str());
        }
        if(index < 500 || index > 1000 && index < 6000) {
            cache.next();
            continue;
        }

        pcl::io::loadPCDFile(var.string(), *cloud);

        feature_velodyne(cloud, features);

        Eigen::Matrix4d this_tr = corr * cache.next();
        final_traces.push_back(this_tr);
        pcl::PointCloud<XYZIRT>::Ptr global;
        transform_cloud(features.plane_features, global, this_tr);

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*global, msg);

        msg.header.frame_id = "map";
        msg.header.stamp = ros::Time::now();

        global_map.publish(msg);

        tf::StampedTransform st;
        st.frame_id_ = "map";
        st.child_frame_id_ = "velodyne16";
        st.stamp_ = msg.header.stamp;
        st.setOrigin(tf::Vector3(this_tr(0, 3), this_tr(1, 3), this_tr(2, 3)));
        Eigen::Quaterniond q(this_tr.block<3, 3>(0, 0));
        st.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
        br.sendTransform(st);

        size_t id = vars.loop_detection(cloud, features, this_tr);
        if(id == 0)
            continue;

        for(size_t i = id; i < final_traces.size(); i++) {
            final_traces[i] = vars.tr(i);
        }

        Eigen::Matrix4d C = final_traces.back() * this_tr.inverse();
        corr = C;
    }

    for(size_t i = 0; i < final_traces.size(); i++) {
        Transform tr = from_eigen(final_traces[i]);
        printf("%zd %lf %lf %lf %lf %lf %lf\r\n", i, tr.x, tr.y, tr.z, tr.roll, tr.pitch, tr.yaw);
    }

    return 0;
}

int main(int argc, const char* const* argv) {
    if(argc < 3) {
        printf("Usage: %s <LMCacheTable> <velodyne_clouds>\r\n", argv[0]);
        return 0;
    }
    LMCacheTable cache(argv[1]);

    auto files = list_all_files(argv[2]);
    std::sort(files.begin(), files.end(), sort_cmp);

    files.erase(files.begin());
    files.erase(files.end());

    if(files.size() != cache.items.size()) {
        printf("files.size() != cache.items.size()\r\n");
        return 0;
    }

    ros::init(argc, (char**)argv, "loop_test");
    return run_loop(cache, files);
}
