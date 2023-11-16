#ifndef __LOOP_AGENT__
#define __LOOP_AGENT__

#include "Scancontext.h"
#include "comm.h"

#include <gtsam/geometry/Pose3.h>

using LMTransform = ::Transform;

struct velodyne_frame {
    pcl::PointCloud<XYZIRT>::Ptr velodyne_cloud;
    Eigen::Matrix4d transform;
};

struct loop_result {
    size_t source_frame_id;
    size_t target_frame_id;
    gtsam::Pose3 transform;
};

constexpr auto NO_LOOP = std::numeric_limits<size_t>::max();

struct loop_var {
    std::vector<velodyne_frame> frames;

    std::vector<loop_result> loop;

    SCManager sc_manager;

    int loop_counter = 0;
    size_t loop_reset = 5;
    float loop_max_loss = 0.05f;
    size_t min_constriant_node = 0;
    loop_var();

    size_t loop_detection(const pcl::PointCloud<XYZIRT>::Ptr& cloud, const feature_objects& frame,
                          const Eigen::Matrix4d& transform);

    void optimization(size_t from_id);

    const Eigen::Matrix4d& tr(size_t id) {
        return frames[id].transform;
    }

    const Eigen::Matrix4d& btr(size_t id) {
        return tr(frames.size() - id);
    }

    void pop_back();
};

Eigen::Matrix4d solve_GTSAM(const Eigen::Matrix4d& M1, const Eigen::Matrix4d& M2, float loss_M1,
                            float loss_M2);

#endif