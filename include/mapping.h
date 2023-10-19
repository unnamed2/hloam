#ifndef __MAPPING_H__
#define __MAPPING_H__

#include "Scancontext.h"
#include "comm.h"
static inline void transform_cloud(const pcl::PointCloud<PointType>::Ptr& cloud,
                                   pcl::PointCloud<PointType>::Ptr& out,
                                   const Eigen::Matrix4d& matrix) {
    if(cloud == nullptr) {
        out.reset();
        return;
    }

    if(out == nullptr)
        out.reset(new pcl::PointCloud<PointType>);

    out->resize(cloud->size());
    size_t i = 0;
    for(auto&& point: *cloud) {
        Eigen::Vector4d p(point.x, point.y, point.z, 1);

        auto transformed = matrix * p;
        auto new_point = point;
        new_point.x = transformed.x();
        new_point.y = transformed.y();
        new_point.z = transformed.z();
        (*out)[i++] = new_point;
    }

    out->header = cloud->header;
}

static inline void transform_cloud(const feature_objects& cloud, feature_objects& out,
                                   const Eigen::Matrix4d& matrix) {
    transform_cloud(cloud.line_features, out.line_features, matrix);
    transform_cloud(cloud.plane_features, out.plane_features, matrix);
    transform_cloud(cloud.non_features, out.non_features, matrix);
}

template<typename PointTypePtr>
static inline void concat(PointTypePtr& out, const PointTypePtr& cloud) {
    if(cloud == nullptr)
        return;

    if(out == nullptr) {
        out.reset(new pcl::PointCloud<PointType>());
    }

    *out += *cloud;
}

static inline void concat(feature_objects& out, const feature_objects& feature) {
    concat(out.line_features, feature.line_features);
    concat(out.plane_features, feature.plane_features);
    concat(out.non_features, feature.non_features);
}

using LMTransform = ::Transform;

struct velodyne_frame {
    feature_objects velodyne_features;
    Eigen::Matrix4d transform;
};

struct loop_result {
    size_t source_frame_id;
    size_t target_frame_id;
    LMTransform transform;
};

struct loop_var {
    std::vector<velodyne_frame> frames;

    std::vector<loop_result> loop;

    SCManager sc_manager;

    int loop_counter;

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
};

#endif
