#ifndef __RESIDUAL_H__
#define __RESIDUAL_H__

#include <comm.h>

inline size_t size_of(const pcl::PointCloud<PointType>::Ptr& cloud) {
    if(cloud) {
        return cloud->size();
    }
    return 0;
}

inline const PointType* data_of(const pcl::PointCloud<PointType>::Ptr& cloud) {
    if(cloud) {
        return cloud->points.data();
    }
    return nullptr;
}

struct feature_adapter {
    array_adaptor<PointType> corner;
    array_adaptor<PointType> surf;
    array_adaptor<PointType> non;

    feature_adapter(const feature_objects& target):
        corner(data_of(target.line_features), size_of(target.line_features)),
        surf(data_of(target.plane_features), size_of(target.plane_features)),
        non(data_of(target.non_features), size_of(target.non_features)) {
    }
};

using feature_pair = std::pair<feature_objects, feature_adapter>;

struct newton {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    size_t top;
};

newton Ab(std::initializer_list<feature_pair> pairs, const Transform& t = Transform(),
          float* _loss = nullptr);

Transform LM(const feature_objects& source, const feature_objects& target,
             const Transform& initial_guess = Transform(), float* loss = nullptr);
#endif
