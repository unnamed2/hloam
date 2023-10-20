#ifndef __COMM_H__
#define __COMM_H__

#include <condition_variable>
#include <mutex>
#include <nanoflann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <queue>
#include <ros/ros.h>
#include <thread>
struct XYZIRT {
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    std::uint16_t ring;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    XYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, ring,
                                                                         ring)(double, time, time));
template<typename T>
static inline auto p2(T x) -> decltype(x * x) {
    return x * x;
}

template<typename T>
static inline auto distance2(const T& v1, const T& v2) {
    return p2(v1.x - v2.x) + p2(v1.y - v2.y) + p2(v1.z - v2.z);
}

using PointType = XYZIRT;

struct synced_message {
    pcl::PointCloud<PointType>::Ptr velodyne;
    pcl::PointCloud<PointType>::Ptr livox;
    ros::Time time;
};

struct feature_objects {
    pcl::PointCloud<PointType>::Ptr line_features;
    pcl::PointCloud<PointType>::Ptr plane_features;
    pcl::PointCloud<PointType>::Ptr non_features;
};

struct feature_frame {
    feature_objects livox_feature;
    feature_objects velodyne_feature;
};

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
    pcl::transformPointCloud(*cloud, *out, matrix);
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

void feature_livox(const pcl::PointCloud<PointType>::Ptr& cloud, feature_objects& feature);
void feature_velodyne(const pcl::PointCloud<PointType>::Ptr& cloud, feature_objects& feature);

struct Transform {
    double x = 0.0f, y = 0.0f, z = 0.0f;
    double roll = 0.0f, pitch = 0.0f, yaw = 0.0f;
};

static inline Eigen::Matrix4d to_eigen(const Transform& tr) {
    // T = s.Matrix([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    // Rx = s.Matrix([[1, 0, 0, 0], [0, s.cos(rx), -s.sin(rx), 0], [0, s.sin(rx), s.cos(rx), 0], [0,
    // 0, 0, 1]]) Ry = s.Matrix([[s.cos(ry), 0, s.sin(ry), 0], [0, 1, 0, 0], [-s.sin(ry), 0,
    // s.cos(ry), 0], [0, 0, 0, 1]]) Rz = s.Matrix([[s.cos(rz), -s.sin(rz), 0, 0], [s.sin(rz),
    // s.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    // Transformation matrix
    // M = T * Rz * Ry * Rx

    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m(0, 3) = tr.x;
    m(1, 3) = tr.y;
    m(2, 3) = tr.z;

    double cr = cos(tr.roll);
    double sr = sin(tr.roll);
    double cp = cos(tr.pitch);
    double sp = sin(tr.pitch);
    double cy = cos(tr.yaw);
    double sy = sin(tr.yaw);

    m(0, 0) = cp * cy;
    m(0, 1) = cy * sp * sr - cr * sy;
    m(0, 2) = sr * sy + cr * cy * sp;
    m(1, 0) = cp * sy;
    m(1, 1) = cr * cy + sp * sr * sy;
    m(1, 2) = cr * sp * sy - cy * sr;
    m(2, 0) = -sp;
    m(2, 1) = cp * sr;
    m(2, 2) = cr * cp;

    return m;
}

static inline Transform from_eigen(const Eigen::Matrix4d& m) {
    Transform tr;
    tr.x = m(0, 3);
    tr.y = m(1, 3);
    tr.z = m(2, 3);
    tr.pitch = atan2(-m(2, 0), sqrt(p2(m(0, 0)) + p2(m(1, 0))));

    double c = cos(tr.pitch);
    tr.yaw = atan2(m(1, 0) / c, m(0, 0) / c);
    tr.roll = atan2(m(2, 1) / c, m(2, 2) / c);

    return tr;
}

// ===== This example shows how to use nanoflann with these types of containers: =======
// typedef std::vector<std::vector<double> > my_vector_of_vectors_t;
// typedef std::vector<Eigen::VectorXd> my_vector_of_vectors_t;   // This requires #include
// <Eigen/Dense>
// =====================================================================================

/** A simple vector-of-vectors adaptor for nanoflann, without duplicating the storage.
 *  The i'th vector represents a point in the state space.
 *
 *  \tparam DIM If set to >0, it specifies a compile-time fixed dimensionality for the points in the
 * data set, allowing more compiler optimizations. \tparam num_t The type of the point coordinates
 * (typically, double or float). \tparam Distance The distance metric to use: nanoflann::metric_L1,
 * nanoflann::metric_L2, nanoflann::metric_L2_Simple, etc. \tparam IndexType The type for indices in
 * the KD-tree index (typically, size_t of int)
 */

template<typename point_type>
struct array_adaptor {
    using num_t = decltype(point_type::x);
    typedef array_adaptor<point_type> self_t;
    typedef typename nanoflann::metric_L2::template traits<num_t, self_t>::distance_t metric_t;
    typedef nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, 3, size_t> index_t;

    std::shared_ptr<index_t> index; //! The kd-tree index for the user to call its methods as usual
                                    //! with any other FLANN index.

    /// Constructor: takes a const ref to the vector of vectors object with the data points
    array_adaptor(const point_type* array, size_t count, const int leaf_max_size = 10):
        m_data(array), length(count) {
        if(array != nullptr) {
            index.reset(
                new index_t(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size)));
            index->buildIndex();
        }
    }

    const point_type* m_data;
    size_t length;

    /** Query for the \a num_closest closest points to a given point (entered as
     * query_point[0:dim-1]). Note that this is a short-cut method for index->findNeighbors(). The
     * user can also call index->... methods as desired. \note nChecks_IGNORED is ignored but kept
     * for compatibility with the original FLANN interface.
     */
    inline void query(const point_type& query_point, const size_t num_closest, size_t* out_indices,
                      num_t* out_distances_sq, const int nChecks_IGNORED = 10) const {
        num_t val[3] = { query_point.x, query_point.y, query_point.z };
        nanoflann::KNNResultSet<num_t, size_t> resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, val, nanoflann::SearchParams());
    }

    /** @name Interface expected by KDTreeSingleIndexAdaptor
     * @{ */

    const self_t& derived() const {
        return *this;
    }

    self_t& derived() {
        return *this;
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return length;
    }

    // Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const {
        switch(dim) {
        case 0:
            return m_data[idx].x;
        case 1:
            return m_data[idx].y;
        case 2:
            return m_data[idx].z;
        }
        return 0;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation
    // loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be
    //   avoided to redo it again. Look at bb.size() to find out the expected dimensionality (e.g. 2
    //   or 3 for point clouds)
    template<class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const {
        return false;
    }
}; // end of KDTreeVectorOfVectorsAdaptor

struct __AlwaysFalse {
    constexpr bool operator()() const {
        return false;
    }
};

template<typename T>
struct synced_queue {
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond;

    void push(const T& value) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(value);
        cond.notify_one();
    }

    void notify() {
        cond.notify_all();
    }

    template<typename _Pr = __AlwaysFalse>
    std::queue<T> acquire(_Pr pr = __AlwaysFalse()) {
        std::unique_lock<std::mutex> lock(mutex);
        bool pred = false;
        cond.wait(lock, [this, &pr, &pred] {
            pred = pr() || !ros::ok();
            return pred || !queue.empty();
        });

        if(pred) {
            return {};
        }

        std::queue<T> private_queue;
        private_queue.swap(queue);
        return private_queue;
    }
};

struct feature_thread;

std::shared_ptr<feature_thread> create_feature_thread(ros::NodeHandle* nh);

struct mapping_thread;

std::shared_ptr<mapping_thread> create_mapping_thread(ros::NodeHandle* nh);

#include <delegate>

extern delegate<void(const synced_message&)> sync_frame_delegate;
extern delegate<void(const synced_message&, const feature_frame&)> feature_frame_delegate;

#endif