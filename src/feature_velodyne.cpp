#include <Eigen/Dense>
#include <comm.h>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/pcl_base.hpp>
#include <vector>
template<typename T>
struct array_view {
    T* __ptr;
    size_t __size;

    array_view(T* begin, size_t size): __ptr(begin), __size(size) {
    }

    T& operator[](size_t i) {
        return __ptr[i];
    }

    const T& operator[](size_t i) const {
        return __ptr[i];
    }

    T* begin() {
        return __ptr;
    }

    T* end() {
        return __ptr + __size;
    }

    const T* begin() const {
        return __ptr;
    }

    const T* end() const {
        return __ptr + __size;
    }

    size_t size() const {
        return __size;
    }

    bool empty() const {
        return __size == 0;
    }

    T* data() {
        return __ptr;
    }
};

template<typename P1, typename P2>
auto distance2(const P1& p1, const P2& p2) {
    return p2(p1.x - p2.x) + p2(p1.y - p2.y) + p2(p1.z - p2.z);
}

struct smoothness_t {
    float value;
    size_t ind;

    bool operator<(const smoothness_t& other) const {
        return value < other.value;
    }
};

struct ranged_points {
    size_t range_offsets[64];

    template<typename value_type>
    array_view<value_type> ring_span(int ring_id, value_type* points) {
        if(ring_id == 0) {
            return array_view(points, range_offsets[0]);
        }
        return array_view(points + range_offsets[ring_id - 1],
                          range_offsets[ring_id] - range_offsets[ring_id - 1]);
    }
};

template<typename point_type>
struct cloud_featured {
    std::vector<point_type> corner_points;
    std::vector<point_type> surface_points;
};

template<typename continer_type>
inline void downsample_surf(continer_type& surface_points) {
    auto selected = surface_points.begin();
    for(auto i = selected + 1; i != surface_points.end(); i++) {
        if(p2(i->x - selected->x) + p2(i->y - selected->y) + p2(i->z - selected->z) < 4.) {
            continue;
        }
        *++selected = *i;
    }
    surface_points.erase(selected + 1, surface_points.end());
}

template<typename point_type>
inline void get_features(point_type* begin, point_type* end, const size_t* ranges,
                         feature_objects& features) {

    constexpr size_t H_SCAN = 1800;
    constexpr float edgeThreshold = 1.0;
    constexpr float surfThreshold = 0.1;

    ranged_points points;

    std::vector<float> range(std::distance(begin, end));
    std::vector<int> cols(std::distance(begin, end));
    // since points is arranged by time, we can use std::stable_partition to split the points into
    // rings

    point_type* ring_start = begin;
    size_t ring_id = 0;
    while(ring_start != end && ring_id < 64) {
        point_type* ring = ring_start + ranges[ring_id];
        points.range_offsets[ring_id] = ring - begin;
        ring_id++;
        ring_start = ring;
    }

    // now we have the points arranged by rings, we can calculate the features
    // for each ring
    std::vector<float> curvature(std::distance(begin, end));
    std::vector<smoothness_t> smoothness(std::distance(begin, end));

    for(point_type* i = begin; i != end; i++) {
        float d = i->x * i->x + i->y * i->y + i->z * i->z;
        range[i - begin] = std::sqrt(d);

        float angle = std::atan2(i->x, i->y) * 180.0f / M_PI;
        int columnIdn = -round((angle - 90.0f) / (360.0f / H_SCAN)) + H_SCAN / 2;
        if(columnIdn >= H_SCAN)
            columnIdn -= H_SCAN;

        if(columnIdn < 0 || columnIdn >= H_SCAN)
            columnIdn = 0;
        cols[i - begin] = columnIdn;
    }

    std::vector<bool> neighbor_picked(std::distance(begin, end), false);
    std::vector<bool> flag(std::distance(begin, end), false);

    size_t cloudSize = curvature.size();
    for(int i = 5; i < cloudSize - 5; i++) {
        float curv = range[i - 5] + range[i - 4] + range[i - 3] + range[i - 2] + range[i - 1] -
            range[i] * 10 + range[i + 1] + range[i + 2] + range[i + 3] + range[i + 4] +
            range[i + 5];

        curvature[i] = curv * curv;
        smoothness[i].value = curvature[i];
        smoothness[i].ind = i;
    }

    // mark occluded points and parallel beam points
    for(size_t i = 5; i < cloudSize - 6; ++i) {
        // occluded points
        float depth1 = range[i];
        float depth2 = range[i + 1];
        int diff = std::abs(cols[i + 1] - cols[i]);

        if(diff < 10) {
            // 10 pixel diff in range image
            if(depth1 - depth2 > 0.3) {
                neighbor_picked[i - 5] = true;
                neighbor_picked[i - 4] = true;
                neighbor_picked[i - 3] = true;
                neighbor_picked[i - 2] = true;
                neighbor_picked[i - 1] = true;
                neighbor_picked[i] = true;
            } else if(depth2 - depth1 > 0.3) {
                neighbor_picked[i + 1] = true;
                neighbor_picked[i + 2] = true;
                neighbor_picked[i + 3] = true;
                neighbor_picked[i + 4] = true;
                neighbor_picked[i + 5] = true;
                neighbor_picked[i + 6] = true;
            }
        }
        // parallel beam
        float diff1 = std::abs(range[i - 1] - range[i]);
        float diff2 = std::abs(range[i + 1] - range[i]);

        if(diff1 > 0.02 * range[i] && diff2 > 0.02 * range[i])
            neighbor_picked[i] = true;
    }

    for(int i = 0; i < ring_id; i++) {
        auto cloud_span = points.ring_span(i, begin);
        auto smoothness_span = points.ring_span(i, smoothness.data());

        for(int j = 0; j < 6; j++) {

            int sp = (cloud_span.size() * j) / 6;
            int ep = (cloud_span.size() * (j + 1)) / 6 - 1;

            if(sp >= ep)
                continue;

            std::sort(smoothness_span.begin() + sp, smoothness_span.begin() + ep);

            int largestPickedNum = 0;
            for(int k = ep; k >= sp; k--) {
                int ind = smoothness_span[k].ind;
                if(neighbor_picked[ind] == false && curvature[ind] > edgeThreshold &&
                   range[ind] > 2.0f) {
                    largestPickedNum++;
                    if(largestPickedNum <= 20) {
                        flag[ind] = true;
                        features.line_features->push_back(begin[ind]);
                    } else {
                        break;
                    }

                    neighbor_picked[ind] = true;
                    for(int l = 1; l <= 5; l++) {
                        int columnDiff = std::abs(int(cols[ind + l] - cols[ind + l - 1]));
                        if(columnDiff > 10)
                            break;
                        neighbor_picked[ind + l] = true;
                    }
                    for(int l = -1; l >= -5; l--) {
                        int columnDiff = std::abs(int(cols[ind + l] - cols[ind + l + 1]));
                        if(columnDiff > 10)
                            break;
                        neighbor_picked[ind + l] = true;
                    }
                }
            }

            for(int k = sp; k <= ep; k++) {
                int ind = smoothness_span[k].ind;
                if(neighbor_picked[ind] == false && curvature[ind] < surfThreshold &&
                   range[ind] > 2.0f) {

                    flag[ind] = false;
                    neighbor_picked[ind] = true;

                    for(int l = 1; l <= 5; l++) {

                        int columnDiff = std::abs(cols[ind + l] - cols[ind + l - 1]);
                        if(columnDiff > 10)
                            break;

                        neighbor_picked[ind + l] = true;
                    }
                    for(int l = -1; l >= -5; l--) {

                        int columnDiff = std::abs(cols[ind + l] - cols[ind + l + 1]);
                        if(columnDiff > 10)
                            break;

                        neighbor_picked[ind + l] = true;
                    }
                }
            }

            for(int k = sp; k <= ep; k++) {
                if(!flag[smoothness_span[k].ind]) {
                    features.plane_features->push_back(cloud_span[k]);
                }
            }
        }
    }
}

void feature_velodyne(const pcl::PointCloud<PointType>::Ptr& cloud, feature_objects& feature) {
    size_t ring_count[64] = { 0 };
    auto cond = [](auto&& p) {
        auto d = sqrtf(p2(p.x) + p2(p.y) + p2(p.z));
        return d > 2.0 && d < 100.0;
    };

    for(auto& p: *cloud) {
        if(p.ring < 64 && cond(p))
            ring_count[p.ring]++;
    }

    size_t ring_offset[64] = { 0 };
    for(int i = 1; i < 64; i++) {
        ring_offset[i] = ring_offset[i - 1] + ring_count[i - 1];
    }

    std::vector<PointType> points(ring_offset[63] + ring_count[63]);
    for(auto& p: *cloud) {
        if(p.ring < 64 && cond(p))
            points[ring_offset[p.ring]++] = p;
    }

    if(feature.line_features != nullptr) {
        feature.line_features->clear();
    } else {
        feature.line_features.reset(new pcl::PointCloud<PointType>());
    }

    if(feature.plane_features != nullptr) {
        feature.plane_features->clear();
    } else {
        feature.plane_features.reset(new pcl::PointCloud<PointType>());
    }

    feature.non_features.reset();
    get_features(points.data(), points.data() + points.size(), ring_count, feature);
}
