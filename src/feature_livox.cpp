#include "comm.h"

#include <Eigen/Dense>
#include <condition_variable>
#include <mutex>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <queue>
#include <thread>

void detectFeaturePoint2(const pcl::PointCloud<PointType>::Ptr& cloud,
                         pcl::PointCloud<PointType>::Ptr& pointsLessFlat,
                         pcl::PointCloud<PointType>::Ptr& pointsNonFeature) {

    int cloudSize = cloud->points.size();

    pointsLessFlat.reset(new pcl::PointCloud<PointType>());
    pointsNonFeature.reset(new pcl::PointCloud<PointType>());

    pcl::KdTreeFLANN<PointType>::Ptr KdTreeCloud;
    KdTreeCloud.reset(new pcl::KdTreeFLANN<PointType>);
    if(cloud->empty()) {
        printf("detectFeaturePoint2 empty!\r\n");
    }
    KdTreeCloud->setInputCloud(cloud);

    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;

    int num_near = 10;
    int stride = 1;
    int interval = 4;

    for(int i = 5; i < cloudSize - 5; i = i + stride) {
        double thre1d = 0.5;
        double thre2d = 0.8;
        double thre3d = 0.5;
        double thre3d2 = 0.13;

        double disti =
            sqrt(cloud->points[i].x * cloud->points[i].x + cloud->points[i].y * cloud->points[i].y +
                 cloud->points[i].z * cloud->points[i].z);

        if(disti < 30.0) {
            thre1d = 0.5;
            thre2d = 0.8;
            thre3d2 = 0.07;
            stride = 14;
            interval = 4;
        } else if(disti < 60.0) {
            stride = 10;
            interval = 3;
        } else {
            stride = 1;
            interval = 0;
        }

        if(disti > 100.0) {
            num_near = 6;
            pointsNonFeature->points.push_back(cloud->points[i]);
            continue;
        } else if(disti > 60.0) {
            num_near = 8;
        } else {
            num_near = 10;
        }

        KdTreeCloud->nearestKSearch(cloud->points[i], num_near, _pointSearchInd, _pointSearchSqDis);

        if(_pointSearchSqDis[num_near - 1] > 5.0 && disti < 90.0) {
            continue;
        }

        Eigen::Matrix<double, 3, 3> _matA1;
        _matA1.setZero();

        float cx = 0;
        float cy = 0;
        float cz = 0;
        for(int j = 0; j < num_near; j++) {
            cx += cloud->points[_pointSearchInd[j]].x;
            cy += cloud->points[_pointSearchInd[j]].y;
            cz += cloud->points[_pointSearchInd[j]].z;
        }
        cx /= num_near;
        cy /= num_near;
        cz /= num_near;

        float a11 = 0;
        float a12 = 0;
        float a13 = 0;
        float a22 = 0;
        float a23 = 0;
        float a33 = 0;
        for(int j = 0; j < num_near; j++) {
            float ax = cloud->points[_pointSearchInd[j]].x - cx;
            float ay = cloud->points[_pointSearchInd[j]].y - cy;
            float az = cloud->points[_pointSearchInd[j]].z - cz;

            a11 += ax * ax;
            a12 += ax * ay;
            a13 += ax * az;
            a22 += ay * ay;
            a23 += ay * az;
            a33 += az * az;
        }
        a11 /= num_near;
        a12 /= num_near;
        a13 /= num_near;
        a22 /= num_near;
        a23 /= num_near;
        a33 /= num_near;

        _matA1(0, 0) = a11;
        _matA1(0, 1) = a12;
        _matA1(0, 2) = a13;
        _matA1(1, 0) = a12;
        _matA1(1, 1) = a22;
        _matA1(1, 2) = a23;
        _matA1(2, 0) = a13;
        _matA1(2, 1) = a23;
        _matA1(2, 2) = a33;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
        double a1d = (sqrt(saes.eigenvalues()[2]) - sqrt(saes.eigenvalues()[1])) /
            sqrt(saes.eigenvalues()[2]);
        double a2d = (sqrt(saes.eigenvalues()[1]) - sqrt(saes.eigenvalues()[0])) /
            sqrt(saes.eigenvalues()[2]);
        double a3d = sqrt(saes.eigenvalues()[0]) / sqrt(saes.eigenvalues()[2]);

        if(a2d > thre2d || (a3d < thre3d2 && a1d < thre1d)) {
            for(int k = 1; k < interval; k++) {
                pointsLessFlat->points.push_back(cloud->points[i - k]);
                pointsLessFlat->points.push_back(cloud->points[i + k]);
            }
            pointsLessFlat->points.push_back(cloud->points[i]);
        } else if(a3d > thre3d) {
            for(int k = 1; k < interval; k++) {
                pointsNonFeature->points.push_back(cloud->points[i - k]);
                pointsNonFeature->points.push_back(cloud->points[i + k]);
            }
            pointsNonFeature->points.push_back(cloud->points[i]);
        }
    }
}

void FeatureExtract_hap(const pcl::PointCloud<XYZIRT>& msg,
                        pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                        pcl::PointCloud<PointType>::Ptr& laserNonFeature) {
    laserSurfFeature->clear();
    laserNonFeature->clear();

    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());

    laserCloud->clear();
    laserCloud->resize(msg.size());

    double time_base = msg.points.front().time;
    double timeSpan = msg.points.back().time - time_base;

    for(size_t i = 0; i < msg.size(); i++) {
        laserCloud->at(i) = msg.points[i];
        laserCloud->at(i).time = (msg.points[i].time - time_base) / timeSpan;
    }

    detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);
}

void feature_livox(const pcl::PointCloud<PointType>::Ptr& cloud, feature_objects& feature) {
    // no line features for livox-hap
    feature.line_features.reset();

    if(feature.plane_features != nullptr) {
        feature.plane_features->clear();
    } else {
        feature.plane_features.reset(new pcl::PointCloud<PointType>());
    }

    if(feature.non_features != nullptr) {
        feature.non_features->clear();
    } else {
        feature.non_features.reset(new pcl::PointCloud<PointType>());
    }

    FeatureExtract_hap(*cloud, feature.plane_features, feature.non_features);
}
