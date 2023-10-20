#include <comm.h>

delegate<void(const synced_message&)> sync_frame_delegate;
delegate<void(const synced_message&, const feature_frame&)> feature_frame_delegate;
delegate<void(pcl::PointCloud<PointType>& cloud_velodyne, pcl::PointCloud<PointType>& cloud_livox,
              ros::Time time)>
    publish_delegate;
