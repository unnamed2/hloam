#include <ros/ros.h>

int main(int argc, char** argv) {
    for(int i = 0; i < argc; i++) {
        ROS_INFO("argv[%d] : %s", i, argv[i]);
    }

    ros::init(argc, argv, "empty_node");
    ros::NodeHandle nh;
    ros::spin();
    printf("Exiting...\r\n");
    return 0;
}
