#include<ros/ros.h>
#include<std_msgs/String.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <opencv2/opencv.hpp>

#include "SPextractor.h"

using namespace SuperPoint;

class ImageGrabber
{
public:
    ImageGrabber(){};

    void GrabImage(const sensor_msgs::ImageConstPtr &msg);

    SPextractor spExtractor = SPextractor("superpoint.pt", 0.015, true, 8, 0.5, true);;

    int total_imgCnt = 0;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SuperPoint_cplus14_ROS");
    ros::start();
    ImageGrabber igb;
    ros::NodeHandle nodeHandler;
    ros::Subscriber sub = nodeHandler.subscribe("/camera/color/image_raw", 1, &ImageGrabber::GrabImage, &igb);
    ros::spin();
    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr &msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat img = cv_ptr->image;
    int width = img.cols;
    int height = img.rows;
    cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
    img.convertTo(img, CV_32F, 1.0f / 255.0f);
    total_imgCnt += 1;
    spExtractor.Run(img, to_string(total_imgCnt));
}