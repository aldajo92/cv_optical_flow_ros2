#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class OpticalFlowNode : public rclcpp::Node
{
public:
    OpticalFlowNode()
        : Node("optical_flow_node"), prev_gray_image_()
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10, std::bind(&OpticalFlowNode::image_callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/optical_flow_image", 10);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat gray_image;
        cv::cvtColor(cv_ptr->image, gray_image, cv::COLOR_BGR2GRAY);

        if (prev_gray_image_.empty())
        {
            prev_gray_image_ = gray_image;
            return;
        }

        std::vector<cv::Point2f> prev_pts, next_pts;
        cv::goodFeaturesToTrack(prev_gray_image_, prev_pts, 100, 0.3, 7);
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray_image_, gray_image, prev_pts, next_pts, status, err);

        for (size_t i = 0; i < next_pts.size(); i++)
        {
            if (status[i])
            {
                cv::line(cv_ptr->image, prev_pts[i], next_pts[i], cv::Scalar(0, 255, 0), 2);
                cv::circle(cv_ptr->image, next_pts[i], 5, cv::Scalar(0, 255, 0), -1);
            }
        }

        prev_gray_image_ = gray_image;

        auto output_msg = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_ptr->image).toImageMsg();
        publisher_->publish(*output_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    cv::Mat prev_gray_image_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OpticalFlowNode>());
    rclcpp::shutdown();
    return 0;
}