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

        // Compute Dense Optical Flow using Farneback
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prev_gray_image_, gray_image, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // Convert Optical Flow to HSV representation
        cv::Mat flow_image = draw_optical_flow(flow);

        prev_gray_image_ = gray_image;

        // Convert to ROS Image Message
        auto output_msg = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, flow_image).toImageMsg();
        publisher_->publish(*output_msg);
    }

    cv::Mat draw_optical_flow(const cv::Mat &flow)
    {
        // Create HSV image where hue represents direction, and brightness represents magnitude
        cv::Mat hsv(flow.size(), CV_8UC3);

        // Split flow into x and y components
        std::vector<cv::Mat> flow_components(2);
        cv::split(flow, flow_components); // flow_components[0] -> x, flow_components[1] -> y

        // Compute magnitude and angle
        cv::Mat magnitude, angle;
        cv::cartToPolar(flow_components[0], flow_components[1], magnitude, angle, true);

        // Normalize magnitude to range [0, 255]
        double mag_max;
        cv::minMaxLoc(magnitude, nullptr, &mag_max);
        if (mag_max > 0) // Avoid division by zero
            magnitude.convertTo(magnitude, CV_8UC1, 255.0 / mag_max);
        else
            magnitude.setTo(cv::Scalar(0));

        // Convert angle to hue (scale from [0,360] to [0,180])
        cv::Mat hue;
        angle.convertTo(hue, CV_8UC1, 180.0 / 360.0);

        // Create saturation channel (full intensity)
        cv::Mat saturation = cv::Mat::ones(hue.size(), CV_8UC1) * 255;

        // Merge HSV channels correctly
        std::vector<cv::Mat> hsv_channels = {hue, saturation, magnitude};
        cv::merge(hsv_channels, hsv);

        // Convert HSV to BGR for visualization
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        return bgr;
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
