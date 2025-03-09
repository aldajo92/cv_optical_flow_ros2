#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// Horn-Schunck Optical Flow Function
void horn_schunck(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &u, cv::Mat &v, float alpha = 1.0, int num_iterations = 100)
{
    // Convert images to float
    cv::Mat I1f, I2f;
    I1.convertTo(I1f, CV_32F);
    I2.convertTo(I2f, CV_32F);

    // Compute Image Gradients (Ensure CV_32F type)
    cv::Mat Ix, Iy, It;
    cv::Sobel(I1f, Ix, CV_32F, 1, 0, 3);
    cv::Sobel(I1f, Iy, CV_32F, 0, 1, 3);
    It = I2f - I1f; // Temporal Gradient

    // Initialize Flow Fields (Ensure CV_32F type)
    u = cv::Mat::zeros(I1.size(), CV_32F);
    v = cv::Mat::zeros(I1.size(), CV_32F);

    // Averaging Kernel
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) / 12.0;

    for (int iter = 0; iter < num_iterations; iter++)
    {
        // Compute Averaged Flow (Ensure CV_32F)
        cv::Mat u_avg, v_avg;
        cv::filter2D(u, u_avg, -1, kernel);
        cv::filter2D(v, v_avg, -1, kernel);

        // Compute Update Terms (Ensure consistent type in arithmetic operations)
        cv::Mat numerator = Ix.mul(u_avg) + Iy.mul(v_avg) + It;
        cv::Mat denominator = (alpha * alpha + Ix.mul(Ix) + Iy.mul(Iy));

        // Ensure denominator is not zero to avoid division errors
        denominator.setTo(1e-6, denominator == 0);

        cv::Mat update = numerator / denominator;

        // Update Flow Fields
        u = u_avg - Ix.mul(update);
        v = v_avg - Iy.mul(update);
    }
}

class HornSchunckOpticalFlowNode : public rclcpp::Node
{
public:
    HornSchunckOpticalFlowNode()
        : Node("horn_schunck_optical_flow_node"), prev_gray_image_()
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10, std::bind(&HornSchunckOpticalFlowNode::image_callback, this, std::placeholders::_1));
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

        // Convert to grayscale
        cv::Mat gray_image;
        cv::cvtColor(cv_ptr->image, gray_image, cv::COLOR_BGR2GRAY);

        if (prev_gray_image_.empty())
        {
            prev_gray_image_ = gray_image;
            return;
        }

        // Compute Dense Optical Flow using Horn-Schunck
        cv::Mat u, v;
        horn_schunck(prev_gray_image_, gray_image, u, v);

        // Convert flow to a visualization image
        cv::Mat flow_image = draw_optical_flow(u, v);

        // Update previous frame
        prev_gray_image_ = gray_image;

        // Convert to ROS Image Message and publish
        auto output_msg = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, flow_image).toImageMsg();
        publisher_->publish(*output_msg);
    }

    cv::Mat draw_optical_flow(const cv::Mat &u, const cv::Mat &v)
    {
        cv::Mat hsv(u.size(), CV_8UC3);

        // Compute magnitude and angle
        cv::Mat magnitude, angle;
        cv::cartToPolar(u, v, magnitude, angle, true);

        // Normalize magnitude to range [0, 255]
        double mag_max;
        cv::minMaxLoc(magnitude, nullptr, &mag_max);
        if (mag_max > 0)
            magnitude.convertTo(magnitude, CV_8UC1, 255.0 / mag_max);
        else
            magnitude.setTo(cv::Scalar(0));

        // Convert angle to hue (scale from [0,360] to [0,180])
        cv::Mat hue;
        angle.convertTo(hue, CV_8UC1, 180.0 / 360.0);

        // Create saturation channel (full intensity)
        cv::Mat saturation = cv::Mat::ones(hue.size(), CV_8UC1) * 255;

        // Merge HSV channels
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
    rclcpp::spin(std::make_shared<HornSchunckOpticalFlowNode>());
    rclcpp::shutdown();
    return 0;
}
