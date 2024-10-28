/*
 *
 * Copyright (c) 2021 Texas Instruments Incorporated
 *
 * All rights reserved not granted herein.
 *
 * Limited License.
 *
 * Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
 * license under copyrights and patents it now or hereafter owns or controls to make,
 * have made, use, import, offer to sell and sell ("Utilize") this software subject to the
 * terms herein.  With respect to the foregoing patent license, such license is granted
 * solely to the extent that any such patent is necessary to Utilize the software alone.
 * The patent license shall not apply to any combinations which include this software,
 * other than combinations with devices manufactured by or for TI ("TI Devices").
 * No hardware patent is licensed hereunder.
 *
 * Redistributions must preserve existing copyright notices and reproduce this license
 * (including the above copyright notice and the disclaimer and (if applicable) source
 * code license limitations below) in the documentation and/or other materials provided
 * with the distribution
 *
 * Redistribution and use in binary form, without modification, are permitted provided
 * that the following conditions are met:
 *
 * *       No reverse engineering, decompilation, or disassembly of this software is
 * permitted with respect to any software provided in binary form.
 *
 * *       any redistribution and use are licensed by TI for use only with TI Devices.
 *
 * *       Nothing shall obligate TI to provide you with source code for the software
 * licensed and provided to you in appCntxtect code.
 *
 * If software source code is provided to you, modification and redistribution of the
 * source code are permitted provided that the following conditions are met:
 *
 * *       any redistribution and use of the source code, including any resulting derivative
 * works, are licensed by TI for use only with TI Devices.
 *
 * *       any redistribution and use of any appCntxtect code compiled from the source code
 * and any resulting derivative works, are licensed by TI for use only with TI Devices.
 *
 * Neither the name of Texas Instruments Incorporated nor the names of its suppliers
 *
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * DISCLAIMER.
 *
 * THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "zed_capture_node.h"

static ZedCameraNode *zed_ros_node;

/**
 * @brief ZED camera ROS node
 *
 * @param [in] resolution  Resolution
 * @param [in] frame_rate  Frame rate
 */
ZedCameraNode::ZedCameraNode(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
{
    // parse ROS parameters
    private_nh.param("camera_mode", camera_mode_, std::string("HD"));
    private_nh.param("frame_rate", frame_rate_, 15.0);
    private_nh.param("frame_id_left",  frame_id_left_, std::string("left_camera"));
    private_nh.param("frame_id_right", frame_id_right_, std::string("right_camera"));
    private_nh.param("device_name", device_name_, std::string("/dev/video1"));
    private_nh.param("encoding", encoding_, std::string("yuv422"));
    private_nh.param("image_topic_left",  image_topic_left_,  std::string("camera/left/image_raw"));
    private_nh.param("image_topic_right", image_topic_right_, std::string("camera/right/image_raw"));
    private_nh.param("camera_info_topic_left",  camera_info_topic_left_,  std::string("camera/left/camera_info"));
    private_nh.param("camera_info_topic_right", camera_info_topic_right_, std::string("camera/right/camera_info"));
    private_nh.param("camera_info_left_yaml",  camera_info_left_yaml_,  std::string(""));
    private_nh.param("camera_info_right_yaml", camera_info_right_yaml_, std::string(""));

    // validate frame rate and update if necessary
    validateFrameRate(camera_mode_, frame_rate_);

    ROS_INFO("Initialize the ZED camera");
    StereoCamera zed(device_name_, camera_mode_, frame_rate_, encoding_);

    // setup publisher
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub_image_left  = it.advertise(image_topic_left_, 1);
    image_transport::Publisher pub_image_right = it.advertise(image_topic_right_, 1);
    ros::Publisher pub_caminfo_left  = nh.advertise<sensor_msgs::CameraInfo>(camera_info_topic_left_, 1);
    ros::Publisher pub_caminfo_right = nh.advertise<sensor_msgs::CameraInfo>(camera_info_topic_right_, 1);

    // populate camera_info
    ROS_INFO("Loading camera info from yaml files");
    sensor_msgs::CameraInfo caminfo_left;
    sensor_msgs::CameraInfo caminfo_right;
    ros::NodeHandle nh_left("left");
    ros::NodeHandle nh_right("right");
    camera_info_manager::CameraInfoManager caminfo_left_manager(nh_left,
        "camera/left", "package://zed_capture/config/"+camera_info_left_yaml_);
    camera_info_manager::CameraInfoManager caminfo_right_manager(nh_right,
        "camera/right", "package://zed_capture/config/"+camera_info_right_yaml_);
    caminfo_left  = caminfo_left_manager.getCameraInfo();
    caminfo_right = caminfo_right_manager.getCameraInfo();
    caminfo_left.header.frame_id  = frame_id_left_;
    caminfo_right.header.frame_id = frame_id_right_;

    // capture and publish images
    cv::Mat   img_left;
    cv::Mat   img_right;
    ros::Rate framerate(frame_rate_);

    while (nh.ok())
    {
        if (!zed.getImages(img_left, img_right))
        {
            ROS_INFO_ONCE("Cannot find the ZED camera");
        }
        else
        {
            ROS_INFO_ONCE("Successfully found the ZED camera");
        }

        ros::Time now = ros::Time::now();
        publishImage(pub_image_left, img_left, "left_frame", now);
        publishImage(pub_image_right, img_right, "right_frame", now);
        publishCamInfo(pub_caminfo_left, caminfo_left, now);
        publishCamInfo(pub_caminfo_right, caminfo_right, now);

        framerate.sleep();
    }
}

ZedCameraNode::~ZedCameraNode() { }

/**
 * @brief Publish the camera_info
 *
 * @param [in] pub_cam_info  camera_info publisher
 * @param [in] cam_info_msg  camera_info message
 * @param [in] tstamp        timestamp
 */
void ZedCameraNode::publishCamInfo(const ros::Publisher &pub_cam_info,
                                   sensor_msgs::CameraInfo &cam_info_msg,
                                   ros::Time tstamp)
{
    cam_info_msg.header.stamp = tstamp;
    pub_cam_info.publish(cam_info_msg);
}

/**
 * @brief Publish the image
 *
 * @param [in]  pub_img       Image publisher
 * @param [in]  img           Image message
 * @param [in]  img_frame_id  Image frame identifier
 * @param [in]  tstamp        Timestamp
 * @param [in]  encoding      image_transport encoding method
 */
void ZedCameraNode::publishImage(const image_transport::Publisher &pub_img,
                                 const cv::Mat &img,
                                 const std::string &img_frame_id,
                                 ros::Time tstamp)
{
    cv_bridge::CvImage cv_img;
    cv_img.image           = img;
    cv_img.encoding        = encoding_;
    cv_img.header.frame_id = img_frame_id;
    cv_img.header.stamp    = tstamp;
    pub_img.publish(cv_img.toImageMsg());
}

/**
 * @brief Validate the frame rate and update it if necessary
 *
 * @param [in] camera_mode Camera mode string
 * @param [in] frame_rate  Camera frame rate
 */
void ZedCameraNode::validateFrameRate(std::string camera_mode, double& frame_rate)
{
    double max_frame_rate;

    if(camera_mode.compare("2K")==0)
        max_frame_rate= 15;
    else if(camera_mode.compare("FHD")==0)
        max_frame_rate= 30;
    else if(camera_mode.compare("FHD2")==0)
        max_frame_rate= 30;
    else if(camera_mode.compare("HD")==0)
        max_frame_rate= 60;
    else if(camera_mode.compare("HD2")==0)
        max_frame_rate= 30;
    else if(camera_mode.compare("VGA")==0)
        max_frame_rate= 100;
    else
        ROS_FATAL("Unknow camera_mode passed");

    if (frame_rate > max_frame_rate)
    {
        ROS_WARN("frame_rate = %f exceeds the max rate with %s, set to %f",
                 frame_rate, camera_mode.c_str(), max_frame_rate);
        frame_rate = max_frame_rate;
    }
}

int main(int argc, char** argv)
{
    try
    {
        ros::init(argc, argv, "zed_camera");
        ros::NodeHandle nh("");
        ros::NodeHandle private_nh("~");

        zed_ros_node = new ZedCameraNode(nh, private_nh);
        return EXIT_SUCCESS;
    }
    catch (std::runtime_error& e)
    {
        ros::shutdown();
        return EXIT_FAILURE;
    }
}
