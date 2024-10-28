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

#include "usb_mono_camera.h"

/**
 * @brief Convert YUYV to UYVY
 *
 * @param [in]  src  Source
 * @param [out] dst  Destination
 * @param [in]  num_pixels  Number of pixels of source image
 */
void yuyv2uyvy(unsigned char *src, unsigned char *dst,
               unsigned long long int num_pixels)
{
    swab(src, dst, num_pixels<<1);
}

/**
 * @brief Mono camera driver
 *
 * @param [in] device_name  Device name of the camera
 * @param [in] camera_mode  Camera mode string: "FHD", "HD", or "VGA"
 * @param [in] frame_rate   Frame rate
 * @param [in] encoding     Encoding scheme: "yuv422" (default) or "bgr8"
 */
MonoCamera::MonoCamera(const std::string device_name,
                           const std::string camera_mode,
                           const double frame_rate,
                           const std::string encoding):
    logger(rclcpp::get_logger("MonoCamera"))
{
    camera_   = new cv::VideoCapture(device_name);
    encoding_ = encoding;

    // config the camera
    setCameraMode(camera_mode);
    setFrameRate(frame_rate);

    if (encoding_.compare("yuv422")==0)
    {
        camera_->set(cv::CAP_PROP_CONVERT_RGB, false);
        RCLCPP_INFO(logger, "YUV422");
    }
    else
    {
        camera_->set(cv::CAP_PROP_CONVERT_RGB, true);
        RCLCPP_INFO(logger, "BGR8");
    }

    // set YUYV mode
    camera_->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y','U','Y','V'));

    RCLCPP_INFO(logger, "Stereo Camera Mode %s, width %.0f, height %.0f",
             camera_mode.c_str(), camera_->get(ID_WIDTH), camera_->get(ID_HEIGHT));
}

MonoCamera::~MonoCamera()
{
    delete camera_;
}

/**
 * @brief Set the camera mode
 *
 * @param [in] camera_mode  Camera mode string: "FHD", "HD", or "VGA"
 */
void MonoCamera::setCameraMode(const std::string camera_mode)
{
    if(camera_mode.compare("FHD")==0)
    {
        width_         = 1920;
        height_        = 1080;
        out_width_     = width_;
        out_height_    = height_;
    }
    else if(camera_mode.compare("HD")==0)
    {
        width_         = 1280;
        height_        = 720;
        out_width_     = width_;
        out_height_    = height_;
    }
    else if(camera_mode.compare("VGA")==0)
    {
        width_         = 640;
        height_        = 480;
        out_width_     = width_;
        out_height_    = height_;
    }
    else
    {
        RCLCPP_FATAL(logger, "Unknown camera_mode: %s", camera_mode.c_str());
    }

    camera_->set(ID_WIDTH, width_);
    camera_->set(ID_HEIGHT, height_);
    width_  = camera_->get(ID_WIDTH);
    height_ = camera_->get(ID_HEIGHT);
}

/**
 * @brief Set the frame rate
 *
 * @param [in] frame_rate  Frame rate
 */
void MonoCamera::setFrameRate(const double frame_rate)
{
    camera_->set(ID_FPS, frame_rate);
    frame_rate_ = camera_->get(ID_FPS);
}

/**
 * @brief Capture the images
 *
 * @param [out] img   Captured image
 */
bool MonoCamera::getImages(cv::Mat &img)
{
    if (camera_->grab())
    {
        // note: images are captured in yuv422:YUYV format
        camera_->retrieve(img);
        if (encoding_.compare("yuv422")==0)
        {
            yuyv2uyvy(img.data, img.data, img.total());
        }

        return true;
    }
    else
    {
        return false;
    }
}
