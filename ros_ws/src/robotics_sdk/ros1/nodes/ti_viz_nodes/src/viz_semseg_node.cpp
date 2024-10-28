/*
 *
 * Copyright (c) 2022 Texas Instruments Incorporated
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
 * licensed and provided to you in object code.
 *
 * If software source code is provided to you, modification and redistribution of the
 * source code are permitted provided that the following conditions are met:
 *
 * *       any redistribution and use of the source code, including any resulting derivative
 * works, are licensed by TI for use only with TI Devices.
 *
 * *       any redistribution and use of any object code compiled from the source code
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
#include <stdio.h>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

using namespace sensor_msgs;
using namespace message_filters;

// OpenCV uses BGR format
static const uint8_t color_map[20][3] =
{{128,  64, 128},{244,  35, 232},{ 70,  70,  70},{102, 102, 156},{190, 153, 153},
 {153, 153, 153},{250, 170,  30},{220, 220,   0},{107, 142,  35},{152, 251, 152},
 { 70, 130, 180},{220,  20,  60},{255,   0,   0},{  0,   0, 142},{  0,   0,  70},
 {  0,  60, 100},{  0,  80, 100},{  0,   0, 230},{119,  11,  32},{128, 128, 128}};


namespace ros_app_viz
{
    /**
     * @brief  VizSemSeg ROS warpper class
     */

    class VizSemSeg
    {
    public:
        /**
         * @brief { Overlays the semantic segmentation tensor on the image }
         *
         */

        VizSemSeg(ros::NodeHandle *nh, ros::NodeHandle *private_nh)
        {
            std::string rectImgTopic;
            std::string ssTensorTopic;
            std::string ssMapImgTopic;

            // input topics
            private_nh->param("rectified_image_topic",    rectImgTopic,  std::string(""));
            private_nh->param("vision_cnn_tensor_topic",  ssTensorTopic, std::string(""));

            // output topics
            private_nh->param("vision_cnn_image_topic",   ssMapImgTopic, std::string(""));

            m_ssMapImgPub = nh->advertise<Image>(ssMapImgTopic, 1);

            message_filters::Subscriber<Image> ssTensorSub(*nh, ssTensorTopic, 1);
            message_filters::Subscriber<Image> rectImgSub(*nh, rectImgTopic, 1);

            TimeSynchronizer<Image, Image> sync(ssTensorSub, rectImgSub, 10);
            sync.registerCallback(boost::bind(&VizSemSeg::callback_vizSemSeg, this, _1, _2));

            ros::spin();
        }

        ~VizSemSeg()
        {

        }

        void callback_vizSemSeg(const ImageConstPtr& tensorPtr,
                                const ImageConstPtr& imagePtr)
        {
            // position in original image
            int32_t i, j;
            // position in tensor
            int32_t y, x;
            // Semantic class
            uint8_t classId;

            // input image size
            int32_t width  = imagePtr->width;
            int32_t height = imagePtr->height;

            // tensor size
            int32_t tensorWidth  = tensorPtr->width;
            int32_t tensorHeight = tensorPtr->height;

            // blending factor
            float bf = 0.5;

            int32_t stride = width * 3;

            float   horScale = tensorWidth *1.0/width;
            float   verScale = tensorHeight*1.0/height;

            cv_bridge::CvImagePtr cv_ssPtr;
            cv_ssPtr = cv_bridge::toCvCopy(imagePtr, sensor_msgs::image_encodings::RGB8);

            for (j = 0; j < height; j++)
            {
                // y pos in tensor
                y = (j * verScale + 0.5);
                for (i = 0; i < width; i++)
                {
                    // x pos in tensor
                    x = (i * horScale + 0.5);
                    classId =  tensorPtr->data[y*tensorWidth + x];

                    cv_ssPtr->image.data[j*stride + i*3]     =
                        (1-bf)*cv_ssPtr->image.data[j*stride + i*3]     + bf*color_map[classId][0];
                    cv_ssPtr->image.data[j*stride + i*3 + 1] =
                        (1-bf)*cv_ssPtr->image.data[j*stride + i*3 + 1] + bf*color_map[classId][1];
                    cv_ssPtr->image.data[j*stride + i*3 + 2] =
                        (1-bf)*cv_ssPtr->image.data[j*stride + i*3 + 2] + bf*color_map[classId][2];
                }
            }

            m_ssMapImgPub.publish(cv_ssPtr->toImageMsg());
        }

    private:
        ros::Publisher m_ssMapImgPub;
    };
}

/**
 * Main
 */
int main(int argc, char **argv)
{
    try
    {
        ros::init(argc, argv, "app_viz_semseg");
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        ros_app_viz::VizSemSeg semSegViz(&nh, &private_nh);

        return EXIT_SUCCESS;
    }
    catch (std::runtime_error& e)
    {
        ros::shutdown();
        return EXIT_FAILURE;
    }

}
