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

#include <common_msgs/ObjectPos3D.h>
#include <common_msgs/Detection3D.h>

#include "geometry_msgs/PolygonStamped.h"
#include "geometry_msgs/Point32.h"

using namespace sensor_msgs;
using namespace message_filters;
using namespace common_msgs;

// OpenCV uses BGR format
static const uint8_t color_map[3] = {128, 64, 128};

namespace ros_app_viz
{
    /**
     * @brief  SDE ROS warpper class
     */

    class VizEStop
    {
    public:
        /**
         * @brief { function_description }
         *
         * @param[in]  resolution  The resolution
         * @param[in]  frame_rate  The frame rate
         */

        VizEStop(ros::NodeHandle *nh, ros::NodeHandle *private_nh)
        {
            std::string rectImgTopic;
            std::string bbTopic;
            std::string ssTensorTopic;
            std::string bbImgTopic;

            // input and tensor sizes
            private_nh->param("width",            m_width,        1280);
            private_nh->param("height",           m_height,       720);
            private_nh->param("tensor_width",     m_tensorWidth,  768);
            private_nh->param("tensor_height",    m_tensorHeight, 432);
            private_nh->param("show_ground",      m_showGround,   0);

            // input topics
            private_nh->param("rectified_image_topic",    rectImgTopic,  std::string(""));
            private_nh->param("bounding_box_topic",       bbTopic,       std::string(""));
            private_nh->param("semseg_cnn_tensor_topic",  ssTensorTopic, std::string(""));

            // output topics
            private_nh->param("bounding_box_image_topic", bbImgTopic,    std::string(""));

            m_bbImgPub    = nh->advertise<Image>(bbImgTopic, 1);
            m_safetyPub   = nh->advertise<geometry_msgs::PolygonStamped>("/estop/safety_bubble", 1);

            message_filters::Subscriber<Detection3D> bbSub(*nh,  bbTopic, 1);
            message_filters::Subscriber<Image> rectImgSub(*nh, rectImgTopic, 1);
            message_filters::Subscriber<Image> ssTensorSub(*nh, ssTensorTopic, 1);

            TimeSynchronizer<Detection3D, Image, Image> sync(bbSub, rectImgSub, ssTensorSub, 10);
            sync.registerCallback(boost::bind(&VizEStop::callback_vizEStop, this, _1, _2, _3));

            ros::spin();
        }

        ~VizEStop()
        {

        }

        void callback_vizEStop(const Detection3D::ConstPtr& objMsg,
                               const ImageConstPtr& imagePtr,
                               const ImageConstPtr& tensorPtr)
        {
            // create and publish bounding box images
            publishBBImage(objMsg, imagePtr, tensorPtr);

            // create polygons for zones
            int numPoints    = 12; // number of points for polygon
            float sbDistance = 5.0;
            geometry_msgs::PolygonStamped safetyBubble;

            safetyBubble.header.frame_id = "map";

            for (int i = 0; i < numPoints; ++i)
            {
                  double angle = i * ( 360.0 / (double) numPoints) * M_PI / 180.0;

                  geometry_msgs::Point32 stop_point;
                  stop_point.z = 0.0;
                  stop_point.x = cos(angle) * sbDistance;
                  stop_point.y = sin(angle) * sbDistance;
                  safetyBubble.polygon.points.push_back(stop_point);
            }
            m_safetyPub.publish(safetyBubble);
        }

        void publishBBImage(const Detection3D::ConstPtr& objMsg,
                            const ImageConstPtr& imagePtr,
                            const ImageConstPtr& tensorPtr)
        {
            int32_t i, j, y, x;
            int32_t numObjs;
            char    strDistance[50];

            uint8_t classId;

            // blending factor
            float   bf        = 0.5;
            int32_t stride    = m_width * 3;

            float   horScale  = m_tensorWidth *1.0/m_width;
            float   verScale  = m_tensorHeight*1.0/m_height;

            int     fontface  = cv::FONT_HERSHEY_SIMPLEX;
            int     thickness = 1;
            int     baseline  = 0;
            double  scale     = 0.6;

            cv::Scalar lineColor;
            cv::Point  pos;
            cv::Size   text;

            Detection3D   detectionMsg = *objMsg;
            ObjectPos3D   objPos;
            ObjectPos3D * objPosArr;

            cv_bridge::CvImagePtr cv_bbPtr;
            cv_bbPtr = cv_bridge::toCvCopy(imagePtr,  sensor_msgs::image_encodings::RGB8);

            numObjs = detectionMsg.num_objects;
            objPosArr = (ObjectPos3D *) detectionMsg.obj_pos3d.data();

            /* draw bounding boxes on detected objects */
            for (i = 0; i < numObjs; i++)
            {
                objPos = objPosArr[i];

#if 1
                // Color based on distance
                if (objPos.front_depth <= 5000)
                {
                    lineColor = CV_RGB(0, 0, 255);
                } else
                {
                    lineColor = CV_RGB(255, 0, 0);
                }
#else
                // Color based on class
                if (objPos.classId == 11)
                {
                    lineColor = CV_RGB(255, 0, 0);
                }
                else if (objPos.classId == 12)
                {
                    lineColor = CV_RGB(0, 255, 0);
                }
                else if (objPos.classId == 13)
                {
                    lineColor = CV_RGB(0, 0, 255);
                }
                else if (objPos.classId == 14)
                {
                    lineColor = CV_RGB(0, 255, 255);
                }
                else if (objPos.classId == 15)
                {
                    lineColor = CV_RGB(255, 255, 0);
                }
#endif

                // Front box
                // P3 P4
                // P1 P2
                cv::line(cv_bbPtr->image, cv::Point(objPos.pf1x, objPos.pf1y), cv::Point(objPos.pf2x, objPos.pf2y), lineColor, 2);
                cv::line(cv_bbPtr->image, cv::Point(objPos.pf2x, objPos.pf2y), cv::Point(objPos.pf3x, objPos.pf3y), lineColor, 2);
                cv::line(cv_bbPtr->image, cv::Point(objPos.pf3x, objPos.pf3y), cv::Point(objPos.pf4x, objPos.pf4y), lineColor, 2);
                cv::line(cv_bbPtr->image, cv::Point(objPos.pf4x, objPos.pf4y), cv::Point(objPos.pf1x, objPos.pf1y), lineColor, 2);

                // Front box
                // P3 P4
                // P1 P2
                cv::line(cv_bbPtr->image, cv::Point(objPos.pr1x, objPos.pr1y), cv::Point(objPos.pr2x, objPos.pr2y), lineColor, 2);
                cv::line(cv_bbPtr->image, cv::Point(objPos.pr2x, objPos.pr2y), cv::Point(objPos.pr3x, objPos.pr3y), lineColor, 2);
                cv::line(cv_bbPtr->image, cv::Point(objPos.pr3x, objPos.pr3y), cv::Point(objPos.pr4x, objPos.pr4y), lineColor, 2);
                cv::line(cv_bbPtr->image, cv::Point(objPos.pr4x, objPos.pr4y), cv::Point(objPos.pr1x, objPos.pr1y), lineColor, 2);

                // pf1 - pr1, pf2 - pr2, pf3 - pr3, pf4 - pr4
                cv::line(cv_bbPtr->image, cv::Point(objPos.pf1x, objPos.pf1y), cv::Point(objPos.pr1x, objPos.pr1y), lineColor, 2);
                cv::line(cv_bbPtr->image, cv::Point(objPos.pf2x, objPos.pf2y), cv::Point(objPos.pr2x, objPos.pr2y), lineColor, 2);
                cv::line(cv_bbPtr->image, cv::Point(objPos.pf3x, objPos.pf3y), cv::Point(objPos.pr3x, objPos.pr3y), lineColor, 2);
                cv::line(cv_bbPtr->image, cv::Point(objPos.pf4x, objPos.pf4y), cv::Point(objPos.pr4x, objPos.pr4y), lineColor, 2);

                sprintf(strDistance, "%.1fm", float(objPos.front_depth / 1000));

                text = cv::getTextSize(strDistance, fontface, scale, thickness, &baseline);
                pos  = cv::Point((objPos.pf1x+objPos.pf2x)/2 - 25, (objPos.pf1y+objPos.pf4y)/2);
#if CV_MAJOR_VERSION <= 3
                cv::rectangle(cv_bbPtr->image, pos + cv::Point(0, baseline), pos + cv::Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);
#else
                cv::rectangle(cv_bbPtr->image, pos + cv::Point(0, baseline), pos + cv::Point(text.width, -text.height), CV_RGB(0,0,0), cv::FILLED);
#endif
                cv::putText(cv_bbPtr->image, strDistance, pos, fontface, scale, CV_RGB(255,255,255), thickness);
                /*
                cv::putText(cv_bbPtr->image, strDistance, cv::Point((objPos.pf1x+objPos.pf2x)/2 - 25, (objPos.pf1y+objPos.pf4y)/2),
                            fontface, scale, CV_RGB(255,255,255), thickness);
                */
            }

            if (m_showGround)
            {
                for (j = 0; j < m_height; j++)
                {
                    // y pos in tensor
                    y = (j * verScale + 0.5);
                    for (i = 0; i < m_width; i++)
                    {
                        // x pos in tensor
                        x = (i * horScale + 0.5);
                        classId =  tensorPtr->data[y*m_tensorWidth + x];

                        // road and sidewalk
                        if (classId == 0 || classId == 1)
                        {
                            cv_bbPtr->image.data[j*stride + i*3]     =
                                (1-bf)*cv_bbPtr->image.data[j*stride + i*3]     + bf*color_map[0];
                            cv_bbPtr->image.data[j*stride + i*3 + 1] =
                                (1-bf)*cv_bbPtr->image.data[j*stride + i*3 + 1] + bf*color_map[1];
                            cv_bbPtr->image.data[j*stride + i*3 + 2] =
                                (1-bf)*cv_bbPtr->image.data[j*stride + i*3 + 2] + bf*color_map[2];
                        }
                    }
                }
            }

            m_bbImgPub.publish(cv_bbPtr->toImageMsg());
        }


    private:
        int32_t                         m_width;
        int32_t                         m_height;
        int32_t                         m_tensorWidth;
        int32_t                         m_tensorHeight;
        int32_t                         m_showGround;
        ros::Publisher                  m_bbImgPub;
        ros::Publisher                  m_safetyPub;
    };
}

/**
 * This tutorial demonstrates simple sending of messages over the ROS system.
 */
int main(int argc, char **argv)
{
    try
    {
        ros::init(argc, argv, "app_viz_estop");
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        ros_app_viz::VizEStop estopViz(&nh, &private_nh);

        return EXIT_SUCCESS;
    }
    catch (std::runtime_error& e)
    {
        ros::shutdown();
        return EXIT_FAILURE;
    }

}


