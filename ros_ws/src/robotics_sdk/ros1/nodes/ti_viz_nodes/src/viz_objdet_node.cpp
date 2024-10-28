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
#include <common_msgs/Detection2D.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "viz_objdet.h"

using namespace sensor_msgs;
using namespace common_msgs;
using namespace message_filters;
using namespace cv;

static cv::Mat overlayBoundingBox(cv::Mat &img,
                                  const int32_t *box,
                                  const int32_t label_id)
{
    int32_t label_id_mod = label_id % 20;
    Scalar box_color = Scalar(color_map[label_id_mod][0],
                              color_map[label_id_mod][1],
                              color_map[label_id_mod][2]);
    Scalar text_color    = Scalar(244,  35, 232);
    Scalar text_bg_color = Scalar(120, 120, 120);
    std::string label = classnames_coco[label_id];

    // Draw bounding box for the detected object
    Point topleft     = Point(box[0], box[1]);
    Point bottomright = Point(box[2], box[3]);
    rectangle(img, topleft, bottomright, box_color, 3);

    // Draw text with detected class with a background box
    Point center        = Point((box[0] + box[2])/2, (box[1] + box[3])/2);
    Point t_topleft     = center + Point(-5, -10);
    Point t_bottomright = center + Point( 80, 10);
    rectangle(img, t_topleft, t_bottomright, text_bg_color, -1);
    putText(img, label, t_topleft + Point(10, 15), FONT_HERSHEY_SIMPLEX, 0.6, text_color);

    return img;
}

namespace ros_app_viz
{
    /**
     * @brief  VizObjDet ROS wrapper class
     */

    class VizObjDet
    {
    public:
        /**
         * @brief { Overlays the bounding boxes on the image }
         *
         */

        VizObjDet(ros::NodeHandle *nh, ros::NodeHandle *private_nh)
        {
            std::string rectImgTopic;
            std::string tensorTopic;
            std::string odMapImgTopic;

            // input topics
            private_nh->param("rectified_image_topic", rectImgTopic, std::string(""));
            private_nh->param("vision_cnn_tensor_topic", tensorTopic, std::string(""));

            // output topics
            private_nh->param("vision_cnn_image_topic", odMapImgTopic, std::string(""));

            m_odMapImgPub = nh->advertise<Image>(odMapImgTopic, 1);

            message_filters::Subscriber<Detection2D> tensorSub(*nh, tensorTopic, 1);
            message_filters::Subscriber<Image> rectImgSub(*nh, rectImgTopic, 1);

            TimeSynchronizer<Detection2D, Image> sync(tensorSub, rectImgSub, 10);
            sync.registerCallback(boost::bind(&VizObjDet::callback_vizObjDet, this, _1, _2));

            ros::spin();
        }

        ~VizObjDet()
        {

        }

        void callback_vizObjDet(const Detection2D::ConstPtr& detMsg,
                                const ImageConstPtr& imagePtr)
        {

            cv_bridge::CvImagePtr cv_outImgPtr;
            cv_outImgPtr = cv_bridge::toCvCopy(imagePtr, sensor_msgs::image_encodings::RGB8);

            int32_t box[4];
            int32_t class_id;
            int32_t num_objects = detMsg->num_objects;

            for (int32_t i = 0; i < num_objects; i++)
            {
                box[0] = detMsg->bounding_boxes[i].xmin;
                box[1] = detMsg->bounding_boxes[i].ymin;
                box[2] = detMsg->bounding_boxes[i].xmax;
                box[3] = detMsg->bounding_boxes[i].ymax;
                class_id = detMsg->bounding_boxes[i].label_id;

                overlayBoundingBox(cv_outImgPtr->image, box, class_id);
            }

            m_odMapImgPub.publish(cv_outImgPtr->toImageMsg());
        }

    private:
        ros::Publisher m_odMapImgPub;
    };
}

/**
 * Main
 */
int main(int argc, char **argv)
{
    try
    {
        ros::init(argc, argv, "app_viz_objdet");
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        ros_app_viz::VizObjDet objDetViz(&nh, &private_nh);

        return EXIT_SUCCESS;
    }
    catch (std::runtime_error& e)
    {
        ros::shutdown();
        return EXIT_FAILURE;
    }
}
