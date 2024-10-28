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
#ifndef _APP_SDE_NODE_H_
#define _APP_SDE_NODE_H_

#include <stdio.h>
#include <thread>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <common_msgs/Disparity.h>
#include <sensor_msgs/PointCloud2.h>

#include <sde.h>

using namespace sensor_msgs;
using namespace message_filters;
using namespace common_msgs;

class SDEAppNode
{
    using ImgSub     = message_filters::Subscriber<Image>;
    using CamInfoSub = message_filters::Subscriber<CameraInfo>;
    using TimeSync   = TimeSynchronizer<Image, Image>;
    using ImgPub     = image_transport::Publisher;
    using ImgTrans   = image_transport::ImageTransport;
    using SubCon     = message_filters::Connection;

    public:
        SDEAppNode(ros::NodeHandle &nodeHdl,
                   ros::NodeHandle &privNodeHdl);
        ~SDEAppNode();

        /** Intercept signal handler */
        void sigHandler(int32_t  sig);

    private:
        /** Initialize App cntxt */ 
        vx_status init();

        /** process input frames */
        void process();

        /** Callback to process camera info data */
        void camInfoCb(const CameraInfoConstPtr& cam_info);

        /** Callback to process input data */
        void imgCb(const ImageConstPtr& left_image_ptr,
                   const ImageConstPtr& right_image_ptr);

        /** Run camera info subscriber thread */
        void subscriberCamInfoThread();

        /** Run publisher thread */
        void publisherThread();

        /** Run iput process thread */
        void inputProcessThread();

        /** OpenVX graph completion event handler */
        void graphCompleteEvtHdlr();

        /** Read yaml algorithm paramters file */
        void readParams();

    private:
        SDEAPP_Context         *m_cntxt{};
        TimeSync               *m_sync{};
        CamInfoSub             *m_sub{};
        ImgTrans               *m_imgTrans{};
        ImgSub                 *m_leftImageSub{};
        ImgSub                 *m_rightImageSub{};
        SubCon                  m_conObj;
        SubCon                  m_camInfoConObj;

        ros::NodeHandle        &m_nodeHdl;
        ros::NodeHandle        &m_privNodeHdl;

        /** Subscriber thread */
        std::thread             m_subThread;

        /** Publisher thread */
        std::thread             m_pubThread;

        /** Input image width */
        uint32_t                m_imgWidth;

        /** Input image height */
        uint32_t                m_imgHeight;

        /** Disparity map publisher */
        Disparity               m_disparityPubData;
        uint32_t                m_disparitySize;
        ros::Publisher          m_disparityPub;

        /** Point Cloud publisher */
        PointCloud2             m_pcPubData;
        uint32_t                m_numPcPoints;
        uint32_t                m_pcSize;
        ros::Publisher          m_pcPub;

        /** Initialize control semaphore */
        Semaphore              *initCtrlSem;
};

#endif /* _APP_SDE_NODE_H_ */

