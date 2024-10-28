/*
 *  Copyright (C) 2022 Texas Instruments Incorporated - http://www.ti.com/
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <ros/ros.h>
#include <bits/stdc++.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/Point.h>
#include <common_msgs/BoundingBox2D.h>
#include <common_msgs/Detection2D.h>
#include <common_msgs/Disparity.h>
#include <common_msgs/Detection2Dp.h>
#include <cmath>

// PCL
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#define NUM_FRAC_BITS 4
#define CLASS_ID_PATCH 92

using namespace sensor_msgs;
using namespace message_filters;
using namespace common_msgs;


struct FilteredPatch
{
    float fltDisp;
    float cx;
    float cy;
    float validPixelRatio;
    // float posx;
    // float posy;
    // float posz;
    // float range;
};


/**
 * @brief  ObjdetDisparityFusion ROS node class
 */

class ObjdetDisparityFusion
{
public:
    /**
     * @brief { Calculate and add spatial information to each bounding box
     *          and horizontally scan the disparity map to generate point cloud }
     *
     */
    ObjdetDisparityFusion(ros::NodeHandle *nh,
                          ros::NodeHandle *private_nh);

    ~ObjdetDisparityFusion();

    void callback_ObjdetDisparityFusion(const Disparity::ConstPtr& rawDispPtr,
                                        const Detection2D::ConstPtr& detPtr,
                                        const CameraInfo::ConstPtr& camInfoPtr);

    geometry_msgs::Point32 calcBboxPosition(const BoundingBox2D &bbox,
                                            const uint16_t *rawDisparity,
                                            const int32_t dispWidth,
                                            const int32_t dispHeight);

    float extractProcessPatch(const uint16_t *rawDisparity,
                              const int32_t idxStart,
                              const int32_t rowStep,
                              int32_t& outValidPix);

    geometry_msgs::Point32 disparity2Position(const float& disp,
                                              const float& img_x,
                                              const float& img_y);

    float point2Range(const geometry_msgs::Point32& point);

private:
    ros::Publisher m_Detection2Dp_pub;
    std::string    m_rawDisparityTopic;
    std::string    m_tensorTopic;
    std::string    m_outputTopic;
    std::string    m_cameraInfoTopic;
    int            m_bandHeight;
    float          m_baseline;
    float          m_focalLength;
    float          m_distCenterX;
    float          m_distCenterY;
    int            m_confidenceTh;
    float          m_scale = 1.0 / (1 << NUM_FRAC_BITS);
    int            m_disparityFilter;
    float          m_validPixRatioTh;

    // bool           m_horizScanPcl;
    ros::Publisher m_pcl_pub;
    std::string    m_pclTopic;
    std::string    m_pclFrame;
    int32_t        m_disparityWidth;
    int32_t        m_pixelsToExclude;
    int32_t        m_patchSize;
    int32_t        m_patchStride;
    int32_t        m_patchCols;
    int32_t        m_patchRowToUse;
    int            m_pclDisparityFilter;
    FilteredPatch* m_fltPatch;
    float*         m_fltDispModified;

};
