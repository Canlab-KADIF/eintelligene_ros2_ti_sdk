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
#ifndef _CM_POSE_VIZ_CNTXT_H
#define _CM_POSE_VIZ_CNTXT_H


#include "itidl_ti.h"
#include <TI/tivx_img_proc.h>
#include <cm_common.h>

/**
 * \defgroup group_ticore_node_poseviz  Pose Visualization node
 *
 * \brief Set up the Pose Visualization node
 * 
 * \ingroup  group_ticore_node
 *
 */

namespace ti_core_common 
{

#define APP_POSE_VIZ_MAX_IMAGES (8)


/**
 * \brief Constant for representing the Pose Calc node context invalid state.
 * \ingroup group_ticore_node_poseviz
 */
#define CM_POSEVIZ_NODE_CNTXT_STATE_INVALID     (0U)

/**
 * \brief Constant for representing the Pose Calc node context initialization
 *        state. 
 * \ingroup group_ticore_node_poseviz
 */
#define CM_POSEVIZ_NODE_CNTXT_STATE_INIT        (1U)

/**
 * \brief Constant for representing the Pose Calc node context setup state.
 * \ingroup group_ticore_node_poseviz
 */
#define CM_POSEVIZ_NODE_CNTXT_STATE_SETUP       (2U)


/**
 * \brief Pose visualize node create time parameters.
 * \ingroup group_ticore_node_poseviz
 */
typedef struct
{
    /** Pipeline depth. */
    uint8_t                 pipelineDepth;

    /** Image width in pixels. */
    uint32_t                outWidth;

    /** Image height in pixels. */
    uint32_t                outHeight;

    /** Top-down view image path */
    char                    topViewImgPath[CM_MAX_FILE_LEN];

} CM_PoseVizCreateParams;

/**
 * \brief   Pose visualize node context
 * \ingroup group_ticore_node_poseviz
 */
typedef struct 
{
    /** State variable. */
    uint32_t                state;

    /** Pipeline depth. */
    uint8_t                 pipelineDepth;

    /** Node */
    vx_node                 node;

    /** Configuration object */
    vx_array                config;

    /** Background (top-view) image object */
    vx_image                bgImage; 

    /** Output image object */
    vx_image               *outputImage;

    /** Graph parameter index */
    vx_int32                graph_parameter_index;
    
    /** File path to top view image */
    const char             *topViewImgPath;

    /** Pose visualization config params */
    tivxPoseVizParams       vizParams;

} CM_PoseVizNodeCntxt;


/**
 * \brief Function to initialize the Pose Visualization node.
 *
 * \param [in,out] poseVizObj Pose Visualization node context.
 * 
 * \param [in] context The handle to the openVX implementation context.
 * 
 * \param [in] createParams Pose Visualization node context create parameters.
 * 
 * \return VX_SUCCESS on success
 *
 * \ingroup group_ticore_node_poseviz
 */
vx_status CM_poseVizNodeCntxtInit(CM_PoseVizNodeCntxt           *poseVizObj,
                                  vx_context                     context,
                                  const CM_PoseVizCreateParams  *createParams);

/**
 * \brief Function to create and setup the Pose Visualization node.
 *
 * \param [in,out] poseVizObj Pose Visualization node context.
 *
 * \param [in] context The handle to the openVX implementation context.
 * 
 * \param [in] graph The handle to the graph this node belongs to.
 * 
 * \param [in] pose pose matrxix object.
 * 
 * \return VX_SUCCESS on success
 *
 * \ingroup group_ticore_node_poseviz
 */
vx_status CM_poseVizNodeCntxtSetup(CM_PoseVizNodeCntxt    *poseVizObj,
                                   vx_context              context,
                                   vx_graph                graph,
                                   vx_matrix               pose);

/**
 * \brief Function to de-inmitialize the Pose Visualization node.
 *
 * \param [in,out] poseVizObj Pose Visualization node context.
 * 
 * \return VX_SUCCESS on success
 *
 * \ingroup group_ticore_node_poseviz
 */
vx_status CM_poseVizNodeCntxtDeInit(CM_PoseVizNodeCntxt   *poseVizObj);

} // namespace ti_core_common 

#endif // _CM_POSE_VIZ_CNTXT_H
