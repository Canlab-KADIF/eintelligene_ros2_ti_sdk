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

#ifndef _SDE_MULTILAYER_H_
#define _SDE_MULTILAYER_H_

#include <queue>
#include <thread>
#include <mutex>

#include <stdio.h>
#include <stdlib.h>

#include <TI/tivx.h>
#include <TI/tivx_mutex.h>
#include <TI/tivx_stereo.h>

#include <cm_common.h>

/**
 * \defgroup group_applib_sde_multilayer Multi-layer SDE  code.
 * \ingroup group_ptk_applib
 *
 */

#include <utils/perf_stats/include/app_perf_stats.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <tivx_utils_graph_perf.h>
#include <tivx_utils_file_rd_wr.h>

#define ML_SDE_DEFAULT_CORE_MAPPING  (TIVX_TARGET_DSP1)
#define ML_SDE_MAX_LINE_LEN          (256U)

/**
 * \brief Multi-layer SDE  create parameter context.
 *
 * \ingroup group_applib_sde_multilayer
 */
typedef struct
{
    /** OpenVX references */
    vx_context                        vxContext;

    /** OpenVX graph */
    vx_graph                          vxGraph;

    /** SDE params */
    tivx_dmpac_sde_params_t           sdeCfg;

    /** number of layers */
    uint8_t                           numLayers;

    /** input format, U8 or YUV_UYVY  */
    uint8_t                           inputFormat;

    /** median filtering flag */
    uint8_t                           enableMedianFilter;

    /** Input width */
    uint16_t                          width;

    /** Input height */
    uint16_t                          height;

    /** minimum Disparity */
    uint16_t                          minDisparity;

    /** maximum Disparity */
    uint16_t                          maxDisparity;

    /** Pipeline depth, 0, 1, ..., ML_SDE_MAX_PIPELINE_DEPTH */
    uint8_t                           pipelineDepth;

    /** Input object pipeline depth, 
     *  which is the number of each input OpenVX object
     */
    uint8_t                           inputPipelineDepth;

    /** Flag indicating whether or not create input OpenVX object in applib */
    uint8_t                           createInputFlag;

    /** Flag indicating whether or not create output OpenVX object in applib */
    uint8_t                           createOutputFlag;

    /** Input recitifed left image object */
    vx_image                          vxLeftRectImageL0[GRAPH_MAX_PIPELINE_DEPTH];

    /** Input recitifed right image object */
    vx_image                          vxRightRectImageL0[GRAPH_MAX_PIPELINE_DEPTH];

    /** Output image object */
    vx_image                          vxMergeDisparityL0[GRAPH_MAX_PIPELINE_DEPTH];

    /** Output image object */
    vx_image                          vxMedianFilteredDisparity[GRAPH_MAX_PIPELINE_DEPTH];

    /** Disparity Merge Node core mapping */
    const char                       *dispMergeNodeCore;

    /** Hole Filling Node core mapping */
    const char                       *holeFillingNodeCore;

    /** Median Filter Node core mapping */
    const char                       *medianFilterNodeCore;

    /** Disparity Viz Node Core mapping. */
    const char                       *dispVisNodeCore;

} ML_SDE_createParams;

struct ML_SDE_Context;
typedef ML_SDE_Context * ML_SDE_Handle;

/**
 * \brief Function to initialize the .
 *
 * \param [in] createParams  create parameters.
 *
 * \return Valid handle on success. NULL otherwise.
 *
 * \ingroup group_applib_sde_multilayer
 */
ML_SDE_Handle ML_SDE_create(ML_SDE_createParams *createParams);

/**
 * \brief Function to de-init the  and release the memory associated with
 *        the handle.
 *
 * \param [in,out] handle Reference to  handle.
 *
 * \ingroup group_applib_sde_multilayer
 */
void                ML_SDE_delete(ML_SDE_Handle *handle);

/**
 * \brief Function to initialize Scalers' coefficients
 *
 * \param [in,out] handle Reference to  handle.
 *
 * \return VX_SUCCESS on success
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_status           ML_SDE_initScaler(ML_SDE_Handle handle);

/**
 * \brief Function to get the base-layer (layer 0) SDE node handle.
 *        This is needed when a graph is created outside . 
 *        The caller calls this function to put this node into the graph.
 *
 * \param [in] handle Reference to  handle.
 *
 * \return SDE node
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_node             ML_SDE_getSDENodeL0(ML_SDE_Handle handle);

/**
 * \brief Function to get the second-layer (layer 1) SDE node handle
 *        This is needed when a graph is created outside . 
 *        The caller calls this function to put this node into the graph.
 *
 * \param [in] handle Reference to  handle.
 *
 * \return SDE node
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_node             ML_SDE_getSDENodeL1(ML_SDE_Handle handle);

/**
 * \brief Function to get the third-layer (layer 2) SDE node handle. 
 *        This is needed when a graph is created outside . 
 *        The caller calls this function to put this node into the graph.
 *
 * \param [in] handle Reference to  handle.
 *
 * \return SDE node
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_node             ML_SDE_getSDENodeL2(ML_SDE_Handle handle);

/**
 * \brief Function to get the post-processing median fitler node.
 *        This is needed when a graph is created outside . 
 *        The caller calls this function to put this node into the graph.
 * 
 * \param [in] handle Reference to  handle.
 *
 * \return Median filter node
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_node             ML_SDE_getMedFilterNode(ML_SDE_Handle handle);

/**
 * \brief Function to get the merge node that combine disparity maps at
 *        layer 0 and layer 1. This is needed when a graph is
 *        created outside . The caller calls this function  
 *        to put this node into the graph.
 * 
 * \param [in] handle Reference to  handle.
 *
 * \return Merge node
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_node             ML_SDE_getMergeNodeL1(ML_SDE_Handle handle);

/**
 * \brief Function to get the merge node that combine disparity maps at
 *        layer 1 and layer 2. This is needed when a graph is created outside .
 *        The caller calls this function to put this node into the graph.
  * 
 * \param [in] handle Reference to  handle.
 *
 * \return Merge node
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_node             ML_SDE_getMergeNodeL2(ML_SDE_Handle handle);

/**
 * \brief Function to get the Scaler node that create down-sampled layer-1 
 *        left image from layer-0 left image. This is needed when a graph is
 *        created outside . The caller calls this function  
 *        to put this node into the graph. 
 * 
 * \param [in] handle Reference to  handle.
 *
 * \return Scaler node
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_node             ML_SDE_getLeftMSCNodeL1(ML_SDE_Handle handle);

/**
 * \brief Function to get the Scaler node that create down-sampled layer-1 
 *        right image from layer-0 right image. This is needed 
 *        when a graph is created outside . The caller calls this function  
 *        to put this node into the graph. 
 * 
 * \param [in] handle Reference to  handle.
 *
 * \return Scaler node
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_node             ML_SDE_getRightMSCNodeL1(ML_SDE_Handle handle);

/**
 * \brief Function to get the Scaler node that create down-sampled layer-2 
 *        left image from layer-1 left image. This is needed 
 *        when a graph is created outside . The caller calls this function  
 *        to put this node into the graph.
 * 
 * \param [in] handle Reference to  handle.
 *
 * \return Scaler node
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_node             ML_SDE_getLeftMSCNodeL2(ML_SDE_Handle handle);

/**
 * \brief Function to get the Scaler node that create down-sampled layer-2 
 *        right image from layer-1 right image. This is needed 
 *        when a graph is created outside . The caller calls this function 
 *        to put this node into the graph. 
 * 
 * \param [in] handle Reference to  handle.
 *
 * \return Scaler node
 *
 * \ingroup group_applib_sde_multilayer
 */
vx_node             ML_SDE_getRightMSCNodeL2(ML_SDE_Handle handle);

#ifdef __cplusplus
}
#endif

#endif /* _SDE_MULTILAYER_H_ */

