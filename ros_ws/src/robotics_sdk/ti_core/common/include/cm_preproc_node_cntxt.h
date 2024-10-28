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
#if !defined(_CM_PREPROC_NODE_CNTXT_H_)
#define _CM_PREPROC_NODE_CNTXT_H_

#include <TI/tivx_img_proc.h>
#include <cm_common.h>

/**
 * \defgroup group_ticore_node_preproc Pre-processing node
 * 
 * \brief Set up the pre-processing node
 * 
 * \ingroup group_ticore_node
 *
 */

namespace ti_core_common 
{

/**
 * \brief Constant for representing the max number of channels in the output
 *        image after the pre-processinf phase.
 * \ingroup group_ticore_node_preproc
 */
#define CM_PREPROC_MAX_IMG_CHANS            (3U)

/**
 * \brief Constant for representing the preproc output tensor count.
 * \ingroup group_ticore_node_preproc
 */
#define CM_PREPROC_NUM_OUTPUT_TENSORS       (1U)

/**
 * \brief Constant for representing the preproc output tensor dimension count.
 * \ingroup group_ticore_node_preproc
 */
#define CM_PREPROC_OUTPUT_TENSORS_DIMS      (3U)

/**
 * \brief Constant for representing the preproc node context invalid state.
 * \ingroup group_ticore_node_preproc
 */
#define CM_PREPROC_NODE_CNTXT_STATE_INVALID (0U)

/**
 * \brief Constant for representing the preproc node context initialization
 *        state.
 * \ingroup group_ticore_node_preproc
 */
#define CM_PREPROC_NODE_CNTXT_STATE_INIT    (1U)

/**
 * \brief Constant for representing the preproc node context setup state.
 * \ingroup group_ticore_node_preproc
 */
#define CM_PREPROC_NODE_CNTXT_STATE_SETUP   (2U)

/**
 * \brief Pre-processing node create time parameters.
 *
 * \ingroup group_ticore_node_preproc
 */
typedef struct
{
    /** Pipeline depth. */
    uint8_t                 pipelineDepth;

    /** Output Image width in pixels. */
    vx_uint32               outWidth;

    /** Output Image height in pixels. */
    vx_uint32               outHeight;

    /** Mean values to be used in pre-processing stage. */
    float                   mean[CM_PREPROC_MAX_IMG_CHANS];

    /** Scaling values to be used in pre-processing stage. */
    float                   scale[CM_PREPROC_MAX_IMG_CHANS];

} CM_PreProcCreateParams;

/**
 * \brief Pre-processing node context.
 *
 * \ingroup group_ticore_node_preproc
 */
typedef struct
{
    /** State variable. */
    uint32_t                state;

    /** Handle to the pre-processing node. */
    vx_node                 vxNode;

    /** Handle to the vx object for holding the pre-processing configuration
     *  parameters.
     */
    vx_array                vxConfig;

    /** Image pre-processing configuration parameters. */
    tivxImgPreProcParams    imgParams;

    /** Number of output tensors. */
    uint8_t                 numOutTensors;

    /** Pipeline depth. */
    uint8_t                 pipelineDepth;

    /** Output tensor data format. */
    uint16_t                outDataType;

    /** Array of handles to the tensor objects. The number of handles equals
     *  to numOutTensors.
     */
    vx_tensor              *vxOutputTensor;

} CM_PreProcNodeCntxt;

/**
 * \brief Function to initialize the pre-processing node context.
 *
 * \param [in,out] preProcObj Pre-processing node context.
 *
 * \param [in] context The handle to the openVX implementation context.
 *
 * \param [in] createParams Pre-processing node context create parameters.
 *
 * \return VX_SUCCESS on success
 *
 * \ingroup group_ticore_node_preproc
 */
vx_status CM_preProcNodeCntxtInit(
        CM_PreProcNodeCntxt            *preProcObj,
        vx_context                      context,
        const CM_PreProcCreateParams   *createParams);

/**
 * \brief Function to create and setup the pre-processing node.
 *        
 *        The node should have been initialized by calling
 *        CM_preProcNodeCntxtInit() API, prior to invoking this API.
 *
 * \param [in,out] preProcObj Pre-processing node context.
 *
 * \param [in] context The handle to the openVX implementation context.
 *
 * \param [in] graph The handle to the graph this node belongs to.
 *
 * \param [in] inputImage Input image parameter to the node.
 *
 * \return VX_SUCCESS on success
 *
 * \ingroup group_ticore_node_preproc
 */
vx_status CM_preProcNodeCntxtSetup(
        CM_PreProcNodeCntxt    *preProcObj,
        vx_context              context,
        vx_graph                graph,
        vx_image                inputImage);

/**
 * \brief Function to de-initialize the pre-processing node context.
 *
 * \param [in,out] preProcObj Pre-processing node context.
 *
 * \return VX_SUCCESS on success
 *
 * \ingroup group_ticore_node_preproc
 */
vx_status CM_preProcNodeCntxtDeInit(
        CM_PreProcNodeCntxt    *preProcObj);


} // namespace ti_core_common 

#endif /* _CM_PREPROC_NODE_CNTXT_H_ */

