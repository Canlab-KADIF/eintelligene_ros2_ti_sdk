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

#ifndef _SDE_SINGLELAYER_H_
#define _SDE_SINGLELAYER_H_

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
 * \defgroup group_applib_sde_singlelayer Single-layer SDE  code.
 * \ingroup group_ptk_applib
 *
 */

#include <utils/perf_stats/include/app_perf_stats.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <tivx_utils_graph_perf.h>
#include <tivx_utils_file_rd_wr.h>

#define SL_SDE_DEFAULT_CORE_MAPPING  (TIVX_TARGET_DSP1)
#define SL_SDE_MAX_LINE_LEN          (256U)

/**
 * \brief Single-layer SDE  create parameter context.
 *
 * \ingroup group_applib_sde_singlelayer
 */
typedef struct
{
    /** OpenVX references */
    vx_context                        vxContext;

    /** OpenVX graph */
    vx_graph                          vxGraph;

    /** SDE params */
    tivx_dmpac_sde_params_t           sdeCfg;

    /** Input width */
    uint16_t                          width;

    /** Input height */
    uint16_t                          height;

    /** Input format, U8 or YUV_UYVY */
    uint8_t                           inputFormat;

    /** disparity map confidence threshold for visualization */
    uint8_t                           vis_confidence;

    /** minimum Disparity */
    uint16_t                          minDisparity;

    /** maximum Disparity */
    uint16_t                          maxDisparity;

    /** Pipeline depth, 0, 1, ..., SL_SDE_MAX_PIPELINE_DEPTH */
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
    vx_image                          vxLeftRectImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Input recitifed right image object */
    vx_image                          vxRightRectImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Output image object */
    vx_image                          vxSde16BitOutput[GRAPH_MAX_PIPELINE_DEPTH];

    /** Disparity Viz Node Core mapping. */
    const char                       *dispVisNodeCore;

} SL_SDE_createParams;

struct  SL_SDE_Context;
typedef SL_SDE_Context * SL_SDE_Handle;


/**
 * \brief Function to initialize the .
 *
 * \param [in] createParams  create parameters.
 *
 * \return Valid handle on success. NULL otherwise.
 *
 * \ingroup group_applib_sde_singlelayer
 */
SL_SDE_Handle SL_SDE_create(SL_SDE_createParams *createParams);

/**
 * \brief Function to de-init the  and release the memory associated with
 *        the handle.
 *
 * \param [in,out] handle Reference to  handle.
 *
 * \ingroup group_applib_sde_singlelayer
 */
void                SL_SDE_delete(SL_SDE_Handle *handle);

/**
 * \brief Function to get the SDE node handle. This is needed when a graph 
 *        is created outside . The caller calls this function to put this node 
 *        into the graph. 
 *
 * \param [in] handle Reference to  handle.
 *
 * \return LDC node
 *
 * \ingroup group_applib_sde_singlelayer
 */
vx_node             SL_SDE_getSDENode(SL_SDE_Handle handle);

#ifdef __cplusplus
}
#endif

#endif /* _SDE_SINGLELAYER_H_ */

