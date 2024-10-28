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

#ifndef _SDELDC_H_
#define _SDELDC_H_

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
 * \defgroup group_applib_sde_ldc Stereo camera LDC  code.
 * \ingroup group_ptk_applib
 *
 */

#include <utils/perf_stats/include/app_perf_stats.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <tivx_utils_graph_perf.h>
#include <tivx_utils_file_rd_wr.h>

#define SDELDC_DEFAULT_CORE_MAPPING  (TIVX_TARGET_DSP1)
#define SDELDC_MAX_LINE_LEN          (256U)


/**
 * \brief SDE LDC  create parameter context.
 *
 * \ingroup group_applib_sde_ldc
 */
typedef struct
{
    /** OpenVX references */
    vx_context                        vxContext;

    /** Graph handle from the Application */
    vx_graph                          vxGraph;

    /** Input image width */
    uint16_t                          width;

    /** Input image height */
    uint16_t                          height;

    /** Input iamge format: U8 or YUV_UYVY */
    uint16_t                          inputFormat;

    /** pipeline depth */
    uint8_t                           pipelineDepth;

    /** Input object pipeline depth, 
     *  which is the number of each input OpenVX object
     */
    uint8_t                           inputPipelineDepth;

    /** Output object pipeline depth, 
     *  which is the number of each output openVX object
     */
    uint8_t                           outputPipelineDepth;

    /** Flag indicating whether or not create input OpenVX object in applib */
    uint8_t                           createInputFlag;

    /** Flag indicating whether or not create output OpenVX object in applib */
    uint8_t                           createOutputFlag;

    /** Input left image object to a graph*/
    vx_image                          vxInputLeftImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Input right image object to a graph */
    vx_image                          vxInputRightImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Output left image object from a graph */
    vx_image                          vxOutputLeftImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Output right image object from a graph */
    vx_image                          vxOutputRightImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** left LUT file name */
    char                             *leftLutFileName;

    /** right LUT file name */
    char                             *rightLutFileName;

} SDELDC_createParams;

struct  SDELDC_Context;
typedef SDELDC_Context * SDELDC_Handle;

/**
 * \brief Function to initialize the .
 *
 * \param [in] createParams  create parameters.
 *
 * \return Valid handle on success. NULL otherwise.
 *
 * \ingroup group_applib_sde_ldc
 */
SDELDC_Handle  SDELDC_create(SDELDC_createParams *createParams);

/**
 * \brief Function to de-init the  and release the memory associated with
 *        the handle.
 *
 * \param [in,out] handle Reference to  handle.
 *
 * \ingroup group_applib_sde_ldc
 */
void                 SDELDC_delete(SDELDC_Handle *handle);

/**
 * \brief Function to get the LDC node handle for left input image.
 *        This is needed when a graph is created outside . 
 *        The caller calls this function to put this node into the graph.
 *
 * \param [in] handle Reference to  handle.
 *
 * \return LDC node 
 *
 * \ingroup group_applib_sde_ldc
 */
vx_node              SDELCD_getLeftLDCNode(SDELDC_Handle handle);

/**
 * \brief Function to get the LDC node handle for right input image.
 *        This is needed when a graph is created outside . 
 *        The caller calls this function to put this node into the graph.
 * 
 * \param [in] handle Reference to  handle.
 *
 * \return LDC node 
 *
 * \ingroup group_applib_sde_ldc
 */
vx_node              SDELCD_getRightLDCNode(SDELDC_Handle handle);

/**
 * \brief Function to get the output left image object from a graph.
 *        This is needed when this object is passed to a connected graph 
 *        to build a bigger graph by a caller.
 *
 * \param [in] handle Reference to  handle.
 *
 * \return image object 
 *
 * \ingroup group_applib_sde_ldc
 */
vx_image             SDELDC_getOutputLeftImage(SDELDC_Handle handle);

/**
 * \brief Function to get the output right image object from a graph.
 *        This is needed when this object is passed to a connected graph 
 *        to build a bigger graph by a caller.
 *
 * \param [in] handle Reference to  handle.
 *
 * \return image object 
 *
 * \ingroup group_applib_sde_ldc
 */
vx_image             SDELDC_getOutputRightImage(SDELDC_Handle handle);



#ifdef __cplusplus
}
#endif

#endif /* _SDELDC_H_ */

