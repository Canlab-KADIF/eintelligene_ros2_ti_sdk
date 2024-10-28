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

#include "TI/tivx_target_kernel.h"
#include "sde_ldc.h"

#include "sde_ldc_module.h"

using namespace ti_core_common;

typedef struct SDELDC_Context
{
    /** OpenVX References */
    vx_context                        vxContext;

    /** OpenVX graph */ 
    vx_graph                          vxGraph;

    /** Stereo LDC node context */
    LDCObj                            ldcObj;

    /** Input left image object to a graph*/
    vx_image                          vxInputLeftImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Input right image object to a graph */
    vx_image                          vxInputRightImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Output left image object from a graph */
    vx_image                          vxOutputLeftImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Output right image object from a graph */
    vx_image                          vxOutputRightImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** input image width */
    uint16_t                          width;

    /** input image height */
    uint16_t                          height;

    /** input format: U8 or YUV_UYVY */
    uint8_t                           inputFormat;

    /** pipeline depth */
    uint8_t                           pipelineDepth;

    /** Input pipeline depth,
     *  which is the number of each input OpenVX object
     */
    uint8_t                           inputPipelineDepth;

    /** Output pipeline depth 
     *  which is the number of each output openVX object
     */
    uint8_t                           outputPipelineDepth;

    /** flag indicating whether or not create input OpenVX object in applib */
    uint8_t                           createInputFlag;

    /** flag indicating whether or not create output OpenVX object in applib */
    uint8_t                           createOutputFlag;

} SDELDC_Context;

static vx_status SDELDC_setParams(SDELDC_Handle        handle,
                                  SDELDC_createParams *createParams);

static vx_status SDELDC_createGraph(SDELDC_Handle handle);
static void      SDELDC_releaseGraph(SDELDC_Handle handle);


SDELDC_Handle SDELDC_create(SDELDC_createParams *createParams)
{
    SDELDC_Handle      handle;
    vx_status          vxStatus;
 
    handle = new SDELDC_Context();

    /* Set applib-level create parameters */
    vxStatus = SDELDC_setParams(handle, createParams);

    if (vxStatus != (vx_status)VX_SUCCESS)
    {
        LOG_ERROR("SDELDC_setParams() failed.\n");
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Create nodes and graph */
        vxStatus = SDELDC_createGraph(handle);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("SDELDC_createGraph() failed.\n");
        }
    }

    if (vxStatus != (vx_status)VX_SUCCESS)
    {
        delete handle;
        handle = NULL;
    }

    return handle;
} 

void SDELDC_delete(SDELDC_Handle *handle)
{
    if (*handle)
    {
        SDELDC_releaseGraph(*handle);

        delete *handle;
        *handle = NULL;
    }

    return;

} /* SDELDC_delete */


vx_status SDELDC_setParams(SDELDC_Handle        handle,
                           SDELDC_createParams *createParams)
{
    SDELDC_Context *appCntxt;
    vx_df_image     imgFormat;
    vx_status       vxStatus = VX_SUCCESS;
    int32_t         i; 

    appCntxt = (SDELDC_Context *)handle;

    appCntxt->vxContext           = createParams->vxContext;
    appCntxt->vxGraph             = createParams->vxGraph;
    appCntxt->pipelineDepth       = createParams->pipelineDepth;
    appCntxt->inputPipelineDepth  = createParams->inputPipelineDepth;
    appCntxt->outputPipelineDepth = createParams->outputPipelineDepth;
    appCntxt->createInputFlag     = createParams->createInputFlag;
    appCntxt->createOutputFlag    = createParams->createOutputFlag;

    appCntxt->inputFormat         = createParams->inputFormat;
    appCntxt->width               = createParams->width;
    appCntxt->height              = createParams->height;

    /*
     * set up input objects
     */
    if (appCntxt->createInputFlag == 1)
    {
        if (appCntxt->inputFormat == CM_IMG_FORMAT_Y)
        {
            imgFormat = VX_DF_IMAGE_U8;
        }
        else 
        {
            imgFormat = VX_DF_IMAGE_UYVY;
        }

        // input left image
        appCntxt->vxInputLeftImage[0] =
            vxCreateImage(appCntxt->vxContext,
                          appCntxt->width,
                          appCntxt->height,
                          imgFormat);

        if (appCntxt->vxInputLeftImage[0] == NULL)
        {
            LOG_ERROR("vxCreateImage() failed\n");
            vxStatus = VX_FAILURE;
        }

        if (vxStatus == (vx_status)VX_SUCCESS)
        {
            // input right image
            appCntxt->vxInputRightImage[0] =
                vxCreateImage(appCntxt->vxContext,
                              appCntxt->width,
                              appCntxt->height,
                              imgFormat);

            if (appCntxt->vxInputRightImage[0] == NULL)
            {
                LOG_ERROR("vxCreateImage() failed\n");
                vxStatus = VX_FAILURE;
            }
        }

    }
    else 
    {
        for (i = 0; i < appCntxt->pipelineDepth; i++)
        {
            appCntxt->vxInputLeftImage[i] =
                createParams->vxInputLeftImage[i];

            appCntxt->vxInputRightImage[i] =
                createParams->vxInputRightImage[i];
        }
    } 

    /*
     * Set up output objects 
     */
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        if (appCntxt->createOutputFlag == 1)
        {
            if (appCntxt->inputFormat == CM_IMG_FORMAT_Y)
            {
                imgFormat = VX_DF_IMAGE_U8;
            }
            else 
            {
                imgFormat = VX_DF_IMAGE_NV12;
            }

            // output left image
            appCntxt->vxOutputLeftImage[0] =
                vxCreateImage(appCntxt->vxContext,
                              appCntxt->width,
                              appCntxt->height,
                              imgFormat);

            if (appCntxt->vxOutputLeftImage[0] == NULL)
            {
                LOG_ERROR("vxCreateImage() failed\n");
                vxStatus = VX_FAILURE;
            }

            // output right image
            if (vxStatus == (vx_status)VX_SUCCESS)
            {
                appCntxt->vxOutputRightImage[0] =
                    vxCreateImage(appCntxt->vxContext,
                                  appCntxt->width,
                                  appCntxt->height,
                                  imgFormat);

                if (appCntxt->vxOutputRightImage[0] == NULL)
                {
                    LOG_ERROR("vxCreateImage() failed\n");
                    vxStatus = VX_FAILURE;
                }
            }
        }
        else 
        {
            for (i = 0; i < appCntxt->outputPipelineDepth; i++)
            {
                appCntxt->vxOutputLeftImage[i] =
                    createParams->vxOutputLeftImage[i];
            }

            for (i = 0; i < appCntxt->pipelineDepth; i++)
            {
                appCntxt->vxOutputRightImage[i] =
                    createParams->vxOutputRightImage[i];
            }
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        // init LDC
        vxStatus = SDELDC_init_ldc(appCntxt->vxContext,
                                   &appCntxt->ldcObj,
                                   createParams->leftLutFileName,
                                   createParams->rightLutFileName,
                                   createParams->width,
                                   createParams->height);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("SDELDC_init_ldc() failed.\n");
        }
    }

    return vxStatus;
}

vx_status SDELDC_createGraph(SDELDC_Handle handle)
{
    SDELDC_Context *appCntxt = (SDELDC_Context *)handle;
    vx_status       vxStatus = VX_SUCCESS;

    /* Graph */
    // make sure the vxGraph is available
    if (appCntxt->vxGraph == NULL)
    {
        LOG_ERROR("NULL graph handle.\n");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = SDELDC_create_graph_ldc(appCntxt->vxGraph,
                                           &appCntxt->ldcObj, 
                                           appCntxt->vxInputLeftImage[0],
                                           appCntxt->vxOutputLeftImage[0],
                                           appCntxt->vxInputRightImage[0],
                                           appCntxt->vxOutputRightImage[0]);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("SDELDC_create_graph_ldc() failed.\n");
        }
    }

    return vxStatus;

} /* SDELDC_createGraph */


void SDELDC_releaseGraph(SDELDC_Handle handle)
{
    SDELDC_Context * appCntxt;

    appCntxt = (SDELDC_Context *)handle;

    /* release LDC node */
    if (appCntxt->createInputFlag == 1)
    {
        vxReleaseImage(&appCntxt->vxInputLeftImage[0]);
        vxReleaseImage(&appCntxt->vxInputRightImage[0]);
    } 

    if (appCntxt->createOutputFlag == 1)
    {
        vxReleaseImage(&appCntxt->vxOutputLeftImage[0]);
        vxReleaseImage(&appCntxt->vxOutputRightImage[0]);
    }

    SDELDC_deinit_ldc(&appCntxt->ldcObj);
    SDELDC_delete_ldc(&appCntxt->ldcObj);

    return;

} /* SDELDC_releaseGraph */

vx_node  SDELCD_getLeftLDCNode(SDELDC_Handle handle)
{
    SDELDC_Context *appCntxt = (SDELDC_Context *)handle;
    return appCntxt->ldcObj.node_left;
}

vx_node  SDELCD_getRightLDCNode(SDELDC_Handle handle)
{
    SDELDC_Context *appCntxt = (SDELDC_Context *)handle;
    return appCntxt->ldcObj.node_right;
}

vx_image SDELDC_getOutputLeftImage(SDELDC_Handle handle)
{
    SDELDC_Context *appCntxt = (SDELDC_Context *)handle;
    return appCntxt->vxOutputLeftImage[0];
}

vx_image SDELDC_getOutputRightImage(SDELDC_Handle handle)
{
    SDELDC_Context *appCntxt = (SDELDC_Context *)handle;
    return appCntxt->vxOutputRightImage[0];
}
