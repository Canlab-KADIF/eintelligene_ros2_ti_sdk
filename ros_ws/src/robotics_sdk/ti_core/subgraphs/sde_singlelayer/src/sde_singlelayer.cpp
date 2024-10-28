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
#include "sde_singlelayer.h"

using namespace ti_core_common;

typedef struct SL_SDE_Context
{
    /** OVX Node References */
    vx_context                        vxContext;

    /** OpenVX graph */
    vx_graph                          vxGraph;

    /** SDE Node */
    vx_node                           vxDmpacSdeNode;

    /** SDE visualization node */
    vx_node                           vxNodeDisparityVis;

    /** Handle to the data object holding the SDE configuration 
     *  parameters. */
    vx_user_data_object               vxSdeConfig;

    /** Handle to the data object holding the disparity Visualization 
     *  configuration parameters. 
     */
    vx_user_data_object               vxSdeVisConfig;

    /** Left rectified image object */
    vx_image                          vxLeftRectImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Right rectified image object */
    vx_image                          vxRightRectImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Raw disparity map object */
    vx_image                          vxSde16BitOutput[GRAPH_MAX_PIPELINE_DEPTH];

    /** Handle to the data object holding the disparity histogram */
    vx_distribution                   vxHistogram;

    /** input image width */
    uint16_t                          width;
    
    /** input image height */
    uint16_t                          height;

    /** input format, U8 or YUV_UYVY */
    uint8_t                           inputFormat;

    /** SDE config params */
    tivx_dmpac_sde_params_t           sdeCfg;

    /** Disparity visualization params */
    tivx_sde_disparity_vis_params_t   sdeVisCfg;

    /** pipeline depth */
    uint8_t                           pipelineDepth;

    /** Input pipeline depth */
    uint8_t                           inputPipelineDepth;

    /** flag indicating whether or not create input OpenVX object in applib */
    uint8_t                           createInputFlag;

    /** flag indicating whether or not create output OpenVX object in applib */
    uint8_t                           createOutputFlag;

} SL_SDE_Context;

static vx_status SL_SDE_setParams(SL_SDE_Handle        handle,
                                  SL_SDE_createParams *createParams);

static vx_status SL_SDE_createGraph(SL_SDE_Handle handle);

static void      SL_SDE_releaseGraph(SL_SDE_Handle handle);


SL_SDE_Handle SL_SDE_create(SL_SDE_createParams *createParams)
{
    SL_SDE_Handle      handle;
    vx_status          vxStatus;

    handle = new SL_SDE_Context();

    /* Set applib-level create parameters */
    vxStatus = SL_SDE_setParams(handle, createParams);

    if (vxStatus != (vx_status)VX_SUCCESS)
    {
        LOG_ERROR("SL_SDE_setParams() failed.\n");
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Create nodes and graph */
        vxStatus = SL_SDE_createGraph(handle);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("SL_SDE_createGraph() failed.\n");
        }
    }

    if (vxStatus != VX_SUCCESS)
    {
        delete handle;
        handle = NULL;
    }

    return handle;

}

void SL_SDE_delete(SL_SDE_Handle *handle)
{
    if (*handle)
    {
        SL_SDE_releaseGraph(*handle);

        delete *handle;
        *handle = NULL;
    }

    return;

} /* SL_SDE_delete */

vx_status SL_SDE_setParams(SL_SDE_Handle        handle,
                           SL_SDE_createParams *createParams)
{
    SL_SDE_Context *appCntxt;
    vx_df_image     imgFormat;
    vx_status       vxStatus = VX_SUCCESS;
    int32_t         i;

    appCntxt = (SL_SDE_Context *)handle;

    appCntxt->vxContext          = createParams->vxContext;
    appCntxt->vxGraph            = createParams->vxGraph;
    appCntxt->pipelineDepth      = createParams->pipelineDepth;
    appCntxt->inputPipelineDepth = createParams->inputPipelineDepth;
    appCntxt->createInputFlag    = createParams->createInputFlag;
    appCntxt->createOutputFlag   = createParams->createOutputFlag;

    appCntxt->sdeCfg             = createParams->sdeCfg;
    appCntxt->width              = createParams->width;
    appCntxt->height             = createParams->height;
    appCntxt->inputFormat        = createParams->inputFormat;

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
            imgFormat = VX_DF_IMAGE_NV12;
        }

        // input left image
        appCntxt->vxLeftRectImage[0] =
            vxCreateImage(appCntxt->vxContext,
                          appCntxt->width,
                          appCntxt->height,
                          imgFormat);

        if (appCntxt->vxLeftRectImage[0] == NULL)
        {
            LOG_ERROR("vxCreateImage() failed\n");
            vxStatus = VX_FAILURE;
        }

        if (vxStatus == (vx_status)VX_SUCCESS)
        {
            // input right image
            appCntxt->vxRightRectImage[0] =
                vxCreateImage(appCntxt->vxContext,
                              appCntxt->width,
                              appCntxt->height,
                              imgFormat);

            if (appCntxt->vxRightRectImage[0] == NULL)
            {
                LOG_ERROR("vxCreateImage() failed\n");
                vxStatus = VX_FAILURE;
            }
        }
    }
    else 
    {
        for (i = 0; i < appCntxt->inputPipelineDepth; i++)
        {
            appCntxt->vxLeftRectImage[i] = createParams->vxLeftRectImage[i];
        }

        for (i = 0; i < appCntxt->pipelineDepth; i++)
        {
            appCntxt->vxRightRectImage[i] = createParams->vxRightRectImage[i];
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /*
         * set up output objects
         */
        if (appCntxt->createOutputFlag == 1)
        {
            appCntxt->vxSde16BitOutput[0]  =
                vxCreateImage(appCntxt->vxContext,
                              appCntxt->width,
                              appCntxt->height,
                              VX_DF_IMAGE_S16);        

            if (appCntxt->vxSde16BitOutput[0] == NULL)
            {
                LOG_ERROR("vxCreateImage() failed\n");
                vxStatus = VX_FAILURE;
            }
        }
        else 
        {
            for (i = 0; i < appCntxt->pipelineDepth; i++)
            {
                appCntxt->vxSde16BitOutput[i] =
                    createParams->vxSde16BitOutput[i];
            }
        }
    }

    return vxStatus;
}


vx_status SL_SDE_createGraph(SL_SDE_Handle handle)
{
    SL_SDE_Context *appCntxt;
    vx_status       vxStatus = VX_SUCCESS;

    appCntxt = (SL_SDE_Context *)handle;

    /* Mapping Graph */
    if (appCntxt->vxGraph == NULL)
    {
        LOG_ERROR("NULL graph handle.\n");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* DMPAC SDE node */
        appCntxt->vxSdeConfig =
            vxCreateUserDataObject(appCntxt->vxContext,
                                   "tivx_dmpac_sde_params_t",
                                   sizeof(tivx_dmpac_sde_params_t),
                                   &appCntxt->sdeCfg);

        if (appCntxt->vxSdeConfig == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxSetReferenceName((vx_reference)appCntxt->vxSdeConfig,
                           "Stereo_Config");

        /* histogram output from SDE */
        appCntxt->vxHistogram =
            vxCreateDistribution(appCntxt->vxContext, 128, 0, 4096);

        if (appCntxt->vxHistogram == NULL)
        {
            LOG_ERROR("vxCreateDistribution() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxSetReferenceName((vx_reference)appCntxt->vxHistogram,
                           "Stereo_OutputHistogramDistribution_L0");

        /* crearte DMPAC SDE node */
        appCntxt->vxDmpacSdeNode =
            tivxDmpacSdeNode(appCntxt->vxGraph,
                             appCntxt->vxSdeConfig,
                             appCntxt->vxLeftRectImage[0],
                             appCntxt->vxRightRectImage[0],
                             appCntxt->vxSde16BitOutput[0],
                             appCntxt->vxHistogram);

        if (appCntxt->vxDmpacSdeNode == NULL)
        {
            LOG_ERROR("tivxDmpacSdeNode() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxSetReferenceName((vx_reference)appCntxt->vxDmpacSdeNode,
                           "DMPAC_SDE_Processing");

        vxSetNodeTarget(appCntxt->vxDmpacSdeNode,
                        VX_TARGET_STRING,
                        TIVX_TARGET_DMPAC_SDE);
    }

    return vxStatus;

} /* SL_SDE_createGraph */


void SL_SDE_releaseGraph(SL_SDE_Handle handle)
{
    SL_SDE_Context * appCntxt;

    appCntxt = (SL_SDE_Context *)handle;

    /* Layer 0 */
    /* release DMPAC SDE node */
    vxReleaseNode(&appCntxt->vxDmpacSdeNode);
    vxReleaseUserDataObject(&appCntxt->vxSdeConfig);

    if (appCntxt->createInputFlag == 1)
    {
        vxReleaseImage(&appCntxt->vxLeftRectImage[0]);
        vxReleaseImage(&appCntxt->vxRightRectImage[0]);
    } 

    if (appCntxt->createOutputFlag == 1)
    {
        vxReleaseImage(&appCntxt->vxSde16BitOutput[0]);
    }

    vxReleaseDistribution(&appCntxt->vxHistogram);

    return;
} /* SL_SDE_releaseGraph */


vx_node  SL_SDE_getSDENode(SL_SDE_Handle handle)
{
    SL_SDE_Context *appCntxt = (SL_SDE_Context *)handle;
    return appCntxt->vxDmpacSdeNode;
}

