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
#include "sde_triangulate.h"

typedef struct SDE_TRIANG_Context
{
    /** OVX Node References */
    vx_context                        vxContext;

    /** OpenVX graph */
    vx_graph                          vxGraph;

    /** Input image color conversion node */
    vx_node                           vxColorConv;

    /* SDE Triangulation Node */
    vx_node                           vxSdeTriang;

    /** Handle to the data object holding the camera configuration parameters */
    vx_user_data_object               vxStereoCamConfig;

    /** Handle to the data object holding the pointcloud configuration parameters */
    vx_user_data_object               vxStereoPcConfig;

    /** Right rectified image object */
    vx_image                          vxInputRectImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Input rectified image in RGB format */
    vx_image                          vxRGBRectImage;

    /** Raw disparity map object */
    vx_image                          vxInputSde16Bit[GRAPH_MAX_PIPELINE_DEPTH];

    /** Output point cloud */
    vx_user_data_object               vxOutputTriangPC[GRAPH_MAX_PIPELINE_DEPTH];

    /** input image width */
    uint16_t                          width;
    
    /** input image height */
    uint16_t                          height;

    /** Stereo camera params */
    tivx_stereo_cam_params_t          stereoCamCfg;

    /** PointCloud params */
    tivx_stereo_pointcloud_params_t   stereoPcCfg;

    /** pipeline depth */
    uint8_t                           pipelineDepth;

    /** Input pipeline depth */
    uint8_t                           inputPipelineDepth;

    /** flag indicating whether or not create input OpenVX object in applib */
    uint8_t                           createInputFlag;

    /** flag indicating whether or not create output OpenVX object in applib */
    uint8_t                           createOutputFlag;

    /** Color Conversion Node Core mapping. */
    const char                       *ccNodeCore;

    /** Triangulation Node Core mapping. */
    const char                       *triangNodeCore;

} SDE_TRIANG_Context;

static vx_status SDE_TRIANG_setParams(SDE_TRIANG_Handle        handle,
                                      SDE_TRIANG_createParams *createParams);

static vx_status SDE_TRIANG_createGraph(SDE_TRIANG_Handle handle);

static void      SDE_TRIANG_releaseGraph(SDE_TRIANG_Handle handle);


SDE_TRIANG_Handle SDE_TRIANG_create(SDE_TRIANG_createParams *createParams)
{
    SDE_TRIANG_Handle   handle;
    vx_status           vxStatus;

    handle = new SDE_TRIANG_Context();

    /* Set applib-level create parameters */
    vxStatus = SDE_TRIANG_setParams(handle, createParams);

    if (vxStatus != (vx_status)VX_SUCCESS)
    {
        LOG_ERROR("SDE_TRIANG_setParams() failed.\n");
    }

    if (vxStatus == (vx_status) VX_SUCCESS)
    {
        /* Create nodes and graph */
        vxStatus = SDE_TRIANG_createGraph(handle);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("SDE_TRIANG_createGraph() failed.\n");
        }
    }

    if (vxStatus != VX_SUCCESS)
    {
        delete handle;
        handle = NULL;
    }

    return handle;

}

void SDE_TRIANG_delete(SDE_TRIANG_Handle *handle)
{
    if (*handle)
    {
        SDE_TRIANG_releaseGraph(*handle);

        delete *handle;
        *handle = NULL;
    }

    return;

} /* SDE_TRIANG_delete */


vx_status SDE_TRIANG_setParams(SDE_TRIANG_Handle        handle,
                                      SDE_TRIANG_createParams *createParams)
{
    SDE_TRIANG_Context *appCntxt;
    vx_status           vxStatus = VX_SUCCESS;
    int32_t             i;

    appCntxt = (SDE_TRIANG_Context *)handle;

    appCntxt->vxContext          = createParams->vxContext;
    appCntxt->vxGraph            = createParams->vxGraph;
    appCntxt->pipelineDepth      = createParams->pipelineDepth;
    appCntxt->inputPipelineDepth = createParams->inputPipelineDepth;
    appCntxt->createInputFlag    = createParams->createInputFlag;
    appCntxt->createOutputFlag   = createParams->createOutputFlag;

    appCntxt->width              = createParams->width;
    appCntxt->height             = createParams->height;
    appCntxt->stereoCamCfg       = createParams->stereoCamCfg;
    appCntxt->stereoPcCfg        = createParams->stereoPcCfg;

    appCntxt->triangNodeCore     = createParams->triangNodeCore;
    appCntxt->ccNodeCore         = createParams->ccNodeCore;

    if (appCntxt->triangNodeCore == NULL)
    {
        appCntxt->triangNodeCore = SDE_TRIANG_DEFAULT_CORE_MAPPING;
    }

    if (appCntxt->ccNodeCore == NULL)
    {
        appCntxt->ccNodeCore = SDE_TRIANG_DEFAULT_CORE_MAPPING;
    }    

    /*
     * set up input objects
     */
    if (appCntxt->createInputFlag == 1)
    {
        LOG_ERROR("createInputFlag should be false\n");

        vxStatus = VX_FAILURE;
    }
    else 
    {
        /* As many as appCntxt->pipelineDepth */ 
        for (i = 0; i < appCntxt->pipelineDepth; i++)
        {
            appCntxt->vxInputRectImage[i] = createParams->vxInputRectImage[i];
            appCntxt->vxInputSde16Bit[i] = createParams->vxInputSde16Bit[i];
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /*
         * set up output objects
         */
        if (appCntxt->createOutputFlag == 1)
        {
            LOG_ERROR("createOutputFlag should be false\n");

            vxStatus = VX_FAILURE;
        }
        else 
        {
            for (i = 0; i < appCntxt->pipelineDepth; i++)
            {
                appCntxt->vxOutputTriangPC[i] = createParams->vxOutputTriangPC[i];
            }
        }
    }

    return vxStatus;
}


vx_status SDE_TRIANG_createGraph(SDE_TRIANG_Handle handle)
{
    SDE_TRIANG_Context *appCntxt;
    vx_status           vxStatus = VX_SUCCESS;

    appCntxt = (SDE_TRIANG_Context *)handle;;

    /* Graph */
    if (appCntxt->vxGraph == NULL)
    {
        LOG_ERROR("Graph not managed. Graph object should be passed to \n");

        vxStatus = VX_FAILURE;
    }

    /********************************/
    /* Color format conversion node */
    /********************************/
    /* Create RGB rectified image object */
    appCntxt->vxRGBRectImage =
        vxCreateImage(appCntxt->vxContext,
                      appCntxt->width,
                      appCntxt->height,
                      VX_DF_IMAGE_RGB);

    if (appCntxt->vxRGBRectImage == NULL)
    {
        LOG_ERROR("vxCreateUserDataObject() failed\n");
        vxStatus = VX_FAILURE;
    }
    else 
    {
        vxSetReferenceName((vx_reference)appCntxt->vxRGBRectImage,
                           "RGBRectImage");
    }

    /* Create color conversion node */
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        appCntxt->vxColorConv =
            vxColorConvertNode(appCntxt->vxGraph,
                               appCntxt->vxInputRectImage[0],
                               appCntxt->vxRGBRectImage);

        if (appCntxt->vxColorConv == NULL)
        {
            LOG_ERROR("vxColorConvertNode() failed\n");

            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)appCntxt->vxColorConv,
                               "Color_Convert");

            vxStatus = vxSetNodeTarget(appCntxt->vxColorConv,
                                       VX_TARGET_STRING,
                                       appCntxt->ccNodeCore);

            if (vxStatus != (vx_status)VX_SUCCESS)
            {
                LOG_ERROR("vxSetNodeTarget() failed\n");
            }
        }
    }

    /**************************/
    /* SDE Triangulation node */
    /**************************/
    /* Camera config params object */
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        // set stereo camera config from stereoCamCfg
        appCntxt->vxStereoCamConfig = 
                vxCreateUserDataObject(appCntxt->vxContext, 
                                      "StereoCamConfig",
                                       sizeof(tivx_stereo_cam_params_t), 
                                       &appCntxt->stereoCamCfg);

        if (appCntxt->vxStereoCamConfig == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed\n");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)appCntxt->vxStereoCamConfig,
                               "StereoCamConfig");
        }
    }

    /* Stereo pointcloud config params object */
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        // set stereo PC config from stereoPcCfg
        appCntxt->vxStereoPcConfig = 
                vxCreateUserDataObject(appCntxt->vxContext, 
                                      "StereoPcConfig",
                                       sizeof(tivx_stereo_pointcloud_params_t), 
                                       &appCntxt->stereoPcCfg);

        if (appCntxt->vxStereoPcConfig == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed\n");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)appCntxt->vxStereoPcConfig,
                               "StereoPcConfig");
        }
    } 

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        
        appCntxt->vxSdeTriang = tivxSdeTriangulationNode(
                                    appCntxt->vxGraph,
                                    appCntxt->vxStereoCamConfig,
                                    appCntxt->vxStereoPcConfig,
                                    appCntxt->vxRGBRectImage,
                                    appCntxt->vxInputSde16Bit[0],
                                    appCntxt->vxOutputTriangPC[0]);

        if (appCntxt->vxSdeTriang == NULL)
        {
            LOG_ERROR("tivxSdeTriangulationNode() failed\n");

            vxStatus = VX_FAILURE; 
        }
        else
        {
            vxSetReferenceName((vx_reference)appCntxt->vxSdeTriang,
                               "Stereo_Triangulation");

            vxStatus = vxSetNodeTarget(appCntxt->vxSdeTriang,
                                       VX_TARGET_STRING,
                                       appCntxt->triangNodeCore);

            if (vxStatus != (vx_status)VX_SUCCESS)
            {
                LOG_ERROR("vxSetNodeTarget() failed\n");
            }
        }
    }

    return vxStatus;

} /* SDE_TRIANG_createGraph */


void SDE_TRIANG_releaseGraph(SDE_TRIANG_Handle handle)
{
    SDE_TRIANG_Context * appCntxt = (SDE_TRIANG_Context *)handle;

    vxReleaseNode(&appCntxt->vxColorConv);
    vxReleaseNode(&appCntxt->vxSdeTriang);
    vxReleaseImage(&appCntxt->vxRGBRectImage);
    vxReleaseUserDataObject(&appCntxt->vxStereoCamConfig);
    vxReleaseUserDataObject(&appCntxt->vxStereoPcConfig);

    return;
} /* SDE_TRIANG_releaseGraph */


vx_node  SDE_TRIANG_getColorConvNode(SDE_TRIANG_Handle handle)
{
    SDE_TRIANG_Context * appCntxt = (SDE_TRIANG_Context *)handle;

    return appCntxt->vxColorConv;
}

vx_node  SDE_TRIANG_getTriangNode(SDE_TRIANG_Handle handle)
{
    SDE_TRIANG_Context * appCntxt = (SDE_TRIANG_Context *)handle;

    return appCntxt->vxSdeTriang;
}
