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

#include <string.h>
#include <VX/vx.h>
#include <TI/tivx.h>

#include <cm_ldc_node_cntxt.h>

namespace ti_core_common 
{

vx_status CM_ldcNodeCntxtInit(
        CM_LdcNodeCntxt            *ldcObj,
        vx_context                  context,
        const CM_LdcCreateParams   *createParams)
{
    uint8_t    *lut;
    uint32_t    tblWidthDs;
    uint32_t    tblHeightDs;
    uint32_t    lutSize;
    vx_status   vxStatus;

    vxStatus = VX_SUCCESS;
    lut      = NULL;

    if (ldcObj == NULL)
    {
        LOG_ERROR("Parameter 'ldcObj' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (context == NULL)
    {
        LOG_ERROR("Parameter 'context' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (createParams == NULL)
    {
        LOG_ERROR("Parameter 'createParams' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (ldcObj->state != CM_LDC_NODE_CNTXT_STATE_INVALID)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        tblWidthDs = (((createParams->width/
                        (1 << createParams->ssFactor))+1u)+15u)&(~15u);

        tblHeightDs = ((createParams->height/(1 << createParams->ssFactor))+1u);
        lutSize     = tblWidthDs * tblHeightDs * 4;

        lut = (uint8_t *)malloc(lutSize);

        if (lut == NULL)
        {
            LOG_ERROR("Failed to allocate LUT.");
            vxStatus = VX_FAILURE;
        }
        else
        {
            /* Read the LUT entries. */
            FILE *fp = NULL;

            fp = fopen(createParams->lutFilePath, "rb");

            if (fp == NULL)
            {
                LOG_ERROR("Failed to open LUT file [%s] for reading.",
                         createParams->lutFilePath);
                vxStatus = VX_FAILURE;
            }
            else
            {
                uint32_t    size;

                size = fread(lut, 1, lutSize, fp);

                if (size != lutSize)
                {
                    LOG_ERROR("Error reading LUT data. "
                             "Expecting [%d] bytes but read [%d] bytes ",
                             lutSize, size);
                    vxStatus = VX_FAILURE;
                }

                fclose(fp);
            }
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Mesh image */
        ldcObj->vxMeshImage = vxCreateImage(context,
                                          tblWidthDs,
                                          tblHeightDs,
                                          VX_DF_IMAGE_U32);

        if (ldcObj->vxMeshImage == NULL)
        {
            LOG_ERROR("vxCreateImage() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)ldcObj->vxMeshImage,
                               "LdcMeshImage_U32");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vx_imagepatch_addressing_t  imgAddr;
        vx_rectangle_t              rect;

        /* Copy Mesh table */
        rect.start_x = 0;
        rect.start_y = 0;
        rect.end_x   = tblWidthDs;
        rect.end_y   = tblHeightDs;

        imgAddr.dim_x    = tblWidthDs;
        imgAddr.dim_y    = tblHeightDs;
        imgAddr.stride_x = 4u;
        imgAddr.stride_y = tblWidthDs * 4u;

        vxStatus = vxCopyImagePatch(ldcObj->vxMeshImage,
                                    &rect,
                                    0,
                                    &imgAddr,
                                    lut,
                                    VX_WRITE_ONLY,
                                    VX_MEMORY_TYPE_HOST);

        if (VX_SUCCESS != vxStatus)
        {
            LOG_ERROR("vxCopyImagePatch() failed");
        }
    }

    /* When LDC output is graph parameter, LDC output images are provided.
     * So this is commented out for now.
     * ldcObj->outImage may be set to the provided output images later.
     */
#if 0
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Output image */
        ldcObj->outImage = vxCreateImage(context,
                                         createParams->width,
                                         createParams->height,
                                         VX_DF_IMAGE_NV12);

        if (ldcObj->outImage == NULL)
        {
            LOG_ERROR("vxCreateImage() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)ldcObj->outImage,
                               "LdcOutputImage_NV12");
        }
    }
#endif

    /* Create the LDC param object. */
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        tivx_vpac_ldc_params_t  ldcParams;

        tivx_vpac_ldc_params_init(&ldcParams);

        ldcParams.luma_interpolation_type = 1;
        ldcParams.dcc_camera_id           = 0;

        ldcObj->vxParamConfig =
            vxCreateUserDataObject(context,
                                   "tivx_vpac_ldc_params_t",
                                   sizeof(tivx_vpac_ldc_params_t),
                                   &ldcParams);

        if (ldcObj->vxParamConfig == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)ldcObj->vxParamConfig,
                               "LdcConfig");
        }
    }

    /* Create the LDC mesh param object. */
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        tivx_vpac_ldc_mesh_params_t meshParams;

        tivx_vpac_ldc_mesh_params_init(&meshParams);

        meshParams.mesh_frame_width  = createParams->width;
        meshParams.mesh_frame_height = createParams->height;
        meshParams.subsample_factor  = createParams->ssFactor;

        ldcObj->vxMeshConfig =
            vxCreateUserDataObject(context,
                                   "tivx_vpac_ldc_mesh_params_t",
                                   sizeof(tivx_vpac_ldc_mesh_params_t),
                                   &meshParams);

        if (ldcObj->vxMeshConfig == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)ldcObj->vxMeshConfig,
                               "LdcMeshConfig");
        }
    }

    /* Create the LDC region param object. */
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        tivx_vpac_ldc_region_params_t   regionParams;

        memset(&regionParams, 0, sizeof(tivx_vpac_ldc_region_params_t));

        regionParams.out_block_width  = createParams->blockWidth;
        regionParams.out_block_height = createParams->blockHeight;
        regionParams.pixel_pad        = createParams->pixelPad;

        ldcObj->vxRegionConfig =
            vxCreateUserDataObject(context,
                                   "tivx_vpac_ldc_region_params_t",
                                   sizeof(tivx_vpac_ldc_region_params_t),
                                   &regionParams);

        if (ldcObj->vxRegionConfig == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)ldcObj->vxRegionConfig,
                               "LdcRegionConfig");
        }
    }

    if (lut != NULL)
    {
        free(lut);
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        ldcObj->state = CM_LDC_NODE_CNTXT_STATE_INIT;
    }

    return vxStatus;
}

vx_status CM_ldcNodeCntxtSetup(
        CM_LdcNodeCntxt    *ldcObj,
        vx_context          context,
        vx_graph            graph,
        vx_image            inputImage,
        vx_image            outputImage)
{
    vx_status   vxStatus = VX_SUCCESS;

    if (ldcObj == NULL)
    {
        LOG_ERROR("Parameter 'ldcObj' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (context == NULL)
    {
        LOG_ERROR("Parameter 'context' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (graph == NULL)
    {
        LOG_ERROR("Parameter 'graph' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (inputImage == NULL)
    {
        LOG_ERROR("Parameter 'inputImage' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (ldcObj->state != CM_LDC_NODE_CNTXT_STATE_INIT)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        ldcObj->vxNode = tivxVpacLdcNode(graph,
                                         ldcObj->vxParamConfig,
                                         NULL,
                                         ldcObj->vxRegionConfig,
                                         ldcObj->vxMeshConfig,
                                         ldcObj->vxMeshImage,
                                         NULL,
                                         inputImage,
                                         outputImage,
                                         NULL);

        if (ldcObj->vxNode == NULL)
        {
            LOG_ERROR("tivxVpacLdcNode() failed");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = vxSetNodeTarget(ldcObj->vxNode,
                                   VX_TARGET_STRING,
                                   TIVX_TARGET_VPAC_LDC1);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxSetNodeTarget() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)ldcObj->vxNode, "LdcNode");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        ldcObj->state = CM_LDC_NODE_CNTXT_STATE_SETUP;
    }

    return vxStatus;
}

vx_status CM_ldcNodeCntxtSaveOutImage(
        CM_LdcNodeCntxt    *ldcObj,
        const char         *fileName)
{
    vx_status   vxStatus = VX_SUCCESS;

    if (ldcObj == NULL)
    {
        LOG_ERROR("Parameter 'ldcObj' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (fileName == NULL)
    {
        LOG_ERROR("Parameter 'fileName' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (ldcObj->outImage == NULL)
    {
        LOG_ERROR("No valid Image handle to save.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = CM_saveImage(fileName, ldcObj->outImage);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("CM_saveImage() failed");

        }
    }

    return vxStatus;
}

vx_status CM_ldcNodeCntxtDeInit(
        CM_LdcNodeCntxt    *ldcObj)
{
    vx_status   vxStatus = VX_SUCCESS;

    if (ldcObj == NULL)
    {
        LOG_ERROR("Parameter 'ldcObj' NULL.");
        vxStatus = VX_FAILURE;
    }
    else
    {
        if (ldcObj->state == CM_LDC_NODE_CNTXT_STATE_INVALID)
        {
            LOG_ERROR("Invalid state.");
            vxStatus = VX_FAILURE;
        }
        else
        {
            if (ldcObj->vxNode != NULL)
            {
                vxReleaseNode(&ldcObj->vxNode);
            }

            if (ldcObj->vxParamConfig != NULL)
            {
                vxReleaseUserDataObject(&ldcObj->vxParamConfig);
            }

            if (ldcObj->vxMeshConfig != NULL)
            {
                vxReleaseUserDataObject(&ldcObj->vxMeshConfig);
            }

            if (ldcObj->vxRegionConfig != NULL)
            {
                vxReleaseUserDataObject(&ldcObj->vxRegionConfig);
            }

            if (ldcObj->vxMeshImage != NULL)
            {
                vxReleaseImage(&ldcObj->vxMeshImage);
            }

            if (ldcObj->outImage != NULL)
            {
                vxReleaseImage(&ldcObj->outImage);
            }
        }

        ldcObj->state = CM_LDC_NODE_CNTXT_STATE_INVALID;
    }

    return vxStatus;
}

} // namespace ti_core_common 
