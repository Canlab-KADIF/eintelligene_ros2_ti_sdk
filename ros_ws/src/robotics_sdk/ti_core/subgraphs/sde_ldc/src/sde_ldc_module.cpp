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

#include <cm_displarity_utils.h>
#include <sde_ldc_module.h>

using namespace ti_core_common;

vx_status SDELDC_init_ldc(vx_context context,
                          LDCObj *ldcObj,
                          char *leftLutFileName,
                          char *rightLutFileName,
                          uint16_t width,
                          uint16_t height)
{
    vx_status vxStatus = VX_SUCCESS;

    vx_uint32 table_width_ds, table_height_ds, lutSize;
    vx_imagepatch_addressing_t image_addr;
    vx_rectangle_t rect;

    ldcObj->table_width  = width;
    ldcObj->table_height = height;
    ldcObj->ds_factor    = LDC_DS_FACTOR;

    table_width_ds  = (((ldcObj->table_width / 
                      (1 << ldcObj->ds_factor)) + 1u) + 15u) & (~15u);

    table_height_ds = ((ldcObj->table_height /
                      (1 << ldcObj->ds_factor)) + 1u);

    lutSize         = table_width_ds*table_height_ds*4;

    /* Allocate LUT mems and read tables */
    if ((ldcObj->leftLUT = (uint8_t *)malloc(lutSize)) == NULL)
    {
        LOG_ERROR("Allocate left LUT Failed.\n");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        if ((ldcObj->rightLUT = (uint8_t *)malloc(lutSize)) == NULL)
        {
            free(ldcObj->leftLUT);
            LOG_ERROR("Allocate right LUT Failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        int32_t   status;

        status = CM_allocLdcLut(leftLutFileName,
                                rightLutFileName,
                                ldcObj->leftLUT,
                                ldcObj->rightLUT,
                                lutSize);

        if (status == -1)
        {
            free(ldcObj->leftLUT);
            free(ldcObj->rightLUT);
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Mesh Image */
        ldcObj->mesh_img_left  =
            vxCreateImage(context,
                          table_width_ds,
                          table_height_ds,
                          VX_DF_IMAGE_U32);

        if (ldcObj->mesh_img_left == NULL)
        {
            LOG_ERROR("vxCreateImage() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        ldcObj->mesh_img_right =
            vxCreateImage(context,
                          table_width_ds,
                          table_height_ds,
                          VX_DF_IMAGE_U32);

        if (ldcObj->mesh_img_right == NULL)
        {
            LOG_ERROR("vxCreateImage() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Copy Mesh table */
        rect.start_x = 0;
        rect.start_y = 0;
        rect.end_x = table_width_ds;
        rect.end_y = table_height_ds;

        image_addr.dim_x = table_width_ds;
        image_addr.dim_y = table_height_ds;
        image_addr.stride_x = 4u;
        image_addr.stride_y = table_width_ds * 4u;

        // left mesh image
        vxStatus = vxCopyImagePatch(ldcObj->mesh_img_left,
                                   &rect, 0,
                                   &image_addr,
                                   ldcObj->leftLUT,
                                   VX_WRITE_ONLY,
                                   VX_MEMORY_TYPE_HOST);

        if (VX_SUCCESS != vxStatus)
        {
            LOG_ERROR("Copy Left Mesh Image Failed.\n");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        // right mesh image
        vxStatus = vxCopyImagePatch(ldcObj->mesh_img_right,
                                   &rect, 0,
                                   &image_addr,
                                   ldcObj->rightLUT,
                                   VX_WRITE_ONLY,
                                   VX_MEMORY_TYPE_HOST);

        if (VX_SUCCESS != vxStatus)
        {
            LOG_ERROR("Copy Right Mesh Image Failed.\n");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Mesh Parameters */
        memset(&ldcObj->mesh_params, 0, sizeof(tivx_vpac_ldc_mesh_params_t));
        tivx_vpac_ldc_mesh_params_init(&ldcObj->mesh_params);
        ldcObj->mesh_params.mesh_frame_width  = ldcObj->table_width;
        ldcObj->mesh_params.mesh_frame_height = ldcObj->table_height;
        ldcObj->mesh_params.subsample_factor  = ldcObj->ds_factor;

        ldcObj->mesh_config =
            vxCreateUserDataObject(context,
                                   "tivx_vpac_ldc_mesh_params_t",
                                   sizeof(tivx_vpac_ldc_mesh_params_t),
                                   NULL);

        if (ldcObj->mesh_config == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxCopyUserDataObject(ldcObj->mesh_config, 0,
                             sizeof(tivx_vpac_ldc_mesh_params_t),
                             &ldcObj->mesh_params,
                             VX_WRITE_ONLY,
                             VX_MEMORY_TYPE_HOST);

        /* Block Size parameters */
        ldcObj->region_params.out_block_width  = LDC_BLOCK_WIDTH;
        ldcObj->region_params.out_block_height = LDC_BLOCK_HEIGHT;
        ldcObj->region_params.pixel_pad        = LDC_PIXEL_PAD;

        ldcObj->region_config =
            vxCreateUserDataObject(context,
                                   "tivx_vpac_ldc_region_params_t",
                                   sizeof(tivx_vpac_ldc_region_params_t),
                                   NULL);

        if (ldcObj->region_config == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxCopyUserDataObject(ldcObj->region_config, 0,
                             sizeof(tivx_vpac_ldc_region_params_t),
                             &ldcObj->region_params,
                             VX_WRITE_ONLY,
                             VX_MEMORY_TYPE_HOST);

        /* LDC Configuration */
        tivx_vpac_ldc_params_init(&ldcObj->params);
        ldcObj->params.luma_interpolation_type = 1;
        ldcObj->params.dcc_camera_id = 0; //sensorObj->sensorParams.dccId;

        ldcObj->config =
            vxCreateUserDataObject(context,
                                   "tivx_vpac_ldc_params_t",
                                   sizeof(tivx_vpac_ldc_params_t),
                                   NULL);

        if (ldcObj->config == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxCopyUserDataObject(ldcObj->config, 0,
                             sizeof(tivx_vpac_ldc_params_t),
                             &ldcObj->params,
                             VX_WRITE_ONLY,
                             VX_MEMORY_TYPE_HOST);
    }

    // free LUT mems
    if (ldcObj->leftLUT != NULL)
    {
        free(ldcObj->leftLUT);
        ldcObj->leftLUT = NULL;
    }

    if (ldcObj->rightLUT != NULL)
    {
        free(ldcObj->rightLUT);
        ldcObj->rightLUT = NULL;
    }

    return (vxStatus);
}

void SDELDC_deinit_ldc(LDCObj *ldcObj)
{
    vxReleaseUserDataObject(&ldcObj->config);
    vxReleaseUserDataObject(&ldcObj->region_config);
    vxReleaseUserDataObject(&ldcObj->mesh_config);

    vxReleaseImage(&ldcObj->mesh_img_left);
    vxReleaseImage(&ldcObj->mesh_img_right);
}

void SDELDC_delete_ldc(LDCObj *ldcObj)
{
    if (ldcObj->node_left != NULL)
    {
        vxReleaseNode(&ldcObj->node_left);
    }

    if (ldcObj->node_right != NULL)
    {
        vxReleaseNode(&ldcObj->node_right);
    }

    if (ldcObj->leftLUT != NULL)
    {
        free(ldcObj->leftLUT);
        ldcObj->leftLUT = NULL;
    }

    if (ldcObj->rightLUT != NULL)
    {
        free(ldcObj->rightLUT);
        ldcObj->rightLUT = NULL;
    }
}

vx_status SDELDC_create_graph_ldc(vx_graph graph,
                                  LDCObj *ldcObj, 
                                  vx_image leftInputImg,
                                  vx_image leftOutputImg,
                                  vx_image rightInputImg,
                                  vx_image rightOutputImg)
{
    vx_status vxStatus = VX_SUCCESS;

    // LDC node for left image
    ldcObj->node_left = tivxVpacLdcNode(graph,
                                        ldcObj->config, NULL,
                                        ldcObj->region_config,
                                        ldcObj->mesh_config,
                                        ldcObj->mesh_img_left,
                                        NULL,
                                        leftInputImg,
                                        leftOutputImg,
                                        NULL);

    if (ldcObj->node_left == NULL)
    {
        LOG_ERROR("tivxVpacLdcNode() failed.\n");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxSetNodeTarget(ldcObj->node_left, VX_TARGET_STRING, TIVX_TARGET_VPAC_LDC1);
        vxSetReferenceName((vx_reference)ldcObj->node_left,
                           "LDCNode_Left");

        // LDC node for right image
        ldcObj->node_right = tivxVpacLdcNode(graph,
                                             ldcObj->config,
                                             NULL,
                                             ldcObj->region_config,
                                             ldcObj->mesh_config,
                                             ldcObj->mesh_img_right,
                                             NULL,
                                             rightInputImg,
                                             rightOutputImg,
                                             NULL);

        if (ldcObj->node_right == NULL)
        {
            LOG_ERROR("tivxVpacLdcNode() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxSetNodeTarget(ldcObj->node_right, VX_TARGET_STRING, TIVX_TARGET_VPAC_LDC1);
        vxSetReferenceName((vx_reference)ldcObj->node_right,
                           "LDCNode_Right");
    }

    return vxStatus;
}

