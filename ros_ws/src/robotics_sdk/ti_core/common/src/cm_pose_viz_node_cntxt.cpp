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

#include "cm_pose_viz_node_cntxt.h"

namespace ti_core_common 
{

static vx_status CM_setPoseVizCntxtDefaults(CM_PoseVizNodeCntxt *poseVizObj);


vx_status CM_poseVizNodeCntxtInit(CM_PoseVizNodeCntxt           *poseVizObj,
                                  vx_context                     context,
                                  const CM_PoseVizCreateParams  *createParams)
{
    vx_status vxStatus    = VX_SUCCESS;
    vx_enum   config_type = VX_TYPE_INVALID;
    uint32_t  i;

    if (poseVizObj == NULL)
    {
        LOG_ERROR("Parameter 'poseVizObj' NULL.");
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
    else if (poseVizObj->state != CM_POSEVIZ_NODE_CNTXT_STATE_INVALID)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }    

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = CM_setPoseVizCntxtDefaults(poseVizObj);
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseVizObj->pipelineDepth        = createParams->pipelineDepth;
        poseVizObj->topViewImgPath       = createParams->topViewImgPath;

        poseVizObj->vizParams.img_width  = createParams->outWidth;
        poseVizObj->vizParams.img_height = createParams->outHeight;
        poseVizObj->vizParams.frame_cnt  = 0;

        poseVizObj->vizParams.max_background_image_copy = 0;
                                           //createParams->pipelineDepth;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        config_type         = vxRegisterUserStruct(context, sizeof(tivxPoseVizParams));

        if (config_type < VX_TYPE_USER_STRUCT_START || config_type > VX_TYPE_USER_STRUCT_END)
        {
            LOG_ERROR("vxRegisterUserStruct() failed");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseVizObj->config = vxCreateArray(context, config_type, 1);
        if (poseVizObj->config == NULL)
        { 
            LOG_ERROR("vxCreateArray() failed");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = vxAddArrayItems(poseVizObj->config,
                                   1,
                                   &poseVizObj->vizParams,
                                   sizeof(tivxPoseVizParams));

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxAddArrayItems() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)poseVizObj->config, "poseVizObj_config");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseVizObj->outputImage = (vx_image *) tivxMemAlloc(sizeof(vx_image) * poseVizObj->pipelineDepth,
                                                            TIVX_MEM_EXTERNAL);

        if (poseVizObj->outputImage == NULL)
        {
            LOG_ERROR("Failed to allocate output image array");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseVizObj->bgImage = 
            vxCreateImage(context, poseVizObj->vizParams.img_width, poseVizObj->vizParams.img_height, VX_DF_IMAGE_NV12);

        if (poseVizObj->bgImage == NULL)
        { 
            LOG_ERROR("vxCreateImage() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)poseVizObj->bgImage, "poseViz_bgImage");

            // Load top view image
            CM_loadImage((char *)createParams->topViewImgPath, poseVizObj->bgImage);
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        for ( i = 0; i < poseVizObj->pipelineDepth; i++)
        {        
            poseVizObj->outputImage[i] = 
                vxCreateImage(context, poseVizObj->vizParams.img_width, poseVizObj->vizParams.img_height, VX_DF_IMAGE_NV12);

            if (poseVizObj->outputImage[i] == NULL)
            { 
                LOG_ERROR("vxCreateImage() failed");
                vxStatus = VX_FAILURE;
            }
            else
            {
                vxSetReferenceName((vx_reference)poseVizObj->outputImage[i], "poseViz_outputImage");
            }
        }
    } 

    // Copy top view image to outputImage
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        for ( i = 0; i < poseVizObj->pipelineDepth; i++)
        {        
             vxStatus = CM_copyImage2Image(poseVizObj->bgImage,  poseVizObj->outputImage[i]);

             if (vxStatus != (vx_status)VX_SUCCESS)
             {
                LOG_ERROR("CM_copyImage2Image() failed");
                vxStatus = VX_FAILURE;
                break;
             }
        }
    } 

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseVizObj->state = CM_POSEVIZ_NODE_CNTXT_STATE_INIT;
    }

    return vxStatus;
}



vx_status CM_poseVizNodeCntxtSetup(CM_PoseVizNodeCntxt    *poseVizObj,
                                   vx_context              context,
                                   vx_graph                graph,
                                   vx_matrix               pose)
{
    vx_status vxStatus = VX_SUCCESS;

    if (poseVizObj == NULL)
    {
        LOG_ERROR("Parameter 'poseVizObj' NULL.");
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
    else if (pose == NULL)
    {
        LOG_ERROR("Parameter 'pose' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (poseVizObj->state != CM_POSEVIZ_NODE_CNTXT_STATE_INIT)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseVizObj->node = tivxPoseVizNode(graph,
                                          poseVizObj->config,
                                          poseVizObj->bgImage,
                                          pose,
                                          poseVizObj->outputImage[0]);
        
        if (poseVizObj->node == NULL)
        {
            LOG_ERROR("tivxPoseVizNode() failed\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = vxSetNodeTarget(poseVizObj->node,
                                   VX_TARGET_STRING,
                                   TIVX_TARGET_DSP2);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxSetNodeTarget() failed\n");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)poseVizObj->node, "PoseVizNode");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseVizObj->state = CM_POSEVIZ_NODE_CNTXT_STATE_SETUP;
    }


    return vxStatus;
}


vx_status CM_poseVizNodeCntxtDeInit(CM_PoseVizNodeCntxt *poseVizObj)
{
    vx_status   vxStatus = VX_SUCCESS;
    uint32_t    i;

    if (poseVizObj == NULL)
    {
        LOG_ERROR("Parameter 'poseVizObj' NULL.");
        vxStatus = VX_FAILURE;
    }
    else
    {
        if (poseVizObj->state == CM_POSEVIZ_NODE_CNTXT_STATE_INVALID)
        {
            LOG_ERROR("Invalid state.");
            vxStatus = VX_FAILURE;
        }
        else
        {
            if (poseVizObj->node != NULL)
            {
                vxReleaseNode(&poseVizObj->node);
            }

            if (poseVizObj->config != NULL)
            {
                vxReleaseArray(&poseVizObj->config);
            }

            if (poseVizObj->bgImage != NULL)
            {
                vxReleaseImage(&poseVizObj->bgImage);
            }

            if (poseVizObj->outputImage != NULL)
            {
                for (i = 0; i < poseVizObj->pipelineDepth; i++)
                {
                    if (poseVizObj->outputImage[i] != NULL)
                    {
                        vxReleaseImage(&poseVizObj->outputImage[i]);
                    }
                }

                tivxMemFree(poseVizObj->outputImage,
                            sizeof(vx_image) * poseVizObj->pipelineDepth,
                            TIVX_MEM_EXTERNAL);

                poseVizObj->outputImage = NULL;
            }
        }

        poseVizObj->state = CM_POSEVIZ_NODE_CNTXT_STATE_INVALID;
    }

    return vxStatus;
}


#if 0
vx_status writePoseVizOutput(char* file_name, vx_image out_img)
{
    vx_status status;

    status = vxGetStatus((vx_reference)out_img);

    if (status == VX_SUCCESS)
    {
        FILE * fp = fopen(file_name,"wb");

        if (fp == NULL)
        {
            LOG_ERROR("Unable to open file %s \n", file_name);
            return (VX_FAILURE);
        }

        vx_rectangle_t rect;
        vx_imagepatch_addressing_t image_addr;
        vx_map_id map_id;
        void * data_ptr;
        vx_uint32  img_width;
        vx_uint32  img_height;
        vx_uint32  num_bytes;

        vxQueryImage(out_img, VX_IMAGE_WIDTH, &img_width, sizeof(vx_uint32));
        vxQueryImage(out_img, VX_IMAGE_HEIGHT, &img_height, sizeof(vx_uint32));

        rect.start_x = 0;
        rect.start_y = 0;
        rect.end_x = img_width;
        rect.end_y = img_height;
        status = vxMapImagePatch(out_img,
                                &rect,
                                0,
                                &map_id,
                                &image_addr,
                                &data_ptr,
                                VX_READ_ONLY,
                                VX_MEMORY_TYPE_HOST,
                                VX_NOGAP_X);

        //Copy Luma
        num_bytes = fwrite(data_ptr,1,img_width*img_height, fp);

        if (num_bytes != (img_width*img_height))
            LOG_ERROR("Luma bytes written = %d, expected = %d", num_bytes, img_width*img_height);

        vxUnmapImagePatch(out_img, map_id);

        status = vxMapImagePatch(out_img,
                                &rect,
                                1,
                                &map_id,
                                &image_addr,
                                &data_ptr,
                                VX_READ_ONLY,
                                VX_MEMORY_TYPE_HOST,
                                VX_NOGAP_X);


        //Copy CbCr
        num_bytes = fwrite(data_ptr,1,img_width*img_height/2, fp);

        if (num_bytes != (img_width*img_height/2))
            LOG_ERROR("CbCr bytes written = %d, expected = %d", num_bytes, img_width*img_height/2);

        vxUnmapImagePatch(out_img, map_id);

        fclose(fp);
    }

    return(status);
}
#endif


static vx_status CM_setPoseVizCntxtDefaults(CM_PoseVizNodeCntxt *poseVizObj)
{
    vx_status vxStatus = VX_SUCCESS;

    /* Ego localization parametrs*/
    /* Image resolution for pose visualization and other parameter for visulization*/
    poseVizObj->vizParams.img_width      = 2048;
    poseVizObj->vizParams.img_height     = 1024;
    poseVizObj->vizParams.img_num_planes = 3;

    poseVizObj->vizParams.projMat[0][0]  = -0.999974081148704;
    poseVizObj->vizParams.projMat[0][1]  = 0.0;
    poseVizObj->vizParams.projMat[0][2]  = -0.007199793802923;
    poseVizObj->vizParams.projMat[0][3]  = -158.774279623468715;

    poseVizObj->vizParams.projMat[1][0]  = -0.007199793802923;
    poseVizObj->vizParams.projMat[1][1]  = 0.0;
    poseVizObj->vizParams.projMat[1][2]  = 0.999974081148704;
    poseVizObj->vizParams.projMat[1][3]  = 6.612800677299523;

    poseVizObj->vizParams.projMat[2][0]  = 0.0;
    poseVizObj->vizParams.projMat[2][1]  = 1.0;
    poseVizObj->vizParams.projMat[2][2]  = 0.0;
    poseVizObj->vizParams.projMat[2][3]  = 300.0;

    poseVizObj->vizParams.fx             = 1024.0;
    poseVizObj->vizParams.fy             = 1024.0;
    poseVizObj->vizParams.cx             = 1024.0;
    poseVizObj->vizParams.cy             = 512.0;

    return vxStatus;
}

} // namespace ti_core_common 
