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
#include <cm_preproc_node_cntxt.h>

/**
 * \brief Constant for representing the preproc output channel count.
 * \ingroup group_applib_common_preproc
 */
#define CM_PREPROC_OUTPUT_NUM_CHANS         (3U)

namespace ti_core_common 
{

static void CM_preProcNodeCntxtSetParams(
        CM_PreProcNodeCntxt            *preProcObj,
        const CM_PreProcCreateParams   *createParams)
{
    tivxImgPreProcParams   *imgParams;
    int32_t                 i;

    preProcObj->pipelineDepth = createParams->pipelineDepth;

    imgParams = &preProcObj->imgParams;

    for (i = 0; i < 4; i++ )
    {
        imgParams->pad_pixel[i] = 0;
    }

    for (i = 0; i < CM_PREPROC_MAX_IMG_CHANS ; i++)
    {
        imgParams->scale_val[i]  = createParams->scale[i];
        imgParams->mean_pixel[i] = createParams->mean[i];
    }

    imgParams->skip_flag            = 0; // 0 - Do not skip
    imgParams->ip_rgb_or_yuv        = 1; // 0 - RGB; 1 - YUV
    imgParams->color_conv_flag      = 2; // TIADALG_COLOR_CONV_YUV420_BGR;
    imgParams->tidl_8bit_16bit_flag = 0; // 0 - 8 bit. 1 - 16 bit

    if (imgParams->tidl_8bit_16bit_flag == 0)
    {
        preProcObj->outDataType = VX_TYPE_INT8;
    }
    else
    {
        preProcObj->outDataType = VX_TYPE_INT16;
    }

    /* Number of time to clear the output buffer before it gets reused */
    imgParams->clear_count  = 4;
}

vx_status CM_preProcNodeCntxtInit(
        CM_PreProcNodeCntxt            *preProcObj,
        vx_context                      context,
        const CM_PreProcCreateParams   *createParams)
{
    vx_enum     configType;
    vx_status   vxStatus;

    vxStatus = VX_SUCCESS;

    if (preProcObj == NULL)
    {
        LOG_ERROR("Parameter 'preProcObj' NULL.");
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
    else if (preProcObj->state != CM_PREPROC_NODE_CNTXT_STATE_INVALID)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        CM_preProcNodeCntxtSetParams(preProcObj, createParams);

        configType = vxRegisterUserStruct(context,
                                           sizeof(tivxImgPreProcParams));

        if (!((configType >= VX_TYPE_USER_STRUCT_START) &&
             (configType <= VX_TYPE_USER_STRUCT_END)))
        {
            LOG_ERROR("vxRegisterUserStruct() failed.");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        preProcObj->vxConfig = vxCreateArray(context, configType, 1);

        if (preProcObj->vxConfig == NULL)
        {
            LOG_ERROR("vxCreateArray() failed");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = vxAddArrayItems(preProcObj->vxConfig,
                                   1,
                                   &preProcObj->imgParams,
                                   sizeof(tivxImgPreProcParams));

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxAddArrayItems() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)preProcObj->vxConfig,
                               "PreProcNodeConfig");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        preProcObj->vxOutputTensor = (vx_tensor *)
            tivxMemAlloc(sizeof(vx_tensor) * preProcObj->pipelineDepth,
                         TIVX_MEM_EXTERNAL);

        if (preProcObj->vxOutputTensor == NULL)
        {
            LOG_ERROR("Failed to allocate output tensor array");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        int32_t     i;
        vx_size     sizes[CM_PREPROC_OUTPUT_TENSORS_DIMS];

        /* First dimension is width. */
        sizes[0] = createParams->outWidth;

        /* Second domension is height. */
        sizes[1] = createParams->outHeight;

        /* Third dimension is channel count. */
        sizes[2] = CM_PREPROC_OUTPUT_NUM_CHANS;

        for (i = 0; i < preProcObj->pipelineDepth; i++)
        {
            preProcObj->vxOutputTensor[i] =
                vxCreateTensor(context,
                               CM_PREPROC_OUTPUT_TENSORS_DIMS,
                               sizes,
                               preProcObj->outDataType,
                               0);

            if (preProcObj->vxOutputTensor[i] == NULL)
            {
                LOG_ERROR("vxCreateTensor() failed");
                vxStatus = VX_FAILURE;
                break;
            }
            else
            {
                vxSetReferenceName((vx_reference)preProcObj->vxOutputTensor[i],
                                   "PreProcOutputTensor");
            }
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        preProcObj->state = CM_PREPROC_NODE_CNTXT_STATE_INIT;
    }

    return vxStatus;
}

vx_status CM_preProcNodeCntxtSetup(
        CM_PreProcNodeCntxt    *preProcObj,
        vx_context              context,
        vx_graph                graph,
        vx_image                inputImage)
{
    vx_status   vxStatus = VX_SUCCESS;

    if (preProcObj == NULL)
    {
        LOG_ERROR("Parameter 'preProcObj' NULL.");
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
    else if (preProcObj->state != CM_PREPROC_NODE_CNTXT_STATE_INIT)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        preProcObj->vxNode =
            tivxImgPreProcNode(graph,
                               preProcObj->vxConfig,
                               inputImage,
                               preProcObj->vxOutputTensor[0]);

        if (preProcObj->vxNode == NULL)
        {
            LOG_ERROR("tivxImgPreProcNode() failed");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = vxSetNodeTarget(preProcObj->vxNode,
                                   VX_TARGET_STRING,
                                   TIVX_TARGET_DSP1);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxSetNodeTarget() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)preProcObj->vxNode,
                               "PreProcNode");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        preProcObj->state = CM_PREPROC_NODE_CNTXT_STATE_SETUP;
    }

    return vxStatus;
}

vx_status CM_preProcNodeCntxtDeInit(
        CM_PreProcNodeCntxt    *preProcObj)
{
    vx_status   vxStatus = VX_SUCCESS;

    if (preProcObj == NULL)
    {
        LOG_ERROR("Parameter 'preProcObj' NULL.");
        vxStatus = VX_FAILURE;
    }
    else
    {
        if (preProcObj->state == CM_PREPROC_NODE_CNTXT_STATE_INVALID)
        {
            LOG_ERROR("Invalid state.");
            vxStatus = VX_FAILURE;
        }
        else
        {
            if (preProcObj->vxNode != NULL)
            {
                vxReleaseNode(&preProcObj->vxNode);
            }

            if (preProcObj->vxConfig != NULL)
            {
                vxReleaseArray(&preProcObj->vxConfig);
            }

            if (preProcObj->vxOutputTensor != NULL)
            {
                for (uint32_t i = 0; i < preProcObj->pipelineDepth; i++)
                {
                    if (preProcObj->vxOutputTensor[i] != NULL)
                    {
                        vxReleaseTensor(&preProcObj->vxOutputTensor[i]);
                    }
                }

                /* Release the output tensor array. */
                tivxMemFree(preProcObj->vxOutputTensor,
                            sizeof(vx_tensor) * preProcObj->pipelineDepth,
                            TIVX_MEM_EXTERNAL);

                preProcObj->vxOutputTensor = NULL;
            }
        }

        preProcObj->state = CM_PREPROC_NODE_CNTXT_STATE_INVALID;
    }

    return vxStatus;
}

} // namespace ti_core_common 


