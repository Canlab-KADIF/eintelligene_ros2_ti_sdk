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

#include <cm_scaler_node_cntxt.h>

namespace ti_core_common 
{

static void CM_setCoeff(
        tivx_vpac_msc_coefficients_t   *coeff,
        uint32_t                        interpolation)
{
    uint32_t i;
    uint32_t idx;
    uint32_t weight;

    idx = 0;
    coeff->single_phase[0][idx++] = 0;
    coeff->single_phase[0][idx++] = 0;
    coeff->single_phase[0][idx++] = 256;
    coeff->single_phase[0][idx++] = 0;
    coeff->single_phase[0][idx++] = 0;

    idx = 0;
    coeff->single_phase[1][idx++] = 0;
    coeff->single_phase[1][idx++] = 0;
    coeff->single_phase[1][idx++] = 256;
    coeff->single_phase[1][idx++] = 0;
    coeff->single_phase[1][idx++] = 0;

    if (VX_INTERPOLATION_BILINEAR == interpolation)
    {
        idx = 0;
        for (i=0; i<32; i++)
        {
            weight = i<<2;
            coeff->multi_phase[0][idx++] = 0;
            coeff->multi_phase[0][idx++] = 0;
            coeff->multi_phase[0][idx++] = 256-weight;
            coeff->multi_phase[0][idx++] = weight;
            coeff->multi_phase[0][idx++] = 0;
        }
        idx = 0;
        for (i=0; i<32; i++)
        {
            weight = (i+32)<<2;
            coeff->multi_phase[1][idx++] = 0;
            coeff->multi_phase[1][idx++] = 0;
            coeff->multi_phase[1][idx++] = 256-weight;
            coeff->multi_phase[1][idx++] = weight;
            coeff->multi_phase[1][idx++] = 0;
        }
        idx = 0;
        for (i=0; i<32; i++)
        {
            weight = i<<2;
            coeff->multi_phase[2][idx++] = 0;
            coeff->multi_phase[2][idx++] = 0;
            coeff->multi_phase[2][idx++] = 256-weight;
            coeff->multi_phase[2][idx++] = weight;
            coeff->multi_phase[2][idx++] = 0;
        }
        idx = 0;
        for (i=0; i<32; i++)
        {
            weight = (i+32)<<2;
            coeff->multi_phase[3][idx++] = 0;
            coeff->multi_phase[3][idx++] = 0;
            coeff->multi_phase[3][idx++] = 256-weight;
            coeff->multi_phase[3][idx++] = weight;
            coeff->multi_phase[3][idx++] = 0;
        }
    }
    else /* STR_VX_INTERPOLATION_NEAREST_NEIGHBOR */
    {
        idx = 0;
        for (i=0; i<32; i++)
        {
            coeff->multi_phase[0][idx++] = 0;
            coeff->multi_phase[0][idx++] = 0;
            coeff->multi_phase[0][idx++] = 256;
            coeff->multi_phase[0][idx++] = 0;
            coeff->multi_phase[0][idx++] = 0;
        }
        idx = 0;
        for (i=0; i<32; i++)
        {
            coeff->multi_phase[1][idx++] = 0;
            coeff->multi_phase[1][idx++] = 0;
            coeff->multi_phase[1][idx++] = 0;
            coeff->multi_phase[1][idx++] = 256;
            coeff->multi_phase[1][idx++] = 0;
        }
        idx = 0;
        for (i=0; i<32; i++)
        {
            coeff->multi_phase[2][idx++] = 0;
            coeff->multi_phase[2][idx++] = 0;
            coeff->multi_phase[2][idx++] = 256;
            coeff->multi_phase[2][idx++] = 0;
            coeff->multi_phase[2][idx++] = 0;
        }
        idx = 0;
        for (i=0; i<32; i++)
        {
            coeff->multi_phase[3][idx++] = 0;
            coeff->multi_phase[3][idx++] = 0;
            coeff->multi_phase[3][idx++] = 0;
            coeff->multi_phase[3][idx++] = 256;
            coeff->multi_phase[3][idx++] = 0;
        }
    }
}

vx_status CM_scalerNodeCntxtInit(
        CM_ScalerNodeCntxt             *scalerObj,
        vx_context                      context,
        const CM_ScalerCreateParams    *createParams)
{
    tivx_vpac_msc_coefficients_t    coeff;
    vx_status                       vxStatus;

    vxStatus = VX_SUCCESS;

    if (scalerObj == NULL)
    {
        LOG_ERROR("Parameter 'scalerObj' NULL.");
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
    else if (scalerObj->state != CM_SCALER_NODE_CNTXT_STATE_INVALID)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        scalerObj->pipelineDepth = createParams->pipelineDepth;

        scalerObj->outImage = (vx_image *)
            tivxMemAlloc(sizeof(vx_image) * scalerObj->pipelineDepth,
                         TIVX_MEM_EXTERNAL);

        if (scalerObj->outImage == NULL)
        {
            LOG_ERROR("Failed to allocate output image array");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        for (uint32_t i = 0; i < scalerObj->pipelineDepth; i++)
        {
            scalerObj->outImage[i] = vxCreateImage(context,
                                                   createParams->width,
                                                   createParams->height,
                                                   createParams->imageType);

            if (scalerObj->outImage[i] == NULL)
            {
                LOG_ERROR("vxCreateImage() failed");
                vxStatus = VX_FAILURE;
            }
            else
            {
                vxSetReferenceName((vx_reference)scalerObj->outImage[i],
                                   "ScalerOutputImage_NV12");
            }
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Output image */
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Set the coefficients. */
        CM_setCoeff(&coeff, createParams->interpolation);

        scalerObj->vxCoeff =
            vxCreateUserDataObject(context,
                                   "tivx_vpac_msc_coefficients_t",
                                   sizeof(tivx_vpac_msc_coefficients_t),
                                   NULL);

        if (scalerObj->vxCoeff == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxCopyUserDataObject(scalerObj->vxCoeff,
                                 0,
                                 sizeof(tivx_vpac_msc_coefficients_t),
                                 &coeff,
                                 VX_WRITE_ONLY,
                                 VX_MEMORY_TYPE_HOST);
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        scalerObj->state = CM_SCALER_NODE_CNTXT_STATE_INIT;
    }

    return vxStatus;
}

vx_status CM_scalerNodeCntxtSetup(
        CM_ScalerNodeCntxt *scalerObj,
        vx_context          context,
        vx_graph            graph,
        vx_image            inputImage)
{
    vx_status   vxStatus = VX_SUCCESS;

    if (scalerObj == NULL)
    {
        LOG_ERROR("Parameter 'scalerObj' NULL.");
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
    else if (scalerObj->state != CM_SCALER_NODE_CNTXT_STATE_INIT)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        scalerObj->vxNode =
            tivxVpacMscScaleNode(graph,
                                 inputImage,
                                 scalerObj->outImage[0],
                                 NULL,
                                 NULL,
                                 NULL,
                                 NULL);

        if (scalerObj->vxNode == NULL)
        {
            LOG_ERROR("tivxVpacMscScaleNode() failed");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = vxSetNodeTarget(scalerObj->vxNode,
                                   VX_TARGET_STRING,
                                   TIVX_TARGET_VPAC_MSC1);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxSetNodeTarget() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)scalerObj->vxNode,
                               "ScalerNode");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        scalerObj->state = CM_SCALER_NODE_CNTXT_STATE_SETUP;
    }

    return vxStatus;
}

vx_status CM_scalerNodeCntxtSetCoeff(
        CM_ScalerNodeCntxt *scalerObj)
{
    vx_reference    refs[1];
    vx_status       vxStatus = VX_SUCCESS;

    if (scalerObj == NULL)
    {
        LOG_ERROR("Parameter 'scalerObj' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (scalerObj->state != CM_SCALER_NODE_CNTXT_STATE_SETUP)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        refs[0] = (vx_reference)scalerObj->vxCoeff;

        vxStatus = tivxNodeSendCommand(scalerObj->vxNode,
                                       0u,
                                       TIVX_VPAC_MSC_CMD_SET_COEFF,
                                       refs,
                                       1u);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("tivxNodeSendCommand() failed");
        }
    }

    return vxStatus;
}

vx_status CM_scalerNodeCntxtDeInit(
        CM_ScalerNodeCntxt *scalerObj)
{
    vx_status   vxStatus = VX_SUCCESS;

    if (scalerObj == NULL)
    {
        LOG_ERROR("Parameter 'scalerObj' NULL.");
        vxStatus = VX_FAILURE;
    }
    else
    {
        if (scalerObj->state == CM_SCALER_NODE_CNTXT_STATE_INVALID)
        {
            LOG_ERROR("Invalid state.");
            vxStatus = VX_FAILURE;
        }
        else
        {
            if (scalerObj->vxNode != NULL)
            {
                vxReleaseNode(&scalerObj->vxNode);
            }

            if (scalerObj->vxCoeff != NULL)
            {
                vxReleaseUserDataObject(&scalerObj->vxCoeff);
            }

            if (scalerObj->outImage != NULL)
            {
                for (uint32_t i = 0; i < scalerObj->pipelineDepth; i++)
                {
                    if (scalerObj->outImage[i] != NULL)
                    {
                        vxReleaseImage(&scalerObj->outImage[i]);
                    }
                }

                /* Release the output tensor array. */
                tivxMemFree(scalerObj->outImage,
                            sizeof(vx_image) * scalerObj->pipelineDepth,
                            TIVX_MEM_EXTERNAL);

                scalerObj->outImage = NULL;
            }
        }

        scalerObj->state = CM_SCALER_NODE_CNTXT_STATE_INVALID;
    }

    return vxStatus;
}

} // namespace ti_core_common 
