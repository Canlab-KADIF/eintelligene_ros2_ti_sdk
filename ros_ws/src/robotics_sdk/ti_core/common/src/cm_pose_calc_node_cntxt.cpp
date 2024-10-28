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

#include "cm_pose_calc_node_cntxt.h"


namespace ti_core_common 
{


static vx_status CM_setPoseCalcCntxtDefaults(CM_PoseCalcNodeCntxt *poseCalcObj);
static vx_status CM_createInputOutputTensors(vx_context context, CM_PoseCalcNodeCntxt* poseCalcObj);


vx_status CM_poseCalcNodeCntxtInit(CM_PoseCalcNodeCntxt           *poseCalcObj,
                                   vx_context                      context,
                                   int32_t                         numOutTensors,
                                   vx_size                         outTensorDims[][CM_MAX_TENSOR_DIMS],
                                   const CM_PoseCalcCreateParams  *createParams)
{
    vx_status vxStatus  = VX_SUCCESS;
    vx_enum config_type = VX_TYPE_INVALID;

    if (poseCalcObj == NULL)
    {
        LOG_ERROR("Parameter 'poseCalcObj' NULL.");
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
    else if (poseCalcObj->state != CM_POSECALC_NODE_CNTXT_STATE_INVALID)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }    

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = CM_setPoseCalcCntxtDefaults(poseCalcObj);
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseCalcObj->pipelineDepth                  = createParams->pipelineDepth;

        poseCalcObj->inputUpsampleWtPath            = createParams->inputUpsampleWtPath;
        poseCalcObj->inputUpsampleBiasPath          = createParams->inputUpsampleBiasPath;

        poseCalcObj->inputVoxelInfoPath             = createParams->inputVoxelInfoPath;
        poseCalcObj->inputMapFeatPtPath             = createParams->inputMapFeatPtPath;
        poseCalcObj->inputMapFeatDescPath           = createParams->inputMapFeatDescPath;
        poseCalcObj->inputLensDistTablePath         = createParams->inputLensDistTablePath;

        poseCalcObj->vlParams.num_voxels            = createParams->numVoxels;
        poseCalcObj->vlParams.num_map_feat          = createParams->numMapFeat;
        poseCalcObj->vlParams.max_map_feat          = createParams->maxMapFeat;
        poseCalcObj->vlParams.max_frame_feat        = createParams->maxFrameFeat;

        poseCalcObj->vlParams.dl_width              = createParams->dlWidth;
        poseCalcObj->vlParams.dl_height             = createParams->dlHeight;

        poseCalcObj->vlParams.img_width             = createParams->outWidth;
        poseCalcObj->vlParams.img_height            = createParams->outHeight;

        poseCalcObj->vlParams.init_est[0]           = createParams->initPoseEst[0];
        poseCalcObj->vlParams.init_est[1]           = createParams->initPoseEst[1];
        poseCalcObj->vlParams.init_est[2]           = createParams->initPoseEst[2];

        poseCalcObj->vlParams.score_th              = createParams->scoreTh;

        poseCalcObj->vlParams.filter_scale_pw2      = createParams->filterScalePw2;
        poseCalcObj->vlParams.hi_res_desc_scale_pw2 = createParams->hiResDescScalePw2;
        poseCalcObj->vlParams.skip_flag             = createParams->skipFlag;

        // Based on DLR ouput info
        poseCalcObj->vlParams.score_lyr_id          = 0;
        poseCalcObj->vlParams.lo_res_desc_lyr_id    = 1;
        poseCalcObj->vlParams.score_lyr_elm_type    = TIADALG_DATA_TYPE_U08;
        poseCalcObj->vlParams.lo_res_desc_elm_type  = TIADALG_DATA_TYPE_S08;

        for (int32_t j = 0; j < numOutTensors; j++)
        {
            poseCalcObj->vlParams.tidl_tensor_startx[j] = 0;
            poseCalcObj->vlParams.tidl_tensor_pitch[j]  = outTensorDims[j][2];

            if (j == 1)
            {
                poseCalcObj->vlParams.desc_plane_size = outTensorDims[j][2]*outTensorDims[j][1];
            }
        }       
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        config_type         = vxRegisterUserStruct(context, sizeof(tivxVisualLocalizationParams));

        if ( config_type < VX_TYPE_USER_STRUCT_START || config_type > VX_TYPE_USER_STRUCT_END)
        {
            LOG_ERROR("vxRegisterUserStruct() failed");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseCalcObj->config = vxCreateArray(context, config_type, 1);
        if (poseCalcObj->config == NULL)
        { 
            LOG_ERROR("vxCreateArray() failed");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = vxAddArrayItems(poseCalcObj->config,
                                   1,
                                   &poseCalcObj->vlParams,
                                   sizeof(tivxVisualLocalizationParams));

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxAddArrayItems() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)poseCalcObj->config, "poseCalcObj_config");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = CM_createInputOutputTensors(context, poseCalcObj);
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseCalcObj->state = CM_POSECALC_NODE_CNTXT_STATE_INIT;
    }

    return vxStatus;
}



vx_status CM_poseCalcNodeCntxtSetup(CM_PoseCalcNodeCntxt    *poseCalcObj,
                                    vx_context               context,
                                    vx_graph                 graph,
                                    vx_tensor                curFeatTensor,
                                    vx_tensor                curDescTensor)
{
    vx_status vxStatus = VX_SUCCESS;

    if (poseCalcObj == NULL)
    {
        LOG_ERROR("Parameter 'poseCalcObj' NULL.");
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
    else if (curFeatTensor == NULL)
    {
        LOG_ERROR("Parameter 'curFeatTensor' NULL.");
        vxStatus = VX_FAILURE;
    }
    else if (curDescTensor == NULL)
    {
        LOG_ERROR("Parameter 'curDescTensor' NULL.");
        vxStatus = VX_FAILURE;
    }    
    else if (poseCalcObj->state != CM_POSECALC_NODE_CNTXT_STATE_INIT)
    {
        LOG_ERROR("Invalid state.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseCalcObj->node = tivxVisualLocalizationNode(graph,
                                                       poseCalcObj->config,
                                                       poseCalcObj->voxelInfo,       // voxelInfo
                                                       poseCalcObj->mapFeat,         // map_3d_points,
                                                       poseCalcObj->mapDesc,         // map_desc,
                                                       curFeatTensor,                // cur_frame_feat
                                                       curDescTensor,                // cur_frame_desc
                                                       poseCalcObj->wtTable,         // up_samp_wt,
                                                       poseCalcObj->lensTable,       // lens_dist_table,
                                                       poseCalcObj->outArgs,         // tidl out args
                                                       poseCalcObj->poseMatrix[0]);  // poseMatrix
        
        if (poseCalcObj->node == NULL)
        {
            LOG_ERROR("tivxVisualLocalizationNode() failed");
            vxStatus = VX_FAILURE;
        }

    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus = vxSetNodeTarget(poseCalcObj->node,
                                   VX_TARGET_STRING,
                                   TIVX_TARGET_DSP1);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxSetNodeTarget() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)poseCalcObj->node, "VisualLocalizationNode");
        }

    }

//    vx_user_data_object out_args   = (vx_user_data_object)vxGetObjectArrayItem((vx_object_array)tidl_out_args[0], 0);
//    vxReleaseUserDataObject(&out_args);

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        poseCalcObj->state = CM_POSECALC_NODE_CNTXT_STATE_SETUP;
    }
    

    return vxStatus;
}


vx_status CM_poseCalcNodeCntxtDeInit(CM_PoseCalcNodeCntxt *poseCalcObj)
{
    vx_status   vxStatus = VX_SUCCESS;
    uint32_t    i;

    if (poseCalcObj == NULL)
    {
        LOG_ERROR("Parameter 'poseCalcObj' NULL.");
        vxStatus = VX_FAILURE;
    }
    else
    {
        if (poseCalcObj->state == CM_POSECALC_NODE_CNTXT_STATE_INVALID)
        {
            LOG_ERROR("Invalid state.");
            vxStatus = VX_FAILURE;
        }
        else
        {
            if (poseCalcObj->node != NULL)
            {
                vxReleaseNode(&poseCalcObj->node);
            }

            if (poseCalcObj->config != NULL)
            {
                vxReleaseArray(&poseCalcObj->config);
            }

            if (poseCalcObj->poseMatrix != NULL)
            {
                for (i = 0; i < poseCalcObj->pipelineDepth; i++)
                {
                    if (poseCalcObj->poseMatrix[i] != NULL)
                    {
                        vxReleaseMatrix(&poseCalcObj->poseMatrix[i]);
                    }
                }

                tivxMemFree(poseCalcObj->poseMatrix,
                            sizeof(vx_matrix) * poseCalcObj->pipelineDepth,
                            TIVX_MEM_EXTERNAL);

                poseCalcObj->poseMatrix = NULL;
            }

            if (poseCalcObj->voxelInfo != NULL)
            {
                vxReleaseTensor(&poseCalcObj->voxelInfo);
            }

            if (poseCalcObj->mapFeat != NULL)
            {
                vxReleaseTensor(&poseCalcObj->mapFeat);
            }

            if (poseCalcObj->mapDesc != NULL)
            {
                vxReleaseTensor(&poseCalcObj->mapDesc);
            }

            if (poseCalcObj->wtTable != NULL)
            {
                vxReleaseTensor(&poseCalcObj->wtTable);
            }

            if (poseCalcObj->lensTable != NULL)
            {
                vxReleaseTensor(&poseCalcObj->lensTable);
            }

            if (poseCalcObj->outArgs != NULL)
            {
                vxReleaseUserDataObject(&poseCalcObj->outArgs);
            }
        }

        poseCalcObj->state = CM_POSECALC_NODE_CNTXT_STATE_INVALID;
    }

    return vxStatus;

}


static vx_status CM_setPoseCalcCntxtDefaults(CM_PoseCalcNodeCntxt *poseCalcObj)
{
    vx_status vxStatus = VX_SUCCESS;

    memset(&poseCalcObj->vlParams, 0, sizeof(tivxVisualLocalizationParams));

    poseCalcObj->vlParams.voxel_size[0]   = 3;
    poseCalcObj->vlParams.voxel_size[1]   = 3;
    poseCalcObj->vlParams.voxel_size[2]   = 3;

    poseCalcObj->vlParams.map_range[0][0] = -250;
    poseCalcObj->vlParams.map_range[0][1] = 250;
    poseCalcObj->vlParams.map_range[1][0] = -5;
    poseCalcObj->vlParams.map_range[1][1] = 5;
    poseCalcObj->vlParams.map_range[2][0] = -250;
    poseCalcObj->vlParams.map_range[2][1] = 250;

    poseCalcObj->vlParams.max_feat_match  = 50;
    poseCalcObj->vlParams.max_map_feat    = 5000;

    /*Scale of last layer convolution weights in power of 2. This information is present in the file tidl_net_onnx_tiad_jdakaze_pw2.bin_paramDebug.csv
    this csv files gets gnerated while importing the onnx model through TIDL.
    */
    poseCalcObj->vlParams.filter_scale_pw2 = 8; 

    /*This is the scale of descriptor in higher resolution. Data range can be found through onnx model, and float data range is [-256 to 256],
      that is why scale is 0.5 ( -1 in power of 2) to make it fixed point range in [-128 to 127] to fit in 8 signed bit
    */
    /*Acctual scale is 0.5, but wanted it to give in orignial scale. It is same to score buffer scale*/
    poseCalcObj->vlParams.hi_res_desc_scale_pw2 = -1;

    poseCalcObj->vlParams.lens_dist_table_size  = 655;
    poseCalcObj->vlParams.score_th              = 128;

    poseCalcObj->vlParams.dl_width              = 768;
    poseCalcObj->vlParams.dl_height             = 384;
    poseCalcObj->vlParams.img_width             = 1280;
    poseCalcObj->vlParams.img_height            = 720;

    poseCalcObj->vlParams.is_ip_fe              = 0;

    poseCalcObj->vlParams.fx                    = 1024.0;
    poseCalcObj->vlParams.fy                    = 1024.0;
    poseCalcObj->vlParams.cx                    = 1024.0;
    poseCalcObj->vlParams.cy                    = 512.0;

    return vxStatus;
}


static vx_status CM_createInputOutputTensors(vx_context context, CM_PoseCalcNodeCntxt* poseCalcObj)
{
    vx_status vxStatus = VX_SUCCESS;
    vx_size   output_sizes[CM_MAX_TENSOR_DIMS];
    uint32_t  i;

    /* Creating matrix for pose of size 4x4. Previous frame pose is used as estimated location for current frame*/
    poseCalcObj->poseMatrix = (vx_matrix *) tivxMemAlloc(sizeof(vx_matrix) * poseCalcObj->pipelineDepth,
                                                         TIVX_MEM_EXTERNAL);

    if (poseCalcObj->poseMatrix == NULL)
    {
        LOG_ERROR("Failed to allocate output pose matrix array");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        for ( i = 0; i < poseCalcObj->pipelineDepth; i++)
        {
            poseCalcObj->poseMatrix[i] = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 4);

            if (poseCalcObj->poseMatrix[i] == NULL)
            {
                LOG_ERROR("vxCreateMatrix() failed");
                vxStatus = VX_FAILURE;
                break;
            }
            else
            {
                vxSetReferenceName((vx_reference)poseCalcObj->poseMatrix[i],
                                   "PoseCalc_EstimatePose");
            }
        }
    }

    /* Creating map related tensors */
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        output_sizes[0] = sizeof(tiadalg_voxel_info);
        output_sizes[1] = poseCalcObj->vlParams.num_voxels;
        poseCalcObj->voxelInfo  = vxCreateTensor(context, 2, output_sizes, VX_TYPE_UINT8, 0);

        if (poseCalcObj->voxelInfo == NULL)
        {
            LOG_ERROR("vxCreateTensor() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)poseCalcObj->voxelInfo,
                               "PoseCalc_VoxelInfo");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        output_sizes[0] = sizeof(tiadalg_map_feat)>>2;
        output_sizes[1] = poseCalcObj->vlParams.num_map_feat;
        poseCalcObj->mapFeat = vxCreateTensor(context, 2, output_sizes, VX_TYPE_FLOAT32, 0);

        if (poseCalcObj->mapFeat == NULL)
        {
            LOG_ERROR("vxCreateTensor() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)poseCalcObj->mapFeat,
                               "PoseCalc_Map3DPoints");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        output_sizes[0] = sizeof(tiadalg_feat_desc) >> 1;
        output_sizes[1] = poseCalcObj->vlParams.num_map_feat;
        poseCalcObj->mapDesc = vxCreateTensor(context, 2, output_sizes, VX_TYPE_INT16, 0);

        if (poseCalcObj->mapDesc == NULL)
        {
            LOG_ERROR("vxCreateTensor() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)poseCalcObj->mapDesc,
                               "PoseCalc_MapFeatDesc");
        }
    }  

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Even if upsampling weight is not needed in external feat flow, but passing it to avoid null reference //if (obj->is_feat_comp_ext == 0x0) */
        output_sizes[0] = 7*7*64+2*64; /*size in elements, each element is int8. 7x7 filter for 64 plane and 64 biases*/
        poseCalcObj->wtTable = vxCreateTensor(context, 1, output_sizes, VX_TYPE_INT8, 0);

        if (poseCalcObj->wtTable == NULL)
        {
            LOG_ERROR("vxCreateTensor() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)poseCalcObj->wtTable,
                               "PoseCalc_UpsamplingWeights");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        output_sizes[0] = poseCalcObj->vlParams.lens_dist_table_size;
        poseCalcObj->lensTable = vxCreateTensor(context, 1, output_sizes, VX_TYPE_FLOAT32, 0);

        if (poseCalcObj->lensTable == NULL)
        {
            LOG_ERROR("vxCreateTensor() failed");
            vxStatus = VX_FAILURE;
        }
        else
        {
            vxSetReferenceName((vx_reference)poseCalcObj->lensTable,
                               "PoseCalc_LensDistTable");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        // tivxVisualLocalizationNode() needs TIDL_outArgs, which is available only with TIDL model, as a parameter
        // Since DLR model output tensor is already scaled by outArgs->scale,
        // we create a dummy outArgs, whose scale values are all equal to 1.
        vx_int32  i;
        vx_uint32 capacity;
        vx_map_id map_id;
        void     *outArgs_buffer = NULL;

        capacity             = sizeof(TIDL_outArgs);
        poseCalcObj->outArgs = vxCreateUserDataObject(context, "TIDL_outArgs", capacity, NULL);

        vxMapUserDataObject(poseCalcObj->outArgs, 0, capacity, &map_id,
                            (void **)&outArgs_buffer, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

        if (outArgs_buffer)
        {
            TIDL_outArgs *prms = (TIDL_outArgs*) outArgs_buffer;
            prms->iVisionOutArgs.size  = sizeof(TIDL_outArgs);

            for (i = 0; i < TIDL_NUM_OUT_BUFS; i++)
            {
                prms->scale[i] = 1.0;
            }
        }
        else
        {
            LOG_ERROR("Unable to allocate memory for outArgs! %d bytes\n", capacity);
            vxStatus = VX_FAILURE;
        }

        vxUnmapUserDataObject(poseCalcObj->outArgs, map_id);
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Fill tensors from files */
        CM_fill1DTensor(poseCalcObj->voxelInfo, poseCalcObj->inputVoxelInfoPath);
        CM_fill1DTensor(poseCalcObj->mapFeat,   poseCalcObj->inputMapFeatPtPath);
        CM_fill1DTensor(poseCalcObj->mapDesc,   poseCalcObj->inputMapFeatDescPath);

        /* Format of weights tensor is 7x7*64 (8b filter coefficient) + 2*64 (16b bias)*/
        CM_fill1DTensorFrom2Bin(poseCalcObj->wtTable,
                                poseCalcObj->inputUpsampleWtPath,
                                poseCalcObj->inputUpsampleBiasPath);

        if (poseCalcObj->vlParams.is_ip_fe == 0x1)
        {
            CM_fill1DTensor(poseCalcObj->lensTable, poseCalcObj->inputLensDistTablePath);
        }
    }

    return vxStatus;
}

} // namespace ti_core_common 
