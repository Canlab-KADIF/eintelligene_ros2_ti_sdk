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
#include "ss_sde_detection.h"

using namespace ti_core_common;

typedef struct SS_DETECT_Context
{
    /** OVX Node References */
    vx_context                        vxContext;

    /** OpenVX graph */
    vx_graph                          vxGraph;

    /** Point Cloud creation node */
    vx_node                           vxPCNode;

    /** Occupancy Grid based detection node */
    vx_node                           vxOGNode;

    /** Handle to the data object holding the point cloud creation node config parameters */
    vx_user_data_object               vxPCConfig;

    /** Data object holding the occupancy grid creation node config parameters */
    vx_user_data_object               vxOGConfig;

    /** Input right rectified image object */
    vx_image                          vxRightRectImage[GRAPH_MAX_PIPELINE_DEPTH];

    /** Input raw disparity map object */
    vx_image                          vxSde16BitOutput[GRAPH_MAX_PIPELINE_DEPTH];

    /** Input semantic segmentation tensor */
    vx_tensor                         vxSSMapTensor[GRAPH_MAX_PIPELINE_DEPTH];

    /** Output point cloud object */
    vx_user_data_object               vxPointCloud;

    /** Output bounding box object */
    vx_user_data_object               vx3DBoundBox[GRAPH_MAX_PIPELINE_DEPTH];

    /** Flag to indicate if the  should create and manage the graph. */
    bool                              manageGraph{false};

    /** input image/disparity width */
    int16_t                           width;

    /** input image/disparity height */
    int16_t                           height;

    /** input tensor (ss map) width */
    int16_t                           tensorWidth;

    /** input tensor (ss map) height */
    int16_t                           tensorHeight;

    /** input image format: 0: bmp (U8), 1: yuv420 */
    uint8_t                           inputFormat;

    /** Point cloud creation configuration parameters */
    tivx_ss_sde_point_cloud_params_t  pcCfg;

    /** Occupancy grid creation configuration parameters */
    tivx_ss_sde_og_detection_params_t ogCfg;

    /** Point Cloud Node Core mapping. */
    const char                       *pcNodeCore;

    /** Occpancy Grid Node Core mapping. */
    const char                       *ogNodeCore;

    /** Resource lock. */
    std::mutex                        paramRsrcMutex;

    /** graph parameter tracking */
    SS_DETECT_graphParams      paramDesc[SS_DETECT_MAX_PIPELINE_DEPTH];

    /** A queue for holding free descriptors. */
    SS_DETECT_graphParamQ      freeQ;

    /** Queue for output processing. */
    SS_DETECT_graphParamQ      outputQ;

    /** Base value to be used for any programmed VX events. */
    uint32_t                          vxEvtAppValBase;

    /** pipeline depth */
    uint8_t                           pipelineDepth;

    /** Flag to indicate if the graph should be exported
     * 0 - disable
     * 1 - enable
     */
    uint8_t                           exportGraph;

    /** Real-time logging enable.
     * 0 - disable
     * 1 - enable
     */
    uint8_t                           rtLogEnable;

    /** Performance monitoring. */
    app_perf_point_t                  ssDetectPerf;

    /** Flag to track if the performance counter has been initialized. */
    bool                              startPerfCapt;

    /** For manangeGraph = false
     *  Indicate whether input objects are created in Applib 
     */
    uint8_t                           createInputFlag;

    /** Indicate whether output objects are created in Applib */
    uint8_t                           createOutputFlag;

} SS_DETECT_Context;

static vx_status SS_DETECT_setParams(SS_DETECT_Handle        handle,
                                     SS_DETECT_createParams *createParams);
static vx_status SS_DETECT_createGraph(SS_DETECT_Handle handle);
static void      SS_DETECT_releaseGraph(SS_DETECT_Handle handle);

static void      SS_DETECT_setupNodes(SS_DETECT_Handle       handle, 
                                             vx_image        rightImage, 
                                             vx_image        disparity16, 
                                             vx_tensor       ssMap, 
                                             SS_DETECT_graphParams *gpDesc);

static void      SS_DETECT_copyImageObject(SS_DETECT_Handle handle,
                                           vx_image srcImage,
                                           vx_image dstImage);

static void      SS_DETECT_copyTensorObject(SS_DETECT_Handle handle,
                                            vx_tensor srcTensor,
                                            vx_tensor dstTensor);

int32_t          SS_DETECT_releaseParamRsrc(SS_DETECT_Context *appCntxt,
                                            uint32_t rsrcIndex);

int32_t          SS_DETECT_getFreeParamRsrc(SS_DETECT_Context *appCntxt,
                                            SS_DETECT_graphParams  **gpDesc);


int32_t SS_DETECT_process(SS_DETECT_Handle handle, vx_image rightImage, vx_image disparity16, vx_tensor ssMap)
{
    SS_DETECT_Context      *appCntxt = (SS_DETECT_Context *)handle;
    SS_DETECT_graphParams  *gpDesc;
    vx_status               vxStatus;
    int32_t                 status;

    // get free params source
    status = SS_DETECT_getFreeParamRsrc(appCntxt, &gpDesc);

    if (status < 0)
    {
        /* No free descriptors available. */
        LOG_ERROR("SS_DETECT_getFreeParamRsrc() failed.\n");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        // set up nodes
        SS_DETECT_setupNodes(handle, rightImage, disparity16, ssMap, gpDesc);

        vxStatus =
            vxGraphParameterEnqueueReadyRef(appCntxt->vxGraph,
                                            0,
                                            (vx_reference*)&gpDesc->vxRightRectImage,
                                            1);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxGraphParameterEnqueueReadyRef() failed.\n");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus =
            vxGraphParameterEnqueueReadyRef(appCntxt->vxGraph,
                                            1,
                                            (vx_reference*)&gpDesc->vxSde16BitOutput,
                                            1);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxGraphParameterEnqueueReadyRef() failed.\n");
        }
    }


    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus =
            vxGraphParameterEnqueueReadyRef(appCntxt->vxGraph,
                                            2,
                                            (vx_reference*)&gpDesc->vxSSMapTensor,
                                            1);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxGraphParameterEnqueueReadyRef() failed.\n");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxStatus =
            vxGraphParameterEnqueueReadyRef(appCntxt->vxGraph,
                                            3,
                                            (vx_reference*)&gpDesc->vx3DBoundBox,
                                            1);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("vxGraphParameterEnqueueReadyRef() failed.\n");
        }
    }

    if (vxStatus != (vx_status)VX_SUCCESS)
    {
        status = -1;
    }

    return status;

} /* SS_DETECT_process */


SS_DETECT_Handle SS_DETECT_create(SS_DETECT_createParams *createParams)
{
    SS_DETECT_Context    *appCntxt;
    SS_DETECT_Handle      handle;
    vx_status             vxStatus;

    handle = new SS_DETECT_Context();

    /* Set applib-level create parameters */
    vxStatus = SS_DETECT_setParams(handle, createParams);

    if (vxStatus != (vx_status)VX_SUCCESS)
    {
        LOG_ERROR("SS_DETECT_setParams() failed.\n");
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Create nodes and graph */
        vxStatus = SS_DETECT_createGraph(handle);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("SS_DETECT_createGraph() failed.\n");
        }
    }

    if (vxStatus != VX_SUCCESS)
    {
        delete handle;
        handle = NULL;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        appCntxt = (SS_DETECT_Context *)handle;
        appPerfPointSetName(&appCntxt->ssDetectPerf,
                            "SS+SDE Detection GRAPH");
    }

    return handle;

} /* SS_DETECT_create */

void SS_DETECT_delete(SS_DETECT_Handle *handle)
{
    if (*handle)
    {
        SS_DETECT_releaseGraph(*handle);

        delete *handle;
        *handle = NULL;
    }

    return;

} /* SS_DETECT_delete */

vx_status SS_DETECT_setParams(SS_DETECT_Handle        handle,
                              SS_DETECT_createParams *createParams)
{
    SS_DETECT_Context  *appCntxt;
    vx_status           vxStatus;

    vxStatus = VX_SUCCESS;
    appCntxt = (SS_DETECT_Context *)handle;

    /* Check whether we need to create and manage the graph. */
    if (createParams->vxGraph != NULL)
    {
        /* A graph handle has been provided. This indicates that the
         * Application has created the graph and will manage it. We just
         * need to create the nodes and attach it to the graph.
         */
        appCntxt->manageGraph     = false;
        appCntxt->vxGraph         = createParams->vxGraph;

        if (createParams->createInputFlag == 0)
        {
            for (int16_t i = 0; i < createParams->pipelineDepth; i++)
            {
                appCntxt->vxRightRectImage[i] =
                    createParams->vxRightRectImage[i];

                appCntxt->vxSde16BitOutput[i] =
                    createParams->vxSde16BitOutput[i];

                if (appCntxt->vxRightRectImage[i] == NULL)
                {
                    LOG_ERROR("vxRightRectImage NULL.\n");

                    vxStatus = VX_FAILURE;
                    break;
                }

                if (appCntxt->vxSde16BitOutput[i] == NULL)
                {
                    LOG_ERROR("vxSde16BitOutput NULL.\n");

                    vxStatus = VX_FAILURE;
                    break;
                }

                appCntxt->vxSSMapTensor[i] = createParams->vxSSMapTensor[i];

                if (appCntxt->vxSSMapTensor[i] == NULL)
                {
                    LOG_ERROR("vxSSMapTensor NULL.\n");

                    vxStatus = VX_FAILURE;
                }
            }
        }
    }
    else
    {
        /* No graph handle has been provided. We need to create and manage
         * the graph ourselves.
         */
        appCntxt->manageGraph = true;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        appCntxt->vxContext        = createParams->vxContext;
        appCntxt->pipelineDepth    = createParams->pipelineDepth;
        appCntxt->exportGraph      = createParams->exportGraph;
        appCntxt->rtLogEnable      = createParams->rtLogEnable;

        appCntxt->width            = createParams->width;
        appCntxt->height           = createParams->height;
        appCntxt->tensorWidth      = createParams->tensorWidth;
        appCntxt->tensorHeight     = createParams->tensorHeight;
        appCntxt->pcCfg            = createParams->pcCfg;
        appCntxt->ogCfg            = createParams->ogCfg;

        appCntxt->pcNodeCore       = createParams->pcNodeCore;
        appCntxt->ogNodeCore       = createParams->ogNodeCore;

        appCntxt->vxEvtAppValBase  = createParams->vxEvtAppValBase;
        appCntxt->inputFormat      = createParams->inputFormat;

        if (appCntxt->manageGraph == false)
        {
            appCntxt->createInputFlag  = createParams->createInputFlag;
            appCntxt->createOutputFlag = createParams->createOutputFlag;
        }

        if (appCntxt->pcNodeCore == NULL)
        {
            appCntxt->pcNodeCore = SS_DETECT_DEFAULT_CORE_MAPPING;
        }

        if (appCntxt->ogNodeCore == NULL)
        {
            appCntxt->ogNodeCore = SS_DETECT_DEFAULT_CORE_MAPPING;
        }
    }

    return vxStatus;
}

static vx_status  SS_DETECT_setupPipeline(SS_DETECT_Context * appCntxt)
{

    SS_DETECT_graphParams      * paramDesc;
    vx_graph_parameter_queue_params_t   q[SS_DETECT_NUM_GRAPH_PARAMS];
    uint32_t                            i;
    uint32_t                            cnt = 0;
    vx_status                           vxStatus;


    /* vxGraph Param 1 ==> graph param 0. */
    CM_addParamByNodeIndex(appCntxt->vxGraph,
                                appCntxt->vxPCNode,
                                1);
    q[cnt++].refs_list = (vx_reference*)appCntxt->vxRightRectImage;

    /* vxPCNode Param 2 ==> graph param 1. */
    CM_addParamByNodeIndex(appCntxt->vxGraph,
                                appCntxt->vxPCNode,
                                2);
    q[cnt++].refs_list = (vx_reference*)appCntxt->vxSde16BitOutput;

    /* vxPCNode Param 3 ==> graph param 2. */
    CM_addParamByNodeIndex(appCntxt->vxGraph,
                                appCntxt->vxPCNode,
                                3);
    q[cnt++].refs_list = (vx_reference*)appCntxt->vxSSMapTensor;

    /* vxOGNode Param 2 ==> graph param 3. */
    CM_addParamByNodeIndex(appCntxt->vxGraph,
                                appCntxt->vxOGNode,
                                2);
    q[cnt++].refs_list = (vx_reference*)appCntxt->vx3DBoundBox;


    for (i = 0; i < cnt; i++)
    {
        q[i].graph_parameter_index = i;
        q[i].refs_list_size        = appCntxt->pipelineDepth;
    }

    // allocate free Q
    for (i = 0; i < appCntxt->pipelineDepth; i++)
    {
        paramDesc                   = &appCntxt->paramDesc[i];
        paramDesc->vxRightRectImage = appCntxt->vxRightRectImage[i];
        paramDesc->vxSde16BitOutput = appCntxt->vxSde16BitOutput[i];
        paramDesc->vxSSMapTensor    = appCntxt->vxSSMapTensor[i];
        paramDesc->vx3DBoundBox     = appCntxt->vx3DBoundBox[i];

        appCntxt->freeQ.push(paramDesc);
    }

    vxStatus = vxSetGraphScheduleConfig(appCntxt->vxGraph,
                                        VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO,
                                        cnt,
                                        q);
    if (vxStatus == VX_SUCCESS)
    {
        /* explicitly set graph pipeline depth */
        vxStatus = tivxSetGraphPipelineDepth(appCntxt->vxGraph,
                                             appCntxt->pipelineDepth);

        if (vxStatus != VX_SUCCESS)
        {
            LOG_ERROR("tivxSetGraphPipelineDepth() failed\n");
        }
    }
    else
    {
        LOG_ERROR("vxSetGraphScheduleConfig() failed\n");
    }

    if (vxStatus == VX_SUCCESS)
    {
        vxStatus =
            tivxSetNodeParameterNumBufByIndex(appCntxt->vxPCNode,
                                              4, 
                                              appCntxt->pipelineDepth);
    }

    if (vxStatus == VX_SUCCESS)
    {
        uint32_t    appValue;
        appValue =  appCntxt->vxEvtAppValBase + SS_DETECT_GRAPH_COMPLETE_EVENT;

        vxStatus = vxRegisterEvent((vx_reference)appCntxt->vxGraph,
                                   VX_EVENT_GRAPH_COMPLETED,
                                   0,
                                   appValue);

        if (vxStatus != VX_SUCCESS)
        {
            LOG_ERROR("vxRegisterEvent() failed\n");
        }
    }

    return vxStatus;
}


vx_status SS_DETECT_createGraph(SS_DETECT_Handle handle)
{
    SS_DETECT_Context          *appCntxt;
    PTK_PointCloud             *cloud;
    PTK_Alg_StereoOG_obs3DBox  *obs3DBox;
    uint8_t                    *cloudMem;
    uint8_t                    *bbMem;
    PTK_PointCloudConfig        pcConfig;
    int32_t                     pcSize;
    int32_t                     bbSize;
    uint32_t                    i;
    vx_status                   vxStatus = VX_SUCCESS;

    appCntxt = (SS_DETECT_Context *)handle;

    /* Graph */
    if (appCntxt->manageGraph == true)
    {
        vx_df_image imgFormat;
        vx_size     input_size[3];

        input_size[0] = appCntxt->tensorWidth;
        input_size[1] = appCntxt->tensorHeight;
        input_size[2] = 1;

        if (appCntxt->inputFormat == CM_IMG_FORMAT_Y)
        {
            imgFormat = VX_DF_IMAGE_U8;
        }
        else 
        {
            imgFormat = VX_DF_IMAGE_NV12;
        }

        appCntxt->vxGraph = vxCreateGraph(appCntxt->vxContext);

        if (appCntxt->vxGraph == NULL)
        {
            LOG_ERROR("vxCreateGraph() failed.\n");
            vxStatus = VX_FAILURE;
        }

        if (vxStatus == (vx_status)VX_SUCCESS)
        {
            /* input stereo image */ 
            for (i = 0; i < appCntxt->pipelineDepth; i++)
            {
                // input right image
                appCntxt->vxRightRectImage[i] =
                    vxCreateImage(appCntxt->vxContext,
                                  appCntxt->width,
                                  appCntxt->height,
                                  imgFormat);

                if (appCntxt->vxRightRectImage[i] == NULL)
                {
                    LOG_ERROR("vxCreateImage() failed.\n");
                    vxStatus = VX_FAILURE;
                    break;
                }

                vxSetReferenceName((vx_reference)appCntxt->vxRightRectImage[i],
                                   "InputImage");

                // disparity map
                appCntxt->vxSde16BitOutput[i] =
                    vxCreateImage(appCntxt->vxContext,
                                  appCntxt->width,
                                  appCntxt->height,
                                  VX_DF_IMAGE_S16);

                if (appCntxt->vxSde16BitOutput[i] == NULL)
                {
                    LOG_ERROR("vxCreateImage() failed.\n");
                    vxStatus = VX_FAILURE;
                    break;
                }

                vxSetReferenceName((vx_reference)appCntxt->vxSde16BitOutput[i],
                                   "Stereo_OutputDisparityImageS16");

                // SS map tensor
                appCntxt->vxSSMapTensor[i] =
                    vxCreateTensor(appCntxt->vxContext,
                                   3,
                                   input_size,
                                   VX_TYPE_UINT8,
                                   0);

                if (appCntxt->vxSSMapTensor[i] == NULL)
                {
                    LOG_ERROR("vxCreateTensor() failed.\n");
                    vxStatus = VX_FAILURE;
                    break;
                }

                vxSetReferenceName((vx_reference)appCntxt->vxSSMapTensor[i],
                                   "SSMapTensor");
            }
        }
    }

    /*********************************/
    /* Point cloud creation node     */
    /*********************************/
    // PC config
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        appCntxt->vxPCConfig =
                vxCreateUserDataObject(appCntxt->vxContext,
                                       "tivx_ss_sde_point_cloud_params_t",
                                       sizeof(tivx_ss_sde_point_cloud_params_t),
                                       &appCntxt->pcCfg);

        if (appCntxt->vxPCConfig == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxSetReferenceName((vx_reference)appCntxt->vxPCConfig,
                           "PCConfig");

        // allocate point cloud memory and create point cloud object
        pcConfig.maxPoints  = appCntxt->pcCfg.cfgParams.dsWidth *
                              appCntxt->pcCfg.cfgParams.dsHeight;
        pcSize              = PTK_PointCloud_getSize(&pcConfig);
        cloudMem            = new uint8_t[pcSize];

        if (cloudMem == NULL)
        {
            LOG_ERROR("Memory allocation failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        cloud = PTK_PointCloud_init(cloudMem, &pcConfig);

        /* Lidar Point Cloud data. */
        appCntxt->vxPointCloud =
            vxCreateUserDataObject(appCntxt->vxContext,
                                   "StereoPointCloud",
                                   pcSize,
                                   cloud);

        if (appCntxt->vxPointCloud == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxSetReferenceName((vx_reference)appCntxt->vxPointCloud,
                           "StereoPointCloud");

        appCntxt->vxPCNode =
            tivxPointCloudCreationNode(appCntxt->vxGraph,
                                       appCntxt->vxPCConfig,
                                       appCntxt->vxRightRectImage[0],
                                       appCntxt->vxSde16BitOutput[0],
                                       appCntxt->vxSSMapTensor[0],
                                       appCntxt->vxPointCloud);

        if (appCntxt->vxPCNode == NULL)
        {
            LOG_ERROR("tivxPointCloudCreationNode() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxSetReferenceName((vx_reference)appCntxt->vxPCNode,
                           "PointCloudCreation");

        vxSetNodeTarget(appCntxt->vxPCNode,
                        VX_TARGET_STRING,
                        appCntxt->pcNodeCore);

        /**********************************/
        /* OG map detection node          */
        /**********************************/
        appCntxt->vxOGConfig =
        vxCreateUserDataObject(appCntxt->vxContext,
                               "tivx_ss_sde_og_detection_params_t",
                               sizeof(tivx_ss_sde_og_detection_params_t),
                               &appCntxt->ogCfg);

        if (appCntxt->vxOGConfig == NULL)
        {
            LOG_ERROR("vxCreateUserDataObject() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxSetReferenceName((vx_reference)appCntxt->vxOGConfig,
                           "OGConfig");

        bbSize = PTK_Alg_StereoOG_getObsBBSize(&appCntxt->ogCfg.ogParams);
        bbMem  = new uint8_t[bbSize];

        if (bbMem == NULL)
        {
            LOG_ERROR("Memory allocation failed.\n");
            delete [] cloudMem;
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        obs3DBox = PTK_Alg_StereoOG_initObsBB(bbMem);
        for (i = 0; i < appCntxt->pipelineDepth; i++ )
        {
            /* Lidar Point Cloud data. */
            appCntxt->vx3DBoundBox[i] =
                vxCreateUserDataObject(appCntxt->vxContext,
                                       "Obs3DBoundingBox",
                                       bbSize,
                                       obs3DBox);

            if (appCntxt->vx3DBoundBox[i] == NULL)
            {
                LOG_ERROR("vxCreateUserDataObject() failed.\n");
                vxStatus = VX_FAILURE;
                break;
            }

            vxSetReferenceName((vx_reference)appCntxt->vx3DBoundBox[i],
                               "Obs3DBoundingBox");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        appCntxt->vxOGNode =
            tivxOccupancyGridDetectionNode(appCntxt->vxGraph,
                                           appCntxt->vxOGConfig,
                                           appCntxt->vxPointCloud,
                                           appCntxt->vx3DBoundBox[0]);

        if (appCntxt->vxOGNode == NULL)
        {
            LOG_ERROR("tivxOccupancyGridDetectionNode() failed.\n");
            vxStatus = VX_FAILURE;
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxSetReferenceName((vx_reference)appCntxt->vxOGNode,
                           "OccupancyGridDetection");

        vxSetNodeTarget(appCntxt->vxOGNode,
                        VX_TARGET_STRING,
                        appCntxt->ogNodeCore);
    }

    // delete memory
    delete [] cloudMem;
    delete [] bbMem;

    if ((vxStatus == (vx_status)VX_SUCCESS) &&
        (appCntxt->manageGraph == true))
    {
        /* set up the pipeline. */
        vxStatus = SS_DETECT_setupPipeline(appCntxt);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("SS_DETECT_setupPipeline() failed.\n");
        }

        /* Verify graph */
        if (vxStatus == (vx_status)VX_SUCCESS)
        {
            vxStatus = vxVerifyGraph(appCntxt->vxGraph);

            if (vxStatus != (vx_status)VX_SUCCESS)
            {
                LOG_ERROR("vxVerifyGraph() failed.\n");
            }
        }

        if (vxStatus == (vx_status)VX_SUCCESS)
        {
            if (appCntxt->exportGraph == 1)
            {
                tivxExportGraphToDot(appCntxt->vxGraph, ".",
                                     "vx_applib_ss_sde_detection");
            }

            if (appCntxt->rtLogEnable == 1)
            {
                tivxLogRtTraceEnable(appCntxt->vxGraph);
            }
        }
    }

    return vxStatus;

} /* SS_DETECT_createGraph */


void SS_DETECT_releaseGraph(SS_DETECT_Handle handle)
{
    uint32_t           i;
    SS_DETECT_Context *appCntxt;

    appCntxt = (SS_DETECT_Context *)handle;

    vxReleaseUserDataObject(&appCntxt->vxPCConfig);
    vxReleaseUserDataObject(&appCntxt->vxOGConfig);

    if (appCntxt->manageGraph == true)
    {
        for (i = 0; i < appCntxt->pipelineDepth; i++)
        {
            vxReleaseImage(&appCntxt->vxRightRectImage[i]);
            vxReleaseImage(&appCntxt->vxSde16BitOutput[i]);
            vxReleaseTensor(&appCntxt->vxSSMapTensor[i]);
        }
    }

    vxReleaseUserDataObject(&appCntxt->vxPointCloud);
    for (i = 0; i < appCntxt->pipelineDepth; i++)
    {
        vxReleaseUserDataObject(&appCntxt->vx3DBoundBox[i]);
    }

    vxReleaseNode(&appCntxt->vxPCNode);
    vxReleaseNode(&appCntxt->vxOGNode);

    if (appCntxt->manageGraph == true)
    {
        if (appCntxt->rtLogEnable == 1)
        {
            tivxLogRtTraceDisable(appCntxt->vxGraph);
        }

        vxReleaseGraph(&appCntxt->vxGraph);
    }

    return;

} /* SS_DETECT_releaseGraph */


int32_t SS_DETECT_reset(SS_DETECT_Handle handle)
{
    SS_DETECT_Context  *appCntxt;
    appCntxt = (SS_DETECT_Context *)handle;

    /* Reset the performance capture initialization flag. */
    appCntxt->startPerfCapt = false;

    return 0;
}


void SS_DETECT_setupNodes(SS_DETECT_Handle handle,
                          vx_image rightImage,
                          vx_image disparity16,
                          vx_tensor ssMap,
                          SS_DETECT_graphParams  *gpDesc)
{
    SS_DETECT_copyImageObject(handle,  disparity16, gpDesc->vxSde16BitOutput);
    SS_DETECT_copyImageObject(handle,  rightImage,  gpDesc->vxRightRectImage);
    SS_DETECT_copyTensorObject(handle, ssMap,       gpDesc->vxSSMapTensor);

    return;
}

int32_t SS_DETECT_releaseParamRsrc(SS_DETECT_Context  *appCntxt,
                                          uint32_t            rsrcIndex)
{
    SS_DETECT_graphParams        *desc;
    std::unique_lock<std::mutex>   lock(appCntxt->paramRsrcMutex);

    desc = &appCntxt->paramDesc[rsrcIndex];

    appCntxt->outputQ.push(desc);

    return 0;
}

int32_t SS_DETECT_getFreeParamRsrc(SS_DETECT_Context       *appCntxt,
                                   SS_DETECT_graphParams   **gpDesc)
{
    std::unique_lock<std::mutex>   lock(appCntxt->paramRsrcMutex);

    /* Check if we have free og node descriptors available. */
    if (appCntxt->freeQ.empty())
    {
        return -1;
    }

    *gpDesc = appCntxt->freeQ.front();
    appCntxt->freeQ.pop();

    return 0;
}

void SS_DETECT_printStats(SS_DETECT_Handle handle)
{
    SS_DETECT_Context *appCntxt;

    appCntxt = (SS_DETECT_Context *)handle;

    tivx_utils_graph_perf_print(appCntxt->vxGraph);
    appPerfPointPrint(&appCntxt->ssDetectPerf);
    LOG_INFO_RAW("\n");
    appPerfPointPrintFPS(&appCntxt->ssDetectPerf);
    LOG_INFO_RAW("\n");

} /* SS_DETECT_printStats */


void SS_DETECT_exportStats(SS_DETECT_Handle handle)
{
    FILE *fp;
    app_perf_point_t *perf_arr[1];

    SS_DETECT_Context *appCntxt;
    appCntxt = (SS_DETECT_Context *)handle;

    perf_arr[0] = &appCntxt->ssDetectPerf;
    fp = appPerfStatsExportOpenFile(".", "SS_DETECT_datasheet");
    if (NULL != fp)
    {
        appPerfStatsExportAll(fp, perf_arr, 1);
        tivx_utils_graph_perf_export(fp, appCntxt->vxGraph);
        appPerfStatsExportCloseFile(fp);
        appPerfStatsResetAll();
    }
    else
    {
        LOG_ERROR("fp is null\n");
    }
}

int32_t SS_DETECT_getOutBuff(SS_DETECT_Handle handle,
                             vx_image *rightRectImage,
                             vx_image *disparity16,
                             vx_user_data_object * obsBB)
{
    SS_DETECT_Context       * appCntxt;
    SS_DETECT_graphParams   * desc;

    appCntxt = (SS_DETECT_Context*)handle;
    std::unique_lock<std::mutex>    lock(appCntxt->paramRsrcMutex);

    if (appCntxt->outputQ.empty())
    {
        return -1;
    }

    /* Get the descriptor. */
    desc = appCntxt->outputQ.front();


    *rightRectImage  = desc->vxRightRectImage;
    *disparity16     = desc->vxSde16BitOutput;
    *obsBB           = desc->vx3DBoundBox;

    return 0;
}

void SS_DETECT_releaseOutBuff(SS_DETECT_Handle handle)
{
    SS_DETECT_Context        *appCntxt;
    SS_DETECT_graphParams    *desc;

    appCntxt = (SS_DETECT_Context *)handle;
    std::unique_lock<std::mutex>    lock(appCntxt->paramRsrcMutex);

    if (appCntxt->outputQ.empty())
    {
        LOG_ERROR("No output buffers available.\n");

        return;
    }

    desc = appCntxt->outputQ.front();
    appCntxt->outputQ.pop();

    /* Push the descriptor into the free queue. */
    appCntxt->freeQ.push(desc);
}

void SS_DETECT_copyImageObject(SS_DETECT_Handle handle,
                               vx_image srcImage,
                               vx_image dstImage)
{
    SS_DETECT_Context         *appCntxt = (SS_DETECT_Context*)handle;
    uint8_t                   *data_ptr_src;
    vx_map_id                  map_id;
    vx_rectangle_t             rect;
    vx_imagepatch_addressing_t image_addr;
    vx_status                  vxStatus;

    rect.start_x = 0;
    rect.start_y = 0;
    rect.end_x   = appCntxt->width;
    rect.end_y   = appCntxt->height;

    // get source pointer
    vxStatus = vxMapImagePatch(srcImage,
                               &rect,
                               0,
                               &map_id,
                               &image_addr,
                               (void **)&data_ptr_src,
                               VX_READ_ONLY,
                               VX_MEMORY_TYPE_HOST,
                               VX_NOGAP_X);

    if (vxStatus != (vx_status)VX_SUCCESS)
    {
        LOG_ERROR("vxMapImagePatch() failed.\n");
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        vxUnmapImagePatch(srcImage, map_id);

        // get destination pointer
        vxCopyImagePatch(dstImage,
                         &rect,
                         0,
                         &image_addr,
                         data_ptr_src,
                         VX_WRITE_ONLY,
                         VX_MEMORY_TYPE_HOST);
    }
}

void SS_DETECT_copyTensorObject(SS_DETECT_Handle handle,
                                vx_tensor srcTensor,
                                vx_tensor dstTensor)
{
    SS_DETECT_Context * appCntxt = (SS_DETECT_Context*)handle;

    vx_map_id map_id_src, map_id_dst;
    vx_size   num_dims;
    vx_size   start[CM_MAX_TENSOR_DIMS];
    vx_size   output_strides[CM_MAX_TENSOR_DIMS];
    vx_size   output_sizes[CM_MAX_TENSOR_DIMS];
    vx_status vxStatus = VX_SUCCESS;

    uint8_t * src_buffer = NULL, *dst_buffer = NULL;

    uint32_t   i;
    uint8_t * ssSrc, *ssDst;

    start[0] = start[1] = start[2] = 0;
    output_sizes[0] = appCntxt->tensorWidth;
    output_sizes[1] = appCntxt->tensorHeight;
    output_sizes[2] = 1;
    output_strides[0] = 1;
    output_strides[1] = output_sizes[0];
    output_strides[2] = output_strides[1]*output_sizes[1];

    vxQueryTensor(srcTensor,
                  VX_TENSOR_NUMBER_OF_DIMS,
                        &num_dims,
                              sizeof(vx_size));

    if (num_dims >= CM_MAX_TENSOR_DIMS)
    {
        LOG_ERROR("Invalid number of dims read [%ld].", 
                            __FUNCTION__, __LINE__, num_dims);

        vxStatus = VX_FAILURE;
    }
    else
    {
        vxStatus = tivxMapTensorPatch(srcTensor,
                                      num_dims,
                                      start,
                                      output_sizes,
                                      &map_id_src,
                                      output_strides,
                                      (void **)&src_buffer,
                                      VX_READ_ONLY,
                                      VX_MEMORY_TYPE_HOST);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("tivxMapTensorPatch() failed.");
            vxStatus = VX_FAILURE;
        }

        vxStatus = tivxMapTensorPatch(dstTensor,
                                      num_dims,
                                      start,
                                      output_sizes,
                                      &map_id_dst,
                                      output_strides,
                                      (void **)&dst_buffer,
                                      VX_WRITE_ONLY,
                                      VX_MEMORY_TYPE_HOST);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            LOG_ERROR("tivxMapTensorPatch() failed.");
            vxStatus = VX_FAILURE;
        }
    }

    // read tensor data (semantic segmentation map)
    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        ssSrc = (uint8_t *)src_buffer;
        ssDst = (uint8_t *)dst_buffer;
        for (i = 0; i < output_sizes[1]; i++)
        {
            memcpy(ssDst, ssSrc, output_sizes[0]);
            ssSrc += output_sizes[0];
            ssDst += output_sizes[0];
        }

        tivxUnmapTensorPatch(srcTensor, map_id_src);
        tivxUnmapTensorPatch(dstTensor, map_id_dst);
    }
}

int32_t SS_DETECT_processEvent(SS_DETECT_Handle handle, vx_event_t * event)
{
    uint32_t                   numRefs;
    uint32_t                   index;
    int32_t                    status;
    vx_reference               ref;
    vx_status                  vxStatus = VX_SUCCESS;

    SS_DETECT_Context * appCntxt = (SS_DETECT_Context *)handle;

    if ((appCntxt->manageGraph == true) &&
        (event->type == VX_EVENT_GRAPH_COMPLETED))
    {
        uint32_t    i;

        if (appCntxt->startPerfCapt == false)
        {
            appCntxt->startPerfCapt = true;
        }
        else
        {
            appPerfPointEnd(&appCntxt->ssDetectPerf);
        }

        appPerfPointBegin(&appCntxt->ssDetectPerf);

        /* Node execution is complete. Deque all the parameters
         * for this node.
         */
        for (i = 0; i < SS_DETECT_NUM_GRAPH_PARAMS; i++)
        {
            vxStatus = vxGraphParameterDequeueDoneRef(appCntxt->vxGraph,
                                                      i,
                                                      &ref,
                                                      1,
                                                      &numRefs);
            if (vxStatus != VX_SUCCESS)
            {
                LOG_ERROR("vxGraphParameterDequeueDoneRef() failed\n");

                return -1;
            }
        }

        /* The last one to deque is vxOutInstMap parameter. Search and
         * identify the resource index.
         */
        index = appCntxt->pipelineDepth;
        for (i = 0; i < appCntxt->pipelineDepth; i++)
        {
            if (ref == (vx_reference)appCntxt->vx3DBoundBox[i])
            {
                index = i;
                break;
            }
        }

        if (index == appCntxt->pipelineDepth)
        {
            LOG_ERROR("Resource look up failed\n");

            return -1;
        }

        /* Mark the dequeued resource as free. */
        status = SS_DETECT_releaseParamRsrc(appCntxt, index);

        if (status < 0)
        {
            LOG_ERROR("SFMOG_releaseParamRsrc() failed.\n",
                       __FUNCTION__,
                       __LINE__);

            return -1;
        }
    }

    return 0;

} /* SS_DETECT_processEvent. */

void SS_DETECT_waitGraph(SS_DETECT_Handle handle)
{
    SS_DETECT_Context *appCntxt = (SS_DETECT_Context *)handle;

    if (appCntxt->manageGraph == true)
    {
        vxWaitGraph(appCntxt->vxGraph);

        /* Wait for the output queue to get flushed. */
        while (appCntxt->freeQ.size() != appCntxt->pipelineDepth)
        {
            std::this_thread::sleep_for (std::chrono::milliseconds(50));
        }
    }
}

vx_user_data_object SS_DETECT_get3DBBObject(SS_DETECT_Handle handle, int16_t index)
{
    SS_DETECT_Context *appCntxt = (SS_DETECT_Context *)handle;

    return appCntxt->vx3DBoundBox[index];
}

vx_node SS_DETECT_getPCNode(SS_DETECT_Handle handle)
{
    SS_DETECT_Context *appCntxt = (SS_DETECT_Context *)handle;

    return appCntxt->vxPCNode;
}

vx_node SS_DETECT_getOGNode(SS_DETECT_Handle handle)
{
    SS_DETECT_Context *appCntxt = (SS_DETECT_Context *)handle;

    return appCntxt->vxOGNode;
}
