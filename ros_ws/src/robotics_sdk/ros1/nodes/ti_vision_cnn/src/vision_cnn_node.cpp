/*
 *
 * Copyright (c) 2020 Texas Instruments Incorporated
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
 * licensed and provided to you in appCntxtect code.
 *
 * If software source code is provided to you, modification and redistribution of the
 * source code are permitted provided that the following conditions are met:
 *
 * *       any redistribution and use of the source code, including any resulting derivative
 * works, are licensed by TI for use only with TI Devices.
 *
 * *       any redistribution and use of any appCntxtect code compiled from the source code
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

#include <vector>

#include <cv_bridge/cv_bridge.h>

#include <vision_cnn_node.h>

VisionCnnNode::VisionCnnNode(ros::NodeHandle &nodeHdl,
                             ros::NodeHandle &privNodeHdl):
    m_nodeHdl(nodeHdl),
    m_privNodeHdl(privNodeHdl),
    m_imgTrans(new ImgTrans(nodeHdl))
{
    vx_status   vxStatus;

    // Initialize the CNN context
    vxStatus = this->init();

    if (vxStatus != (vx_status)VX_SUCCESS)
    {
        ROS_INFO("Application Initialization failed.");
        exit(-1);
    }

    // Cache the input and output image sizes
    m_inputImgWidth  = m_cntxt->inputImageWidth;
    m_inputImgHeight = m_cntxt->inputImageHeight;
    m_outImgWidth    = m_cntxt->outImageWidth;
    m_outImgHeight   = m_cntxt->outImageHeight;

    // launch the publisher threads
    auto pubThread = std::thread([this]{publisherThread();});
    pubThread.detach();
}

void VisionCnnNode::subscriberThread()
{
    std::string topicName;
    bool        status;

    // Query the topicname to subscribe to
    status = m_privNodeHdl.getParam("input_topic_name", topicName);

    if (status == false)
    {
        ROS_INFO("Config parameter 'input_topic_name' not found.");
        exit(-1);
    }

    m_sub = new ImgSub(m_nodeHdl, topicName, 1);

    if (m_sub == nullptr)
    {
        ROS_ERROR("new ImgSub() failed.");
        exit(-1);
    }

    m_conObj = m_sub->registerCallback(&VisionCnnNode::imgCb, this);

    ROS_INFO_STREAM("Subscribed to the topic: " << topicName);
}

void VisionCnnNode::publisherThread()
{
    std::string rectImgTopic;
    std::string rectImgFrame;
    std::string outTensorTopic;
    bool        status;
    bool        latch = true;

    status = m_privNodeHdl.getParam("rectified_image_topic", rectImgTopic);
    if (status == false)
    {
        ROS_ERROR("Config parameter 'rectified_image_topic' not found.");
        exit(-1);
    }

    status = m_privNodeHdl.getParam("rectified_image_frame_id", rectImgFrame);
    if (status == false)
    {
        ROS_ERROR("Config parameter 'rectified_image_frame_id' not found.");
        exit(-1);
    }

    // rectified image in YUV420 (NV12)
    m_rectImageSize = 1.5 * m_inputImgWidth * m_inputImgHeight;
    m_rectImagePubData.data.assign(m_rectImageSize, 0);
    m_rectImagePubData.width    = m_inputImgWidth;
    m_rectImagePubData.height   = m_inputImgHeight;
    m_rectImagePubData.step     = m_inputImgWidth;
    m_rectImagePubData.encoding = "yuv420";
    m_rectImagePubData.header.frame_id = rectImgFrame;

    // Create the publisher for the rectified image
    m_rectImgPub = m_imgTrans->advertise(rectImgTopic, latch);
    ROS_INFO_STREAM("Created Publisher for topic: " << rectImgTopic);

    status = m_privNodeHdl.getParam("vision_cnn_tensor_topic", outTensorTopic);
    if (status == false)
    {
        ROS_ERROR("Config parameter 'vision_cnn_tensor_topic' not found.");
        exit(-1);
    }

    m_outTensorSize = m_cntxt->outTensorSize;
    m_outTensorPubData.data.assign(m_outTensorSize, 0);
    m_outTensorPubData.width    = m_outImgWidth;
    m_outTensorPubData.height   = m_outImgHeight;
    m_outTensorPubData.step     = m_outImgWidth;
    m_outTensorPubData.encoding = "mono8";

    m_taskType = m_cntxt->postProcObj->getTaskType();

    if (m_taskType == "segmentation")
    {
        // Create the publisher for the output tensor
        m_outTensorPub = m_imgTrans->advertise(outTensorTopic, latch);
    }
    else if (m_taskType == "detection")
    {
        // Create the publisher for the rectified image
        m_odPub = m_nodeHdl.advertise<Detection2D>(outTensorTopic, 1);
    }
    else
    {
        ROS_ERROR("Unsupported taskType");
        exit(-1);
    }

    ROS_INFO_STREAM("Created Publisher for topic: " << outTensorTopic);

    // To gauranttee all publisher objects are set up before running
    // subscriberThread()
    // launch the subscriber thread
    auto subThread = std::thread([this]{subscriberThread();});
    subThread.detach();

    // Start processing the output from the inference chain
    processCompleteEvtHdlr();

    // Shutdown the publishers
    m_rectImgPub.shutdown();

    m_outTensorPub.shutdown();
}

void VisionCnnNode::imgCb(const ImageConstPtr& imgPtr)
{
    if (m_cntxt->state == VISION_CNN_STATE_INIT)
    {
        VISION_CNN_run(m_cntxt, imgPtr->data.data(), imgPtr->header.stamp.toNSec());
    }
    else
    {
        m_conObj.disconnect();
    }
}

void VisionCnnNode::processCompleteEvtHdlr()
{
    ROS_INFO("%s Launched.", __FUNCTION__);

    while (m_cntxt->exitOutputThread == false)
    {
        vx_reference    outRef{};
        vx_status       vxStatus;

        /* Wait for the data ready semaphore. */
        if (m_cntxt->outputCtrlSem)
        {
            m_cntxt->outputCtrlSem->wait();
        }

        if (m_cntxt->exitOutputThread == true)
        {
            break;
        }

        VISION_CNN_getOutBuff(m_cntxt,
                             &m_cntxt->vxInputDisplayImage,
                             &outRef,
                             &m_cntxt->outTimestamp);

        // Extract rectified right image data (YUV420 or NV12)
        vxStatus = CM_extractImageData(m_rectImagePubData.data.data(),
                                       m_cntxt->vxInputDisplayImage,
                                       m_rectImagePubData.width,
                                       m_rectImagePubData.height,
                                       2,
                                       0);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            ROS_ERROR("CM_extractImageData() failed.");
        }

        // Extract raw tensor data
        vxStatus = CM_extractTensorData(m_outTensorPubData.data.data(),
                                        (vx_tensor)outRef,
                                        m_outTensorSize);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            ROS_ERROR("CM_extractTensorData() failed.");
        }

        // Release the output buffers
        VISION_CNN_releaseOutBuff(m_cntxt);

        if (vxStatus == (vx_status)VX_SUCCESS)
        {
            ros::Time time;
            time.fromNSec(m_cntxt->outTimestamp);

            // Publish Rectified (Right) image
            m_rectImagePubData.header.stamp = time;
            m_rectImgPub.publish(m_rectImagePubData);

            // Publish output tensor
            if (m_taskType == "segmentation")
            {
                m_outTensorPubData.header.stamp = time;
                m_outTensorPub.publish(m_outTensorPubData);
            }
            else // m_taskType == "detection"
            {
                float          *data;
                Detection2D     det;

                data             = reinterpret_cast<float *>(m_outTensorPubData.data.data());
                det.header.stamp = time;
                det.num_objects  = static_cast<int32_t>(*data++);
                det.bounding_boxes.assign(det.num_objects, BoundingBox2D{});

                for (int32_t i = 0; i < det.num_objects; i++)
                {
                    auto &box = det.bounding_boxes[i];

                    box.xmin       = static_cast<int32_t>(*data++);
                    box.ymin       = static_cast<int32_t>(*data++);
                    box.xmax       = static_cast<int32_t>(*data++);
                    box.ymax       = static_cast<int32_t>(*data++);
                    box.label_id   = static_cast<int32_t>(*data++);
                    box.confidence = *data++;
                }

                m_odPub.publish(det);
            }
        }
    }

    ROS_INFO("%s Exiting.", __FUNCTION__);
}

VisionCnnNode::~VisionCnnNode()
{
    if (m_imgTrans)
    {
        delete m_imgTrans;
    }

    if (m_sub)
    {
        delete m_sub;
    }

    if (m_cntxt)
    {
        delete m_cntxt;
    }
}

void VisionCnnNode::readParams()
{
    std::string                     str;
    bool                            status;
    int32_t                         tmp;

    /* Get LUT file path information. */
    status = m_privNodeHdl.getParam("lut_file_path", str);

    if (status == false)
    {
        ROS_INFO("Config parameter 'lut_file_path' not found.");
        exit(-1);
    }

    snprintf(m_cntxt->ldcLutFilePath, VISION_CNN_MAX_LINE_LEN-1, "%s", str.c_str());

    /* Get TIDL network path information. */
    status = m_privNodeHdl.getParam("dl_model_path", str);

    if (status == false)
    {
        ROS_INFO("Config parameter 'dl_model_path' not found.");
        exit(-1);
    }

    snprintf(m_cntxt->dlModelPath, VISION_CNN_MAX_LINE_LEN-1, "%s", str.c_str());

    /* Get input image width information. */
    m_privNodeHdl.param("width", m_cntxt->inputImageWidth, VISION_CNN_DEFAULT_IMAGE_WIDTH);

    /* Get input image height information. */
    m_privNodeHdl.param("height", m_cntxt->inputImageHeight, VISION_CNN_DEFAULT_IMAGE_HEIGHT);

    /* Get input image format information. */
    status = m_privNodeHdl.param("image_format", tmp, 0);

    if (status == false)
    {
        ROS_INFO("Config parameter 'image_format' not found.");
        exit(-1);
    }

    if (tmp == CM_IMG_FORMAT_UYVY)
    {
        m_cntxt->inputImageFormat = VX_DF_IMAGE_UYVY;
    }
    else if (tmp == CM_IMG_FORMAT_NV12)
    {
        m_cntxt->inputImageFormat = VX_DF_IMAGE_NV12;
    }
    else
    {
        ROS_INFO("Config parameter 'image_format' not supported.");
        exit(-1);
    }

    /* Get enable ldc node flag */
    m_privNodeHdl.param("enable_ldc_node", tmp, 1);
    m_cntxt->enableLdcNode = (bool)tmp;

    if (m_cntxt->enableLdcNode == 0 && m_cntxt->inputImageFormat != VX_DF_IMAGE_NV12)
    {
        ROS_INFO("LDC Node can be bypassed only when input is NV12.");
        exit(-1);
    }

    /* Get class count information. */
    status = m_privNodeHdl.getParam("num_classes", tmp);

    if (status == false)
    {
        ROS_INFO("Config parameter 'num_classes' not found.");
        exit(-1);
    }

    m_cntxt->numClasses = (int16_t)tmp;

    /* Get interactive mode flag information. */
    m_privNodeHdl.param("is_interactive", tmp, 0);
    m_cntxt->is_interactive = (uint8_t)tmp;

    /* Get pipeline depth information. */
    m_privNodeHdl.param("pipeline_depth", tmp, 1);
    m_cntxt->pipelineDepth = (uint8_t)tmp;

    /* Get graph export flag information. */
    m_privNodeHdl.param("exportGraph", tmp, 0);
    m_cntxt->exportGraph = (uint8_t)tmp;

    /* Get perf export flag information. */
    m_privNodeHdl.param("exportPerfStats", tmp, 0);
    m_cntxt->exportPerfStats = (uint8_t)tmp;

    /* Get perf export flag information. */
    m_privNodeHdl.param("detVizThreshold", m_cntxt->detVizThreshold, 0.5f);

    /* Get real-time logging enable information. */
    m_privNodeHdl.param("rtLogEnable", tmp, 0);
    m_cntxt->rtLogEnable = (uint8_t)tmp;

    /* set the ti_logger level */
    m_privNodeHdl.param("log_level", tmp, static_cast<int32_t>(ERROR));
    logSetLevel((LogLevel) tmp);

    return;
}

vx_status VisionCnnNode::init()
{
    vx_status           vxStatus;

    vxStatus = VX_SUCCESS;

    //tivx_set_debug_zone(VX_ZONE_INFO);

    m_cntxt = new VISION_CNN_Context();

    if (m_cntxt == nullptr)
    {
        ROS_ERROR("new VISION_CNN_Context() failed.");
        vxStatus = VX_FAILURE;
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Read the node parameters. */
        readParams();

        /* Initialize the Application context */
        vxStatus = VISION_CNN_init(m_cntxt);

        if (vxStatus != (vx_status)VX_SUCCESS)
        {
            ROS_ERROR("VISION_CNN_init() failed.");
        }
    }

    if (vxStatus == (vx_status)VX_SUCCESS)
    {
        /* Launch processing threads. */
        VISION_CNN_launchProcThreads(m_cntxt);
    }

    return vxStatus;
}

void VisionCnnNode::sigHandler(int32_t  sig)
{
    (void)sig;

    if (m_cntxt)
    {
        VISION_CNN_intSigHandler(m_cntxt);
    }
}
