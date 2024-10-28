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

#ifndef _CM_POST_PROCESS_IMAGE_OBJECT_DETECT_H_
#define _CM_POST_PROCESS_IMAGE_OBJECT_DETECT_H_

/* Module headers. */
#include <cm_post_process_image.h>

/**
 * \defgroup group_ticore_obj_detect Object Detection post-processing
 *
 * \brief Class implementing the image based object detection post-processing
 *        logic.
 *
 * \ingroup group_ticore_post_proc
 */

namespace ti_core_common
{
    /** Post-processing for image based object detection.
     *
     * \ingroup group_ticore_obj_detect
     */
    class CmPostprocessImageObjDetect : public CmPostprocessImage
    {
        public:
            /** Constructor.
             *
             * @param config Configuration information not present in YAML
             */
            CmPostprocessImageObjDetect(const PostprocessImageConfig  &config);

            /** Function operator
             *
             * This is the heart of the class. The application uses this
             * interface to execute the functionality provided by this class.
             *
             * @param inData  Input data frame
             * @param outData Output data frame with results overlaid
             * @param results Detection output results from the inference
             */
            void *operator()(void              *inData,
                             void              *outData,
                             VecDlTensorPtr    &results);

            /** Destructor. */
            ~CmPostprocessImageObjDetect();

        private:
            /** Multiplicative factor to be applied to X co-ordinates. */
            float                   m_scaleX{1.0f};

            /** Multiplicative factor to be applied to Y co-ordinates. */
            float                   m_scaleY{1.0f};

        private:
            /**
             * Assignment operator.
             *
             * Assignment is not required and allowed and hence prevent
             * the compiler from generating a default assignment operator.
             */
            CmPostprocessImageObjDetect &
                operator=(const CmPostprocessImageObjDetect& rhs) = delete;
    };

} // namespace ti_core_common

#endif /* _CM_POST_PROCESS_IMAGE_OBJECT_DETECT_H_ */

