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

#ifndef _CM_PRE_PROCESS_IMAGE_H_
#define _CM_PRE_PROCESS_IMAGE_H_

/* Module headers. */
#include <cm_common.h>

/**
 * \defgroup group_ticore_pre_proc TI Robotics SDK Image Pre-processing
 *
 * \brief Class providing interface for generic pre-processing logic.
 *
 * \ingroup group_ticore_base
 */

namespace ti_core_common
{
    /** Base class for images based post-processing. This class forms as a base
     * class for different concrete post-processing a;lgorithms. This does not
     * provide polymorphic operations since the language does not allow virtual
     * functions that are abstract and templated.
     *
     * The design is that this class holds common data across different objects
     * and provides helpe functions for parsing and storing configuration data.
     * Any configuration specific data needed beyoond this basic capability will
     * be handled by the sub-classes as needed.
     *
     * \ingroup group_ticore_pre_proc
     */
    class CmPreprocessImage
    {
        public:
            /** Constructor.
             *
             * @param config Configuration information not present in YAML
             */
            CmPreprocessImage(const PreprocessImageConfig &config);

            /** Function operator
             *
             * This is the heart of the class. The application uses this
             * interface to execute the functionality provided by this class.
             */
            virtual int32_t operator()(const void **inData, VecDlTensorPtr &outData);

            /** Destructor. */
            virtual ~CmPreprocessImage();

            /** Factory method for making a specifc pre-process object based on the
             * configuration passed.
             *
             * @param config   Configuration information not present in YAML
             * @returns A valid pre-process object if success. A nullptr otherwise.
             */
            static CmPreprocessImage* makePreprocessImageObj(const PreprocessImageConfig  &config);

        private:
            /**
             * Assignment operator.
             *
             * Assignment is not required and allowed and hence prevent
             * the compiler from generating a default assignment operator.
             */
            CmPreprocessImage & operator=(const CmPreprocessImage& rhs) = delete;

        private:
            /** Configuration information. */
            const PreprocessImageConfig m_config;
    };

} // namespace ti_core_common

#endif /* _CM_PRE_PROCESS_IMAGE_H_ */
