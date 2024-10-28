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
#ifndef _SDE_LDC_MODULE
#define _SDE_LDC_MODULE

#include "sde_ldc.h"

/**
 * \defgroup group_applib_sde_ldc_module Stereo camera LDC node setup code 
 * \ingroup group_applib_sde_ldc
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Constant for LDC down-sampling factor
 * \ingroup group_applib_sde_ldc_module
 */
#define LDC_DS_FACTOR       (2)

/**
 * \brief Constant for LDC processing block width
 * \ingroup group_applib_sde_ldc_module
 */
#define LDC_BLOCK_WIDTH     (64)

/**
 * \brief Constant for LDC processing block height
 * \ingroup group_applib_sde_ldc_module
 */
#define LDC_BLOCK_HEIGHT    (16)

/**
 * \brief Constant to enable/disable pixel padding in LDC
 * \ingroup group_applib_sde_ldc_module
 */
#define LDC_PIXEL_PAD       (1)

/**
 * \brief Stereo LDC node context.
 *
 * \ingroup group_applib_sde_ldc_module
 */
typedef struct 
{
    /** Handle to the LDC node for left */
    vx_node                       node_left;

    /** Handle to the LDC node for right */
    vx_node                       node_right;

    /** Handle to the Mesh image for left */
    vx_image                      mesh_img_left;

    /** Handle to the Mesh image for right */
    vx_image                      mesh_img_right;

    /** LDC configuration parameters */
    tivx_vpac_ldc_params_t        params;

    /** Handle to the data object holding the LDC configuration parameters. */
    vx_user_data_object           config;

    /**  LDC mesh configuration parameters */
    tivx_vpac_ldc_mesh_params_t   mesh_params;

    /** Handle to the data object holding the LDC mesh configuration
     *  parameters.
     */
    vx_user_data_object           mesh_config;

    /** LDC region configuration parameters */
    tivx_vpac_ldc_region_params_t region_params;

    /** Handle to the data object holding the LDC region configuration
     *  parameters.
     */
    vx_user_data_object           region_config;

    /** LDC tabel width that is the same to input width */
    vx_uint32                     table_width;

    /** LDC tabel height that is the same to input height */
    vx_uint32                     table_height;

    /** LDC down-sampling factor since LDC table is down-sampled */
    vx_uint32                     ds_factor;

    /** Pointer to the LDC LUT for left */
    uint8_t                      *leftLUT;

    /** Pointer to the LDC LUT for right */
    uint8_t                      *rightLUT;

} LDCObj;

/**
 * \brief Function to initialize the stereo ldc node context.
 *
 * \param [in] context The handle to the openVX implementation context.
 * 
 * \param [in,out] ldcObj Ldc node context.
 *
 * \param [in] leftLutFileName left LUT file name
 * 
 * \param [in] rightLutFileName right LUT file name
 *
 * \param [in] width image width
 * 
 * \param [in] heigth image height
 * 
 * \return VX_SUCCESS on success
 *
 * \ingroup group_applib_sde_ldc_module
 */
vx_status SDELDC_init_ldc(vx_context context, LDCObj *ldcObj,
                                char *leftLutFileName, char *rightLutFileName,
                                uint16_t width, uint16_t height);

/**
 * \brief Function to de-initialize the ldc node context.
 *
 * \param [in,out] ldcObj Ldc node context.
 *
 * \ingroup group_applib_sde_ldc_module
 */                                
void      SDELDC_deinit_ldc(LDCObj *ldcObj);

/**
 * \brief Function to delete the ldc node context.
 *
 * \param [in,out] ldcObj Ldc node context.
 *
 * \ingroup group_applib_sde_ldc_module
 */
void      SDELDC_delete_ldc(LDCObj *ldcObj);


/**
 * \brief Function to create a stereo LDC graph
 *
 * \param [in] graph The handle to the graph this node belongs to.
 *
 * \param [in,out] ldcObj Ldc node context.
 *
 * \param [in] leftInputImg The handle to input left image
 * 
 * \param [in] leftOutputImg The handle to output left image
 * 
 * \param [in] rightInputImg The handle to input right image
 * 
 * \param [in] rightOutputImg The handle to output right image
 *
 * \ingroup group_applib_sde_ldc_module
 */
vx_status SDELDC_create_graph_ldc(vx_graph graph, LDCObj *ldcObj,
                                        vx_image leftInputImg, vx_image leftOutputImg,
                                        vx_image rightInputImg, vx_image rightOutputImg);

#ifdef __cplusplus
}
#endif

#endif
