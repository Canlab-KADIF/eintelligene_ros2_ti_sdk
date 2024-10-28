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

#include <stdio.h>

/* Module headers. */
#include <cm_displarity_utils.h>

namespace ti_core_common
{
#define NUM_FRAC_BITS        4
#define SDE_DISPARITY_OFFSET 3

static const unsigned char SDE_falseColorLUT_RGB[3][260] = {
    {128,64,0,24,33,28,24,21,18,14,10,6,3,1,2,2,2,3,2,3,2,2,2,2,3,3,3,2,2,2,3,3,2,3,1,3,3,2,2,3,2,3,3,2,2,3,2,2,3,3,3,3,2,2,4,2,3,3,2,3,3,2,2,3,3,3,2,2,3,2,2,3,1,3,2,3,2,3,3,3,2,2,2,2,3,2,3,2,3,3,3,3,2,2,2,3,2,3,2,4,2,1,3,2,2,2,3,3,3,2,2,2,1,8,13,20,26,31,38,44,50,56,63,67,74,81,86,93,99,104,110,117,123,129,136,140,147,155,159,166,172,177,183,191,196,202,209,214,219,225,231,238,244,249,255,254,255,255,255,255,255,255,255,255,254,255,255,254,255,255,255,255,255,255,255,255,254,255,255,255,255,255,255,255,255,255,255,255,255,255,254,255,255,255,255,255,255,255,255,255,255,255,254,255,255,254,255,255,255,255,255,255,255,254,254,255,254,255,255,255,254,255,255,255,255,255,255,255,255,255,255,255,255,255,255,254,255,255,255,254,255,255,255,254,255,254,255,255,255,255,255,254,255,255,255,255,255,255,254,255},
    {128,64,0,4,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,6,12,19,25,32,37,43,51,57,63,70,76,82,89,95,101,109,115,120,127,133,140,146,152,158,165,172,178,184,190,196,204,209,216,222,229,236,241,247,254,254,254,255,254,254,255,254,254,255,254,255,254,254,254,254,253,254,253,254,254,253,255,253,253,254,254,254,254,254,254,254,253,254,253,254,254,253,254,254,254,254,253,254,253,254,254,253,254,254,254,253,254,254,254,254,253,254,253,254,255,254,254,254,254,254,254,255,254,255,254,254,255,254,254,254,255,254,255,255,255,255,255,252,249,247,244,241,239,237,234,231,230,227,225,222,219,217,215,211,209,207,205,201,200,198,195,192,189,187,184,181,179,177,174,171,169,168,164,162,160,157,154,152,150,147,144,142,139,138,135,132,130,126,124,122,120,116,114,112,109,107,105,100,97,94,90,87,83,81,76,73,70,67,63,59,57,52,49,45,43,39,35,31,29,25,21,18,15,11,7,4,1,0,0,1,0,1,0,1,1,0,1,1,1,1,1,255},
    {128,64,0,74,96,101,104,108,113,116,120,125,129,135,142,148,153,160,166,174,179,185,192,198,205,211,217,224,230,235,242,248,255,255,255,255,255,254,255,255,255,255,255,254,253,255,255,255,254,255,255,255,255,255,255,255,254,254,255,255,255,255,254,254,255,255,255,255,255,255,255,255,255,249,242,236,231,224,217,210,205,199,192,186,179,173,169,162,155,149,144,138,130,123,117,112,105,99,91,87,80,73,67,60,54,48,41,35,28,23,17,9,2,5,4,4,3,3,4,3,3,2,3,4,4,4,4,4,3,3,2,3,3,2,5,4,4,4,3,4,3,3,2,3,3,4,4,4,4,3,3,4,3,3,2,2,3,4,5,2,3,4,5,2,3,4,3,3,4,4,3,3,4,3,3,3,4,3,4,3,4,3,3,4,2,3,3,4,3,4,3,2,3,4,3,2,3,4,4,3,3,4,2,3,4,3,2,3,4,2,2,3,4,2,3,2,2,3,3,2,2,3,2,2,3,3,2,2,3,3,2,2,3,3,2,2,3,2,2,2,3,2,2,2,3,9,16,23,27,34,40,48,53,59,66,73,77,85,89,255}
};

void CM_visualizeSdeDisparity(tivx_sde_disparity_vis_params_t  *params,
                              int16_t                          *disparity,
                              uint8_t                          *disparity_rgb,
                              int32_t                           disparity_width,
                              int32_t                           disparity_height,
                              int32_t                           disparity_stride,
                              int32_t                           disparity_rgb_stride)
{
    int32_t i, j, value;
    int16_t local_max;
    int16_t local_min;
    int32_t winWidth  = disparity_width;
    int32_t winHeight = disparity_height;

    local_min = ((int16_t) params->disparity_min) * -3;
    local_max = (((int16_t) params->disparity_max) * 64) + 63;

    float scaleFactor = (float)((1 << NUM_FRAC_BITS) * (local_max - local_min)) / 255;

    int8_t sign;
    int32_t outDisparity;

    uint8_t valid;
    uint8_t * R, *G, *B;

    int16_t pixDisparity;

    // create disparity_rgb
    R = (uint8_t *)disparity_rgb;
    G = (uint8_t *)disparity_rgb + 1;
    B = (uint8_t *)disparity_rgb + 2;

    // create the falseColor map
    scaleFactor = ((float)1.0) / scaleFactor;

    for (j= 0; j < winHeight; j++)
    {
        for (i= 0; i < winWidth; i++)
        {
            // In operation mode, minimum disparity should be non-negative,
            // so sign bit can be ignored
            pixDisparity = disparity[i];
            sign = (pixDisparity >> 15) == 0 ? 1: -1;
            outDisparity = (pixDisparity >> 3) & 0xFFF;
            outDisparity *= sign;

            // check if disparity is valid based on confidence threshold
            valid = ((pixDisparity & 0x3) >= params->vis_confidence);

            // Shift disparity so that minimum disparity and unknown disparity are both zero
            value = (int)(outDisparity * scaleFactor * valid) + SDE_DISPARITY_OFFSET;

            *R = (unsigned char)(SDE_falseColorLUT_RGB[0][value]);
            *G = (unsigned char)(SDE_falseColorLUT_RGB[1][value]);
            *B = (unsigned char)(SDE_falseColorLUT_RGB[2][value]);

            R += 3;
            G += 3;
            B += 3;
        }

        R += (disparity_rgb_stride - 3*winWidth);
        G += (disparity_rgb_stride - 3*winWidth);
        B += (disparity_rgb_stride - 3*winWidth);

        disparity += disparity_stride;
    }
}

int32_t CM_allocLdcLut(char        *leftLutFileName,
                       char        *rightLutFileName,
                       uint8_t     *leftLUT,
                       uint8_t     *rightLUT,
                       uint32_t     lutSize)
{
    uint32_t size;
    FILE * lfp, *rfp;

    if ((lfp = fopen(leftLutFileName, "rb")) == NULL)
    {
        LOG_ERROR("Read Left LUT file Failed.\n");
        return -1;
    }

    if ((rfp = fopen(rightLutFileName, "rb")) == NULL)
    {
        LOG_ERROR("Read Right LUT file Failed.\n");
        fclose(lfp);
        return -1;
    }

    if ((size = fread(leftLUT,  1, lutSize, lfp)) != lutSize)
    {
        LOG_ERROR("Read left LUT Failed.\n");
        fclose(lfp);
        fclose(rfp);
        return -1;
    }

    if ((size = fread(rightLUT,  1, lutSize, rfp)) != lutSize)
    {
        LOG_ERROR("Read right LUT Failed.\n");
        fclose(lfp);
        fclose(rfp);
        return -1;
    }

    fclose(lfp);
    fclose(rfp);

    return 0;
}

} // namespace ti_core_common

