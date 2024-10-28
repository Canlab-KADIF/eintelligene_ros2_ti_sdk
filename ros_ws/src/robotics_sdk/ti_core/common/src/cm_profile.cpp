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
#include <mutex>
#include <thread>

#include <utils/perf_stats/include/app_perf_stats.h>
#include <cm_profile.h>
#include <utils/include/edgeai_perfstats.h>

namespace ti_core_common
{
#define CM_PROFILE_MAX_NUM_RESETS   3

/**
 * Hold the processing time of different operations
 */
struct proctime {
    float average;
    uint64_t samples;
};

static map<string, struct proctime> gProcTime;

static bool     gPerfCtrlThreadRunning{false};
static bool     gStopCtrlThread{false};
static thread   gPerfCtrlThreadId;
static mutex    gMutex;
static int32_t  gResetCnt = 0;

void CM_reportProctime(string tag, float value)
{

    if (gProcTime.find(tag) == gProcTime.end()) {
        gProcTime[tag].average = value;
        gProcTime[tag].samples = 1;
        gProcTime.insert({tag, { value, 1}});
    } else {
        gProcTime[tag].average =
                (gProcTime[tag].average * gProcTime[tag].samples + value) /
                    (gProcTime[tag].samples + 1);
        gProcTime[tag].samples += 1;
    }
}

void CM_printProctime(FILE *fp)
{
    fprintf(fp, "+------------------------------------------------------------+\n");
    fprintf(fp, "|                   Average processing time                  |\n");
    fprintf(fp, "+------------------------------------------------------------+\n");

    for (auto lat : gProcTime)
    {
        string tag = lat.first;
        float avg = lat.second.average;
        uint64_t samples = lat.second.samples;

        fprintf(fp, "| %-25s:", tag.c_str());
        fprintf(fp, "%8.4f ms", avg);
        fprintf(fp, " from %7ld samples |\n", samples);
    }

    fprintf(fp, "+------------------------------------------------------------+\n");
}

void CM_resetProctime()
{
    for (auto lat : gProcTime)
    {
        string tag = lat.first;

        gProcTime[tag].average = 0;
        gProcTime[tag].samples = 0;
    }
}

int32_t CM_perfLaunchCtrlThread(const char *sub_dir)
{
    unique_lock<mutex>  lock(gMutex);

    if (!gPerfCtrlThreadRunning)
    {
        ti::utils::enableReport(true, sub_dir);
        gPerfCtrlThreadRunning = true;
    }

    return 0;
}

int32_t CM_perfStopCtrlThread()
{
    unique_lock<mutex> lock(gMutex);

    if (gPerfCtrlThreadRunning)
    {
        gStopCtrlThread = true;
        gPerfCtrlThreadRunning = false;
        ti::utils::disableReport();
        ti::utils::waitForPerfThreadExit();
        gStopCtrlThread = false;
    }

    return 0;
}

} // namespace ti_core_common
