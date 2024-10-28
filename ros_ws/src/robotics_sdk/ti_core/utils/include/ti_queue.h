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
#if !defined(_TI_QUEUE_H_)
#define _TI_QUEUE_H_

#include <queue>
#include <mutex>
#include <stdexcept>

#include <ti_semaphore.h>

/**
 * \defgroup group_ticore_utils_queue Utility functions for queue
 *
 * \brief Provides Queue class for multi-thread processing of graphs
 *
 * \ingroup  group_ticore_utils
 *
 */

namespace ti::utils {

    /**
     * \brief Dummy Mutex structure
     * \ingroup group_ticore_utils_queue
     */
    struct MutexDummy
    {
        void lock(){}
        void unlock(){};
    };

    /**
     * \brief Dummy Semaphore structure
     * \ingroup group_ticore_utils_queue
     */
    struct SemaphoreDummy
    {
        SemaphoreDummy(int32_t cnt = 0){}
        void notify(){}
        void wait(){}
        bool try_wait(){ return true; }
    };

    /**
     * \brief Queue class
     * \ingroup group_ticore_utils_queue
     */
    template <typename T,
              typename MutexT = MutexDummy,
              typename SemT = SemaphoreDummy>
    class Queue
    {
        public:
            /**
             * \brief Queue class constructor
             * \ingroup group_ticore_utils_queue
             */
            Queue()
            {
            }

            /**
             * \brief Get the oldest element from a queue
             * \ingroup group_ticore_utils_queue
             */
            T *peek()
            {
                std::unique_lock<MutexT>    lock(m_mutex);
                T                          *val{nullptr};

                /* Check if we have descriptors available. */
                if (!m_q.empty())
                {
                    val = m_q.front();
                }

                return val;
            }

            /**
             * \brief Pop the oldest element from a queue
             * \ingroup group_ticore_utils_queue
             */
            T *pop()
            {
                /* Check if we have data. */
                m_sem.wait();

                std::unique_lock<MutexT>    lock(m_mutex);
                T                          *val{nullptr};

                /* Check if we have descriptors available. */
                if (!m_q.empty())
                {
                    val = m_q.front();
                    m_q.pop();
                }

                return val;
            }

           /**
             * \brief Push a value into a queue
             * \ingroup group_ticore_utils_queue
             */
            int32_t push(T *val)
            {
                std::unique_lock<MutexT>    lock(m_mutex);

                m_q.push(val);

                m_sem.notify();

                return 0;
            }

           /**
             * \brief Flush a queue
             * \ingroup group_ticore_utils_queue
             */
            void flush()
            {
                std::unique_lock<MutexT>    lock(m_mutex);

                while( !m_q.empty() )
                {
                    m_q.pop();
                }
            }

            /**
             * \brief Get the number of elements in a queue
             * \ingroup group_ticore_utils_queue
             */
            int32_t size()
            {
                std::unique_lock<MutexT>    lock(m_mutex);
                return m_q.size();
            }

        private:
            /**
             * \brief A queue for holding free descriptors
             * \ingroup group_ticore_utils_queue
             */
            std::queue<T*>  m_q;

            /**
             * \brief Resource lock. Used get/put from/to freeQ
             * \ingroup group_ticore_utils_queue
             */
            MutexT          m_mutex;

            /**
             * \brief Synchronization control
             * \ingroup group_ticore_utils_queue
             */
            SemT            m_sem{};
    };

    /**
     * \brief Single thread safe queue with no mutex and semaphore control.
     * \ingroup group_ticore_utils_queue
     */
    template <typename T>
    using SingleThreadQ = Queue<T>;

    /**
     * \brief Multi-thread safe queue with mutex control.
     * \ingroup group_ticore_utils_queue
     */
    template <typename T, typename MutexT = std::mutex>
    using MultiThreadQ = Queue<T, MutexT>;

    /**
     * \brief Multi-thread safe queue with mutex control.
     * \ingroup group_ticore_utils_queue
     */
    template <typename T,
              typename MutexT = std::mutex,
              typename SemT = Semaphore>
    using MultiThreadTaskQ = Queue<T, MutexT, SemT>;

} // namespace ti::utils

#endif // _TI_QUEUE_H_

