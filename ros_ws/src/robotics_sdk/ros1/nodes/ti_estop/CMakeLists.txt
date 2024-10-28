cmake_minimum_required(VERSION 3.5)
project(ti_estop)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/ros1_common.cmake)

if(${BUILD_CORE_NODES} AND ${SDE_AVAILABLE})
    # AMR E-Stop example
    set(APP_ESTOP_SRCS
        src/estop_main.cpp
        src/estop_node.cpp)

    # Build nodes
    build_node(estop APP_ESTOP_SRCS)
endif()
