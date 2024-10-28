cmake_minimum_required(VERSION 3.5)
project(ti_vision_cnn)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/ros1_common.cmake)

if(${BUILD_CORE_NODES})
    # Vision CNN app
    set(APP_VISION_CNN_SRCS
        src/vision_cnn_main.cpp
        src/vision_cnn_node.cpp)

    # Build nodes
    build_node(vision_cnn APP_VISION_CNN_SRCS)
endif()
