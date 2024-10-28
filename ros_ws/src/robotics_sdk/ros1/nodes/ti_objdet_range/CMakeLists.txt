cmake_minimum_required(VERSION 3.5)
project(ti_objdet_range)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/ros1_common.cmake)

# PCL has dependency on libusb-dev which is assumed to be installed in the container
find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

# find_package(Eigen REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIR})

if(${BUILD_CORE_NODES} AND ${SDE_AVAILABLE})
    # objdet_range ROS node
    set(OBJDET_RANGE_SRCS src/objdet_disparity_fusion.cpp)
    build_node(objdet_disparity_fusion OBJDET_RANGE_SRCS)

endif()
