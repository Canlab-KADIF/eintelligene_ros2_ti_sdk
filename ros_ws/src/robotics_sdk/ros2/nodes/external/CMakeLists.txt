cmake_minimum_required(VERSION 3.5)

project(ti_external)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common_platform.cmake)

if(${BUILD_CORE_NODES})
    find_package(ament_cmake REQUIRED)

    set(TI_CORE_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
    add_subdirectory(${TI_CORE_ROOT_DIR} ${TI_CORE_BUILD_DIR}/ti_core)

    ament_package()

endif()
