cmake_minimum_required(VERSION 2.8.3)
project(zed_capture)

find_package(catkin REQUIRED COMPONENTS
    cmake_modules
    roscpp cv_bridge
    image_transport
    camera_info_manager
)

find_package(OpenCV REQUIRED)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(${catkin_LIBRARY_DIRS})

catkin_package(
    LIBRARIES
    CATKIN_DEPENDS
)

# node
add_executable(zed_capture
    src/zed_capture_node.cpp
    src/usb_stereo_camera.cpp
)
target_link_libraries(zed_capture ${catkin_LIBRARIES} ${OpenCV_LIBS})
