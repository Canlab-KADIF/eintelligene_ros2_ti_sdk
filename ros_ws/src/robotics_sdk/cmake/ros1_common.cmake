find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge image_transport sensor_msgs std_msgs message_generation common_msgs)
find_package(OpenCV REQUIRED)

catkin_package(CATKIN_DEPENDS roscpp cv_bridge image_transport sensor_msgs message_runtime)

include(${CMAKE_CURRENT_LIST_DIR}/common_platform.cmake)

include_directories(${catkin_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS})

# Function for building a node:
# ARG0: app name
# ARG1: source list
function(build_node)
    set(app ${ARGV0})
    set(src ${ARGV1})

    add_executable(${app} ${${src}})
    target_link_libraries(${app}
                          ${catkin_LIBRARIES}
                          -Wl,--start-group
                          ${COMMON_LINK_LIBS}
                          ${TARGET_LINK_LIBS}
                          -Wl,--end-group
                          )

    add_dependencies(${app}
                     ${common_msgs_EXPORTED_TARGETS})

endfunction(build_node)

