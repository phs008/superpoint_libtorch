cmake_minimum_required(VERSION 3.19)
project(SuperPoint_cplus14_ROS)

set(CMAKE_CXX_STANDARD 11)

include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
rosbuild_init()

message(${PROJECT_SOURCE_DIR})






rosbuild_add_executable(SuperPoint_cplus14_ROS
        ros_main.cpp
        )

target_link_libraries(SuperPoint_cplus14_ROS
        ${SuperPoint_cplus}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES})