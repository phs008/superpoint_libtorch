cmake_minimum_required(VERSION 3.19)
project(SuperPoint_cplus)

add_definitions(-DCXX_VERSION=${CXX_VERSION})

message('You have selection cxx ${CXX_VERSION}, so superpoint would running on libtorch 1.3.0')

if (${CXX_VERSION} MATCHES 11)
    set(CMAKE_CXX_STANDARD 11)
    set(Torch_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/libtorch_1_3_0/share/cmake/Torch) #libtorch_1_3_0
else ()
    set(CMAKE_CXX_STANDARD 14)
    set(Torch_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/libtorch_1_8_1/share/cmake/Torch) #libtorch_1_3_0
endif ()

message(${PROJECT_SOURCE_DIR})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")


include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()

IF (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF (CMAKE_COMPILER_IS_GNUCC)

#find_package(Eigen3 3.1.0 REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost 1.65.1 REQUIRED COMPONENTS system filesystem)
find_package(OpenCV 3.0 QUIET)

if (NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif ()
endif ()


include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        src/TimeClock.cc
        src/SuperPoint.cc
        src/SPextractor.cc)

target_link_libraries(${PROJECT_NAME}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS})

message(PROJECT_SOURCE_DIR)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(main
        main.cpp)

target_link_libraries(main
        ${PROJECT_NAME}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS})