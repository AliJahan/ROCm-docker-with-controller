cmake_minimum_required(VERSION 3.5...3.22)

project (Controller)

set(CONTROL_PORT "9090" CACHE STRING "ZQM port for Controller")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE Debug)
    set(DEBUG_MODE "true" CACHE STRING "Debug mode for porject")
else()
    set(CMAKE_BUILD_TYPE Release)
    set(DEBUG_MODE "false" CACHE STRING "Debug mode for porject")
endif()
# find rocm_smi lib
set(GPU_TARGETS "gfx906" CACHE STRING "GPU targets to compile for") # for MI50 (change based on your GPU arch)
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm/")
find_package(rocm_smi REQUIRED)

include_directories(include)
include_directories(${ROCM_SMI_INCLUDE_DIRS})

set(SRCS src/controller.cpp
         src/resource_controller.cpp)
set_source_files_properties(${SRCS} PROPERTIES COMPILE_DEFINITIONS "DEBUG_MODE=${DEBUG_MODE}")
add_executable(controller src/main.cpp)

target_sources(controller PUBLIC ${SRCS})
target_compile_definitions(controller PRIVATE CONTROL_PORT="${CONTROL_PORT}" DEBUG_MODE=${DEBUG_MODE})
set_target_properties(controller PROPERTIES CXX_STANDARD 17)
target_link_libraries(controller PUBLIC -lrt pthread zmq ${ROCM_SMI_LIB_DIR}/librocm_smi64.so)
#set_target_properties(controller PROPERTIES RUNTIME_OUTPUT_DIRECTORY "/usr/local/bin/")
# for local test uncomment lines #30 and uncomment below lines
# installs controller binary in build dir
set_target_properties(controller PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
# compiles master (communication with controller outside container) and installs it binary in build dir
add_executable(master src/master.cpp)
target_link_libraries(master PUBLIC -lrt pthread zmq)
set_target_properties(master PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
