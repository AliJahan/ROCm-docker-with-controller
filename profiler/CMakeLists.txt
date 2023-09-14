cmake_minimum_required(VERSION 3.5...3.22)

project (Profiler)


add_executable(master master.cpp)

set_target_properties(master PROPERTIES CXX_STANDARD 17)
target_link_libraries(master PUBLIC -lrt pthread zmq)
set_target_properties(master PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
