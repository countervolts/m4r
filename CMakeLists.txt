cmake_minimum_required(VERSION 3.20)
project(m4r CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT WIN32)
  find_package(Vulkan REQUIRED)
endif()
find_program(GLSLANG glslangValidator REQUIRED)
find_package(Python3 REQUIRED)

# vkhip's donor compute shader, embedded as SPIR-V
set(DONOR_HDR ${CMAKE_BINARY_DIR}/gen/donor_spv.h)
add_custom_command(OUTPUT ${DONOR_HDR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
  COMMAND ${GLSLANG} -V --target-env vulkan1.3 ${CMAKE_SOURCE_DIR}/runtime/vkhip/donor.comp -o ${CMAKE_BINARY_DIR}/gen/donor.spv
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/spv_header.py ${CMAKE_BINARY_DIR}/gen/donor.spv ${DONOR_HDR} donor_spv
  DEPENDS ${CMAKE_SOURCE_DIR}/runtime/vkhip/donor.comp ${CMAKE_SOURCE_DIR}/scripts/spv_header.py)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/sacneh.spv
  COMMAND ${GLSLANG} -V --target-env vulkan1.3 ${CMAKE_SOURCE_DIR}/shaders/sacneh.comp -o ${CMAKE_BINARY_DIR}/sacneh.spv
  DEPENDS ${CMAKE_SOURCE_DIR}/shaders/sacneh.comp)
add_custom_target(shaders ALL DEPENDS ${CMAKE_BINARY_DIR}/sacneh.spv)

set(M4R_RUNTIME_SOURCES runtime/vk.cpp runtime/vkfn.cpp runtime/kernel.cpp runtime/plan.cpp runtime/m4r_api.cpp
    runtime/vkhip/vkhip.cpp ${DONOR_HDR})
if(WIN32)
  # m4r.dll for OptiScaler under Proton: Vulkan entry points come from the game's device
  add_library(m4r SHARED ${M4R_RUNTIME_SOURCES})
  target_compile_definitions(m4r PRIVATE M4R_DYNAMIC_VK VK_NO_PROTOTYPES)
  target_include_directories(m4r PRIVATE runtime ${CMAKE_BINARY_DIR}/gen ${M4R_VULKAN_INCLUDE})
  target_link_options(m4r PRIVATE -static -static-libgcc -static-libstdc++)
  set_target_properties(m4r PROPERTIES PREFIX "")
  return()
endif()
add_library(m4r_runtime STATIC ${M4R_RUNTIME_SOURCES})
target_include_directories(m4r_runtime PUBLIC runtime ${CMAKE_BINARY_DIR}/gen)
target_link_libraries(m4r_runtime PUBLIC Vulkan::Vulkan)

add_executable(test_kernel tests/test_kernel.cpp)
target_link_libraries(test_kernel m4r_runtime)

add_executable(test_plan tests/test_plan.cpp)
target_link_libraries(test_plan m4r_runtime)

add_executable(run_bbr tests/run_bbr.cpp)
target_link_libraries(run_bbr m4r_runtime)

add_executable(run_fg tests/run_fg.cpp)
target_link_libraries(run_fg m4r_runtime)
