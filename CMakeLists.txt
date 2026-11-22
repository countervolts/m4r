cmake_minimum_required(VERSION 3.20)
project(m4r CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT WIN32)
  find_package(Vulkan REQUIRED)
endif()
# native Windows (MSVC / clang-cl): Vulkan's headers only (VULKAN_SDK); everything runs on AMD's HIP runtime (runtime/hipvk)
set(M4R_NATIVE_WINDOWS OFF)
if(WIN32 AND NOT CMAKE_CROSSCOMPILING)
  set(M4R_NATIVE_WINDOWS ON)
  if(NOT M4R_VULKAN_INCLUDE)
    set(M4R_VULKAN_INCLUDE $ENV{VULKAN_SDK}/Include)
  endif()
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
if(M4R_NATIVE_WINDOWS)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")   # static CRT: the DLL loads into games
  set(HIPVK_SOURCES runtime/hipvk/hipvk.cpp runtime/hipvk/hip_dyn.cpp)
  add_library(m4r_runtime STATIC ${M4R_RUNTIME_SOURCES} ${HIPVK_SOURCES})
  target_compile_definitions(m4r_runtime PUBLIC M4R_DYNAMIC_VK M4R_HIPVK VK_NO_PROTOTYPES NOMINMAX _CRT_SECURE_NO_WARNINGS)
  target_include_directories(m4r_runtime PUBLIC runtime ${CMAKE_BINARY_DIR}/gen ${M4R_VULKAN_INCLUDE})
  if(MSVC)
    target_compile_options(m4r_runtime PUBLIC /utf-8 /EHsc)
  endif()
  # m4r.dll: the C API (m4r.h) for the OptiScaler plugin (optiscaler/m4r_asi)
  add_library(m4r SHARED runtime/m4r_dll.cpp)
  target_link_libraries(m4r PRIVATE m4r_runtime)
  if(MSVC)
    target_link_options(m4r PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:m4r_runtime>)
  endif()
  set_target_properties(m4r PROPERTIES PREFIX "")
  foreach(t run_bbr run_fg run_denoise run_denoise_api test_plan)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} m4r_runtime)
  endforeach()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")   # _Float16 conversions (run_denoise_api) live in compiler-rt
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} /clang:-print-libgcc-file-name /clang:--rtlib=compiler-rt OUTPUT_VARIABLE RTLIB OUTPUT_STRIP_TRAILING_WHITESPACE)
    target_link_libraries(run_denoise_api ${RTLIB})
  endif()
  return()
endif()
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

add_executable(test_scratch tests/test_scratch.cpp)
target_link_libraries(test_scratch m4r_runtime)

add_executable(vkhip_parse tests/vkhip_parse.cpp)
target_link_libraries(vkhip_parse m4r_runtime)
add_executable(run_bbr tests/run_bbr.cpp)
target_link_libraries(run_bbr m4r_runtime)
add_executable(run_denoise tests/run_denoise.cpp)
target_link_libraries(run_denoise m4r_runtime)
add_executable(run_denoise_api tests/run_denoise_api.cpp)
target_link_libraries(run_denoise_api m4r_runtime)

add_executable(run_fg tests/run_fg.cpp)
target_link_libraries(run_fg m4r_runtime)

add_executable(ref_framegen tests/ref_framegen.cpp)
target_link_libraries(ref_framegen m4r_runtime ${CMAKE_DL_LIBS})
