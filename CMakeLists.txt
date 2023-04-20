include(cmake/HunterGate.cmake)

# use config in cmake/Hunter/config.cmake
HunterGate(
  URL "https://github.com/ruslo/hunter/archive/v0.23.112.tar.gz"
  SHA1 "4b894e1d5d203f0cc9a77431dbb1b486ab6f4430"
  LOCAL
)

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)



project(cdlp_project LANGUAGES CXX CUDA)

hunter_add_package(fmt)
hunter_add_package(spdlog)
find_package(spdlog CONFIG REQUIRED)
hunter_add_package(cub)
find_package(cub CONFIG REQUIRED)
hunter_add_package(Catch)
find_package(pangolin CONFIG REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(DEFAULT_BUILD_TYPE "Release")




set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(CTest)

add_executable(cdlp_project
  src/eval.cu
  src/cdlp.cu
  src/lcc.cu

  include/helper.hpp
  include/cdlp.hu
  include/lcc.hu

  common/catch.hpp
  common/clara.hpp
  common/utils.hpp
)


if(DEFINED GRAPH_PREFIX_PATH)
  message(STATUS "compiling with GRAPH_PREFIX_PATH=${GRAPH_PREFIX_PATH}")
  target_compile_definitions(cdlp_project PRIVATE GRAPH_PREFIX_PATH=${GRAPH_PREFIX_PATH})
else()
  message(FATAL_ERROR "please define GRAPH_PREFIX_PATH")
endif()

target_compile_features(cdlp_project PUBLIC cxx_std_11)

target_link_libraries(cdlp_project pangolin::pangolin32)


include_directories(cdlp_project
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/common
)

enable_testing()
