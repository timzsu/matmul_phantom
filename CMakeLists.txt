cmake_minimum_required(VERSION 3.20)

project(NEXUS-CUDA LANGUAGES CXX CUDA)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Debugging config
option(SHOW_WARNINGS "Enable warnings" OFF)
if (SHOW_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Wpedantic,-Wno-comment")
endif()

# Language config
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA config
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native) # auto-detect GPU
endif()
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --default-stream per-thread")

# Print CUDA info
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")

# PhantomFHE config
option(PHANTOM_USE_CUDA_PTX "Use CUDA PTX Assembly" ON)
message(STATUS "Use CUDA PTX Assembly: ${PHANTOM_USE_CUDA_PTX}")
if (PHANTOM_USE_CUDA_PTX)
    add_compile_definitions(PHANTOM_USE_CUDA_PTX)
endif()

# PhantomFHE
include_directories(thirdparty/phantom-fhe/include)

# NexusCUDA
file(GLOB_RECURSE NEXUS_CUDA_SOURCE_FILES CONFIGURE_DEPENDS
  "src/matmul_test.cu"
  "src/matrix_mul.cu"
  "src/ckks_evaluator.cu"
  "src/row_pack.cpp"
  "thirdparty/phantom-fhe/src/*.cu"
)

find_package(Eigen3 REQUIRED)

find_package(Catch2 REQUIRED)
add_executable(matmul_test 
  ${NEXUS_CUDA_SOURCE_FILES}
)
target_include_directories(matmul_test PUBLIC "thirdparty/phantom-fhe/include")
target_link_libraries(matmul_test
  PRIVATE
  Eigen3::Eigen
  Catch2::Catch2WithMain
)
enable_testing()
add_test(NAME matmul_test COMMAND matmul_test)