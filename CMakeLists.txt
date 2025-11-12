cmake_minimum_required(VERSION 3.14)

# HIP support is optional; enable via -DENABLE_HIP=ON when building on systems with ROCm
option(ENABLE_HIP "Enable HIP/ROCm GPU acceleration" OFF)

# Set HIP compiler if HIP is enabled (must be set before project() declaration)
if(ENABLE_HIP)
    # Allow user to specify ROCm path, fallback to common locations
    if(NOT DEFINED ROCM_PATH)
        if(EXISTS "/opt/rocm-7.0.0")
            set(ROCM_PATH "/opt/rocm-7.0.0")
        elseif(EXISTS "/opt/rocm")
            set(ROCM_PATH "/opt/rocm")
        else()
            # Let find_package handle it
            set(ROCM_PATH "")
        endif()
    endif()
    
    set(CMAKE_HIP_COMPILER "${ROCM_PATH}/bin/hipcc" CACHE FILEPATH "HIP compiler")
    set(CMAKE_CXX_COMPILER "${CMAKE_HIP_COMPILER}" CACHE FILEPATH "C++ compiler")
endif()

project(MambaModel VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable -Werror to treat warnings as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

if(ENABLE_HIP)
    # Manual HIP configuration for ROCm
    set(HIP_PATH "${ROCM_PATH}" CACHE PATH "HIP installation path")
    set(CMAKE_HIP_COMPILER "${HIP_PATH}/bin/hipcc")

    # Check if HIP compiler exists
    if(EXISTS "${CMAKE_HIP_COMPILER}")
        message(STATUS "Found HIP compiler: ${CMAKE_HIP_COMPILER}")
        set(HIP_FOUND TRUE)

        # Set HIP as the C++ compiler when HIP is enabled
        set(CMAKE_CXX_COMPILER "${CMAKE_HIP_COMPILER}")

        # Add HIP include directories
        include_directories("${HIP_PATH}/include")
        include_directories("${HIP_PATH}/include/hip")

        # Try to find optional ROCm libs; don't fail CMake if missing
        find_package(hipblas QUIET)
        find_package(hipfft QUIET)
        find_package(rocrand QUIET)
    else()
        message(WARNING "HIP compiler not found at ${CMAKE_HIP_COMPILER}. Building without HIP support.")
        set(HIP_FOUND FALSE)
        set(ENABLE_HIP OFF)
    endif()
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenMP)

# JSON library (using nlohmann json)
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# Eigen for linear algebra operations
# Allow user to override via environment variable or cache variable
if(DEFINED ENV{EIGEN3_INCLUDE_DIR})
    set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" CACHE PATH "Eigen3 include directory")
elseif(DEFINED EIGEN3_INCLUDE_DIR)
    # Keep existing value
else()
    # Default location for local development
    set(EIGEN3_INCLUDE_DIR "/tmp/eigen-3.4.0" CACHE PATH "Eigen3 include directory")
endif()

# Check if Eigen3 exists at specified location
if(EXISTS "${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Found Eigen3 at: ${EIGEN3_INCLUDE_DIR}")
    set(Eigen3_FOUND TRUE)
else()
    # Fallback to system-installed Eigen3 locations
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/usr;/usr/local;/opt/homebrew")
    
    # Try to find Eigen3 using find_package
    find_package(Eigen3 3.4 QUIET)
    
    if(EIGEN3_FOUND)
        message(STATUS "Found Eigen3 using find_package: ${EIGEN3_INCLUDE_DIR}")
    else()
        # Try common installation paths
        set(COMMON_EIGEN_PATHS
            /usr/include/eigen3
            /usr/local/include/eigen3
            /opt/homebrew/include/eigen3
        )
        
        foreach(PATH ${COMMON_EIGEN_PATHS})
            if(EXISTS "${PATH}/Eigen")
                set(EIGEN3_INCLUDE_DIR "${PATH}" CACHE PATH "Eigen3 include directory")
                message(STATUS "Found Eigen3 at: ${EIGEN3_INCLUDE_DIR}")
                set(Eigen3_FOUND TRUE)
                break()
            endif()
        endforeach()
        
        if(NOT EIGEN3_FOUND)
            message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 development package or set EIGEN3_INCLUDE_DIR environment variable.")
        endif()
    endif()
endif()

# Add source files
set(SOURCES
    src/tensor_ops.cpp
    src/tokenizer.cpp
    src/layer_norm.cpp
    src/mamba.cpp
    src/transformer.cpp  # Uncommented - all dependencies appear to be available
    src/attention.cpp  # Uncommented - required by transformer.cpp
    src/feedforward.cpp  # Uncommented - required by transformer.cpp
    src/sparse_attention.cpp  # SSM-guided sparse attention implementation
    # src/hybrid_attention.cpp  # Commented out due to compute_reward access issues
    # src/act_controller.cpp  # Commented out due to Eigen issues
    # src/attention_benchmark.cpp  # Commented out due to Eigen issues
    # src/hybrid_attention_benchmark.cpp  # Commented out due to dependency on hybrid_attention
    # src/layer_comparison_benchmark.cpp  # Commented out due to dependencies
    # src/financial_prediction_demo.cpp  # Commented out due to type conflicts
    src/main.cpp
)

set(MATRYOSHKA_SOURCES
    # All matryoshka sources commented out due to dependencies
    # src/tensor_ops.cpp
    # src/tokenizer.cpp
    # src/layer_norm.cpp
    # src/mamba.cpp
    # src/transformer.cpp  # Commented out due to missing definitions
    # src/matryoshka_encoder.cpp  # Commented out due to save_matrix issues
    # src/matryoshka_trainer.cpp  # Commented out due to dependencies
    # src/matryoshka_benchmark.cpp  # Commented out due to dependencies
)

set(HEADERS
    include/mamba.h
    include/layer_norm.h
    include/matryoshka_encoder.h
    include/tensor_ops.h
    include/tokenizer.h
    include/hybrid_attention.h
    include/act_controller.h
    include/sparse_attention.h  # SSM-guided sparse attention header
)

# Create executables
add_executable(mamba_model ${SOURCES})

# Test executables with required source files
set(TEST_SOURCES
    test_sparse_attention.cpp
    src/mamba.cpp
    src/sparse_attention.cpp
    src/tensor_ops.cpp
    src/layer_norm.cpp
)

add_executable(test_sparse_attention ${TEST_SOURCES})

target_compile_definitions(mamba_model PRIVATE MAX_SEQUENCE_LENGTH=100000)
target_compile_definitions(test_sparse_attention PRIVATE MAX_SEQUENCE_LENGTH=100000)

target_include_directories(test_sparse_attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Include directories
target_include_directories(mamba_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CURL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

if(ENABLE_HIP)
    target_include_directories(mamba_model PRIVATE
        ${HIP_PATH}/include
        ${HIP_PATH}/include/hipblas
        ${HIP_PATH}/include/hipfft
        ${HIP_PATH}/include/rocrand
    )
endif()


# Link libraries
set(MAMBA_LINK_LIBS
    ${CURL_LIBRARIES}
    Threads::Threads
    nlohmann_json::nlohmann_json
)

if(ENABLE_HIP)
    list(APPEND MAMBA_LINK_LIBS
        hip::device
        hip::host
    )
    if(TARGET hipblas)
        list(APPEND MAMBA_LINK_LIBS hipblas)
    endif()
    if(TARGET hipfft)
        list(APPEND MAMBA_LINK_LIBS hipfft)
    endif()
    if(TARGET rocrand)
        list(APPEND MAMBA_LINK_LIBS rocrand)
    endif()
endif()

# Link libraries for main executable
target_link_libraries(mamba_model PRIVATE ${MAMBA_LINK_LIBS})

target_link_libraries(test_sparse_attention PRIVATE ${MAMBA_LINK_LIBS})
target_include_directories(test_sparse_attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# OpenMP configuration
# Prefer CMake's OpenMP package; on AppleClang fall back to libomp if necessary.
if(OpenMP_CXX_FOUND)
    target_link_libraries(mamba_model PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_HIP)
    # HIP-specific compiler flags
    set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -O3 -march=native -ffast-math")
endif()

# Compiler flags for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(mamba_model PRIVATE
        -O3
        -march=native
        -ffast-math
    )
endif()

if(ENABLE_HIP)
    target_compile_definitions(mamba_model PRIVATE HIP_ENABLED)
endif()

# Enable warnings (suppress known false positives from Eigen3)
target_compile_options(mamba_model PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-array-bounds        # Suppress false positive array bounds warnings in Eigen3 AVX code
    -Wno-maybe-uninitialized # Suppress false positive uninitialized variable warnings in Eigen3 templates
)
