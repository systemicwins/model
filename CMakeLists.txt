cmake_minimum_required(VERSION 3.14)
project(MambaModel VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# HIP support is optional; enable via -DENABLE_HIP=ON when building on systems with ROCm
option(ENABLE_HIP "Enable HIP/ROCm GPU acceleration" OFF)

if(ENABLE_HIP)
    find_package(hip REQUIRED)
    # Try to find optional ROCm libs; don't fail CMake if missing
    find_package(hipblas QUIET)
    find_package(hipfft QUIET)
    find_package(rocrand QUIET)
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
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# Add source files
set(SOURCES
    src/tensor_ops.cpp
    src/tokenizer.cpp
    src/layer_norm.cpp
    src/mamba.cpp
    src/main.cpp
)

set(MATRYOSHKA_SOURCES
    src/tensor_ops.cpp
    src/tokenizer.cpp
    src/layer_norm.cpp
    src/mamba.cpp
    src/matryoshka_encoder.cpp
    src/matryoshka_trainer.cpp
    src/matryoshka_benchmark.cpp
)

set(HEADERS
    include/mamba.h
    include/layer_norm.h
    include/matryoshka_encoder.h
    include/tensor_ops.h
    include/tokenizer.h
)

# Create executables
add_executable(mamba_model ${SOURCES})
target_compile_definitions(mamba_model PRIVATE MAX_SEQUENCE_LENGTH=100000)
add_executable(matryoshka_benchmark ${MATRYOSHKA_SOURCES})
target_compile_definitions(matryoshka_benchmark PRIVATE MAX_SEQUENCE_LENGTH=100000)

# Include directories
target_include_directories(mamba_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CURL_INCLUDE_DIRS}
)

target_include_directories(matryoshka_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CURL_INCLUDE_DIRS}
)

# Link libraries
set(MAMBA_LINK_LIBS
    ${CURL_LIBRARIES}
    Threads::Threads
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
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

target_link_libraries(mamba_model PRIVATE ${MAMBA_LINK_LIBS})
target_link_libraries(matryoshka_benchmark PRIVATE ${MAMBA_LINK_LIBS})

# OpenMP configuration
# Prefer CMake's OpenMP package; on AppleClang fall back to libomp if necessary.
if(OpenMP_CXX_FOUND)
    target_link_libraries(mamba_model PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(matryoshka_benchmark PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "OpenMP not found via CMake; applying Homebrew libomp flags for AppleClang")
    # Try typical Homebrew prefix for Apple Silicon
    set(HOMEBREW_PREFIX "/opt/homebrew")
    if(EXISTS "${HOMEBREW_PREFIX}/lib/libomp.dylib")
        target_include_directories(matryoshka_benchmark PRIVATE "${HOMEBREW_PREFIX}/include")
        target_link_directories(matryoshka_benchmark PRIVATE "${HOMEBREW_PREFIX}/lib")
        target_compile_options(matryoshka_benchmark PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(matryoshka_benchmark PRIVATE omp)
    else()
        # Intel Homebrew fallback
        set(HOMEBREW_PREFIX "/usr/local")
        if(EXISTS "${HOMEBREW_PREFIX}/lib/libomp.dylib")
            target_include_directories(matryoshka_benchmark PRIVATE "${HOMEBREW_PREFIX}/include")
            target_link_directories(matryoshka_benchmark PRIVATE "${HOMEBREW_PREFIX}/lib")
            target_compile_options(matryoshka_benchmark PRIVATE -Xpreprocessor -fopenmp)
            target_link_libraries(matryoshka_benchmark PRIVATE omp)
        else()
            message(WARNING "libomp not found in Homebrew prefixes; matryoshka_benchmark will build without OpenMP")
        endif()
    endif()
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
    target_compile_options(matryoshka_benchmark PRIVATE
        -O3
        -march=native
        -ffast-math
    )
endif()

if(ENABLE_HIP)
    target_compile_definitions(mamba_model PRIVATE HIP_ENABLED)
    target_compile_definitions(matryoshka_benchmark PRIVATE HIP_ENABLED)
endif()

# Enable warnings
target_compile_options(mamba_model PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

target_compile_options(matryoshka_benchmark PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)