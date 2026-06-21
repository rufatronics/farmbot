cmake_minimum_required(VERSION 3.22.1)
project(farmbot_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── ABI-specific compiler flags ──────────────────────────────────────────────
# Fix for the known armeabi-v7a NEON intrinsic redefinition bug in llama.cpp
# (ggml-org/llama.cpp issue #4421, fixed via PR #5702).
# Without this, 32-bit builds fail with "redefinition of 'vcvtnq_s32_f32'".
if(ANDROID_ABI STREQUAL "armeabi-v7a")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon-vfpv4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon-vfpv4")
    add_definitions(-DGGML_USE_NEON=1)
endif()

if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.4a+dotprod")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.4a+dotprod")
endif()

# ── llama.cpp ─────────────────────────────────────────────────────────────────
# Cloned by the CI workflow into app/src/main/cpp/llama.cpp before this runs.
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

if(NOT EXISTS ${LLAMA_DIR}/CMakeLists.txt)
    message(FATAL_ERROR
        "llama.cpp not found at ${LLAMA_DIR}. "
        "The CI workflow must clone it there before the Gradle build runs."
    )
endif()

# Build llama.cpp as a static library, minimal config for mobile:
# no server/examples/tests, CPU-only (no GPU backend needed on these target devices).
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama_build)

# ── JNI bridge ────────────────────────────────────────────────────────────────
add_library(farmbot_jni SHARED
    farmbot_jni.cpp
)

target_include_directories(farmbot_jni PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_libraries(farmbot_jni
    llama
    ggml
    android
    log
)
