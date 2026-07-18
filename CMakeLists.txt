# Require a modern version of CMake to support C++26 Module features natively
cmake_minimum_required(VERSION 3.28)

# Enable C++ and ASM (Assembly). ASM_MASM is specific to MSVC on Windows.
project(AAAEngine LANGUAGES CXX C)

# ==============================================================================
# GLOBAL COMPILER ENGINE PARADIGMS
# ==============================================================================
# We set this to 23 so CMake knows the native mappings, but we use c++latest 
# in the compiler options to unlock the C++26 features.

# --- 1. GLOBALLY ENFORCE C++23 --- 
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# ARCHITECTURE & KERNEL DETECTION
# ==============================================================================
# Dynamically locate and configure the low-level lock-free assembly worker components

# --- 2. DETECT HARDWARE FOR THE FIBER ASM ---
if(WIN32)
    # Wakes up Microsoft Macro Assembler for Windows systems
    enable_language(ASM_MASM)
    set(FIBER_ASM_SRC "FiberJobSystem/FiberContext_Win64.asm")
    
elseif(APPLE)
    # Wakes up GNU Assembler for Apple Silicon (AArch64)
    enable_language(ASM)
    set(FIBER_ASM_SRC "FiberJobSystem/FiberContext_ARM64.S")
    
elseif(UNIX)
    # Wakes up GNU Assembler for Linux environments (POSIX x64)
    enable_language(ASM)
    set(FIBER_ASM_SRC "FiberJobSystem/FiberContext_POSIX64.S")
endif()

# ==============================================================================
# TARGET DEFINITION
# ==============================================================================

# Find all engine source compilation units within the master project scope
file(GLOB ENGINE_SOURCES "*.cpp")

# Define the unified master executable container
add_executable(AAAEngine 
    ${ENGINE_SOURCES} 
    ${FIBER_ASM_SRC}
)

# ==============================================================================
# THE CONTRACT: C++26 MODULE ARCHITECTURE PIPELINE
# ==============================================================================
# CMake completely analyzes the code graph, precompiles files (.ifc/.pcm) 
# and links dependencies in absolute topological order automatically.

# these are the newly created modules
target_sources(AAAEngine PUBLIC
  FILE_SET cxx_modules TYPE CXX_MODULES FILES
    "SIMD/SIMDCustomWrapper.cppm" 
    "SIMD/SIMDVectorMath.cppm"
    "SIMD/Hardware.cppm"
    "Physics/ParticleSystem.cppm"
    "ECS/Systems.cppm"
    "ECS/EntityComponentSystem.cppm"
)

# ==============================================================================
# THIRD-PARTY HEADER SEARCH RECONNAISSANCE
# ==============================================================================
target_include_directories(AAAEngine PRIVATE
    "thirdparty/imgui-master"
    "thirdparty/imgui-master/backends"
    "thirdparty/glad/include"
)

# ==============================================================================
# COMPILER OPTIMIZATION MATRIX (Strict Performance & SIMD Alignment)
# ==============================================================================
# The $<COMPILE_LANGUAGE:CXX> generator expression isolates these flags 
# strictly to cl.exe, protecting ml64.exe from contamination.
# /W4 /WX               # Warning level 4, treat warnings as errors
# /arch:AVX2            # Unlock 256-bit SIMD processing arrays
# /fp:fast              # Enable Floating-Point Math Reordering (FMA code-generation)
# /Zc:preprocessor      # Modern standards-compliant preprocessor tree
# /Qpar                 # Enable Auto-Parallelization engine optimizations
# $<$<CONFIG:Release>:/O2 /MD>  # Maximum execution velocity optimizations
# $<$<CONFIG:Debug>:/Zi /MDd>   # Complete debugging descriptors
# -Wall -Wextra -Werror # Enforce explicit clean programming contracts
# -ffast-math           # Strip IEEE floating-point compliance limits for velocity


if(MSVC)
    target_compile_options(AAAEngine PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /arch:AVX2 /fp:fast /Zc:preprocessor /Qpar /std:c++latest>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /MD>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Zi /MDd>
    )
    
    target_compile_definitions(AAAEngine PRIVATE _CRT_SECURE_NO_WARNINGS)
    
else()
    target_compile_options(AAAEngine PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -ffast-math>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g -O0>
    )

    # Apply x64 specific machine instructions only if target architecture matches hardware
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
        target_compile_options(AAAEngine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mcx16>)
    endif()
endif()

# ==============================================================================
# TELEMETRY & HARDWARE RECONCILIATION LINKING (vcpkg / Local Link)
# ==============================================================================
# 1. vcpkg.json automatically detects the manifest
# 2. downloads the source code for these libraries
# 3. compiles them specifically for the user's OS and architecture
# 4. links them to this engine


# Search manifest layers directly for engine libraries (find the packages installed automatically by vcpkg)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

# Link the libraries directly to our engine target
target_link_libraries(AAAEngine PRIVATE
    glfw
    imgui::imgui
    glad::glad
)

# Cross-platform threading and math libraries
# Resolve system dependency hooks for non-Windows architectures
if(MSVC)
    target_link_libraries(AAAEngine PRIVATE gdi32 shell32 user32 winmm imm32 psapi)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(AAAEngine PRIVATE pthread dl m rt)
endif()
