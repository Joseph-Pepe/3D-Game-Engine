# Require a modern version of CMake to support C++23/26 features
cmake_minimum_required(VERSION 3.28)

# Enable C++ and ASM (Assembly). ASM_MASM is specific to MSVC on Windows.
project(AAAEngine LANGUAGES CXX C)

# ==============================================================================
# ARCHITECTURE & KERNEL DETECTION
# ==============================================================================
# We must dynamically detect the OS and CPU architecture to compile the correct 
# FiberContext assembly file.

if(WIN32)
    # Enable Microsoft Macro Assembler
    enable_language(ASM_MASM)
    set(FIBER_ASM_SRC "FiberContext_Win64.asm")
    
elseif(APPLE)
    # Enable GNU Assembler for Apple Silicon (ARM64)
    enable_language(ASM)
    set(FIBER_ASM_SRC "FiberContext_ARM64.S")
    
elseif(UNIX)
    # Enable GNU Assembler for Linux (POSIX x64)
    enable_language(ASM)
    set(FIBER_ASM_SRC "FiberContext_POSIX64.S")
endif()

# ==============================================================================
# ENGINE EXECUTABLE TARGET
# ==============================================================================

# Find all C++ source files in the current directory
file(GLOB ENGINE_SOURCES "*.cpp")

# Define the main executable and attach our C++ files and the specific ASM file
add_executable(AAAEngine ${ENGINE_SOURCES} ${FIBER_ASM_SRC})

# Enforce C++23 (or C++26 if supported by the local compiler)
set_target_properties(AAAEngine PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# ==============================================================================
# COMPILER FLAGS (Strict Performance & AVX2 Alignment)
# ==============================================================================

if(MSVC)
    # Windows (MSVC) High-Performance Flags
    target_compile_options(AAAEngine PRIVATE
        /W4 /WX               # Warning level 4, treat warnings as errors
        /arch:AVX2            # Unlock 256-bit SIMD registers
        /fp:fast              # Enable Floating-Point Math Reordering (FMA)
        /Zc:preprocessor      # Modern standards-compliant preprocessor
        /Qpar                 # Enable Auto-Parallelization
        $<$<CONFIG:Release>:/O2 /MD>  # Max optimization in Release
        $<$<CONFIG:Debug>:/Zi /MDd>   # Full debug info in Debug
    )
    
    # Silence MSVC's annoying CRT security warnings
    target_compile_definitions(AAAEngine PRIVATE _CRT_SECURE_NO_WARNINGS)

else()
    # POSIX (GCC/Clang) High-Performance Flags
    target_compile_options(AAAEngine PRIVATE
        -Wall -Wextra -Werror # Strict warnings
        -ffast-math           # Enable Floating-Point Math Reordering
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-g -O0>
    )

    # Only enable AVX2 if we are compiling for an x86/x64 architecture
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
        target_compile_options(AAAEngine PRIVATE -mavx2 -mcx16)
    endif()
endif()

# ==============================================================================
# DEPENDENCY LINKING (vcpkg Integration)
# ==============================================================================
# 1. vcpkg.json automatically detects the manifest
# 2. downloads the source code for these libraries
# 3. compiles them specifically for the user's OS and architecture
# 4. links them to this engine

# Find the packages installed automatically by vcpkg
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
if(UNIX AND NOT APPLE)
    target_link_libraries(AAAEngine PRIVATE pthread dl m rt)
endif()
