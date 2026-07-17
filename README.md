# AAA Engine

A modern, high-performance, cross-platform 3D game engine built for speed, scalability, and modern hardware architectures.

## Architecture

This engine is built with a strict separation of concerns, focusing on multi-core utilization and hardware-aligned memory management to rival modern industry standards.

### Core Systems (Phase 1)

* **SIMD Math Library (C++26 Aligned):**
    * `SIMDCustomWrapper.h:` WideBatch (SOA) and FixedBatch4 (AOS) types will be used constantly instead of standard floats or math libraries.
    * Utilizes architecture-specific intrinsics (AVX-512, AVX2, SSE4.1, ARM NEON).
    * Strict separation between CPU-bound Structural of Arrays (SOA - `WideBatch`) for physics/culling and GPU-bound Array of Structs (AOS - `FixedBatch4`) for geometry/uniforms.
    * Features Cody-Waite range reduction for extreme distance precision and Minimax polynomial approximation for trigonometric functions.
    * Lock-free proxy objects for branchless conditional blending.
* **Memory Management:**
    * `Memory.h:` Review LocalLinearArena and ConcurrentPoolAllocator. Note: Standard new and delete are strictly forbidden in gameplay loops.
    * Bypasses the OS heap for high-frequency runtime loops to prevent fragmentation and latency.
    * **Virtual Memory Arena:** Dual-kernel (Windows/POSIX) integration for massive open-world streaming with zero physical RAM overhead until committed.
    * **Frame Allocator:** Zero-overhead double/triple buffering using contiguous byte blocks and placement new.
    * **Local & Concurrent Pool Allocators:** $O(1)$ lock-free bump-to-pool hybrids for game objects, network packets, and audio voices, featuring 128-bit tagged pointers to eliminate the ABA problem.
    * **Aligned Allocators:** Forces the OS to provide memory on strict 16, 32, or 64-byte boundaries, preventing General Protection Faults during SIMD operations.
* **Job System:**
    * `FiberJobSystem/JobSystem.h:` DispatchAsync is used to fan out work to the CPU cores using C++20 coroutines, and YieldFiber handles deep context switching.
    * A hybrid lock-free scheduler handling both stackless coroutines (for lightweight SIMD math chunks) and stackful fibers (for deep middleware context switching).
    * **Work-Stealing Queue:** A 64-bit lock-free Chase-Lev deque immune to the ABA problem.
    * **Fiber Context Switching:** Custom assembly implementations for Windows x64 (MASM), POSIX x64 (System V AMD64 ABI), and ARM64 (AAPCS64) for ultra-fast (~3ns) context swaps

---

## Building the Engine 

This project uses CMake and vcpkg to ensure a frictionless, cross-platform build process.

### 1. System Prerequisites
Ensure your local workstation has the required core build tools installed.
* **Version Control:** Git installed and added to your system `PATH`.
* **Build System:** CMake (v3.28 or newer).
* **Compilers:**
  * **Windows:** Visual Studio 2022 (with the "Desktop development with C++" workload installed).
  * **Linux:** GCC 13+ or Clang 17+.
  * **macOS:** Xcode Command Line Tools.

### 2. Compilation (One-Click Setup)

We provide master build scripts that automatically detect and download the vcpkg package manager, pull dependencies (GLFW, ImGui, GLAD), configure CMake, and compile the engine optimized for your hardware.

**Windows:**
Open your command prompt, clone the repository, and run the batch script:
```cmd
git clone [https://github.com/your-studio/aaa-engine.git](https://github.com/your-studio/aaa-engine.git)
cd aaa-engine
.\build.bat
