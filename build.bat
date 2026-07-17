:: This is a single setup script that handles the entire pipeline automatically.
:: Anyone can clone this repository, type in one command, and walk away.

@echo off
setlocal

:: 1. Check if vcpkg is already installed and bootstrapped
if not exist "vcpkg\vcpkg.exe" (
    echo [AAA Engine] vcpkg not found. Downloading...
    
    :: Clone it if the folder doesn't exist at all
    if not exist "vcpkg" (
        git clone https://github.com/microsoft/vcpkg.git
    )
    
    echo [AAA Engine] Bootstrapping vcpkg...
    call .\vcpkg\bootstrap-vcpkg.bat
)

:: 2. Configure CMake (This triggers the vcpkg.json dependency downloads)
echo [AAA Engine] Generating CMake Project...
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="%~dp0vcpkg\scripts\buildsystems\vcpkg.cmake"

:: 3. Compile the Engine
echo [AAA Engine] Compiling High-Performance Release...
cmake --build build --config Release

echo [AAA Engine] Build Complete! Executable is located in \build\Release\
pause
