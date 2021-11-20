mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release
cmake --install . --config Release
cmake --build . --config Debug
cmake --install . --config Debug
