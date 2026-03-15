# Building the Tracy Profiler (v0.11.1) on Linux

## 1. Prerequisites

You must have a C++ compiler, CMake, and several development libraries installed. On a Debian-based system (like Ubuntu or Debian), you can install them with:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libglfw3-dev libdbus-1-dev
```

## 2. Clone repo or build from IREE third_party

```bash
git clone https://github.com/wolfpld/tracy.git
cd tracy
# git checkout 0.11.1  # (Optional, if you're not on this version)
```

## 3. Manual Fixes

Fix Always macro conflict: Add the following lines before #include "profiler/TracyConfig.hpp" (around line 36):

```C++
#ifdef Always
#undef Always
#endif
```

Fix API mismatch: Comment out the two lines in the wl_surface_listener struct definition (around lines 720-725):

```C++
constexpr struct wl_surface_listener surfaceListener = {
    .enter = SurfaceEnter,
    .leave = SurfaceLeave,
    // .preferred_buffer_scale = SurfacePreferredBufferScale,
    // .preferred_buffer_transform = SurfacePreferredBufferTransform
};
```

Also fix the `third_party/tracy/profiler/CMakeLists.txt`

```CMake
target_link_libraries(${PROJECT_NAME} PRIVATE TracyNfd dl)
```

We add the `dl` at the end.

## 4. Configuration (X11/Legacy Build)

```bash
cmake -B profiler/build -S profiler -DCMAKE_BUILD_TYPE=Release -DLEGACY=ON
cmake --build profiler/build --parallel --config Release
```

## 5. Run it

```bash
./profiler/build/tracy-profiler
```
