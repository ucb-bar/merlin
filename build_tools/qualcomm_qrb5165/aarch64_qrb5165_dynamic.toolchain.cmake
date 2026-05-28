# Dynamic-link cross-toolchain for QRB5165 (Vulkan-capable build).
#
# The static toolchain at aarch64_qrb5165.toolchain.cmake produces binaries
# whose dlopen() of board-side libvulkan_adreno.so aborts on glibc skew
# (toolchain 2.33 vs board 2.31, dl-call-libc-early-init.c assertion).
#
# This toolchain links dynamically against a sysroot rsynced from the board
# itself (/scratch2/agustin/qrb5165_sysroot, populated from qdev's
# /usr/lib/aarch64-linux-gnu + /usr/include). Result: binaries link against
# board-glibc 2.31 directly, dlopen of libvulkan_adreno.so works.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(_TC /ecad/tools/xilinx/Vitis/2023.1/gnu/aarch64/lin/aarch64-linux)
set(CMAKE_C_COMPILER ${_TC}/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${_TC}/bin/aarch64-linux-gnu-g++)
set(CMAKE_AR ${_TC}/bin/aarch64-linux-gnu-ar)
set(CMAKE_RANLIB ${_TC}/bin/aarch64-linux-gnu-ranlib)
set(CMAKE_STRIP ${_TC}/bin/aarch64-linux-gnu-strip)

set(_BOARD_SYSROOT /scratch2/agustin/qrb5165_sysroot)
set(CMAKE_SYSROOT ${_BOARD_SYSROOT})

set(CMAKE_FIND_ROOT_PATH ${_BOARD_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

set(ENV{CFLAGS} "")
set(ENV{CXXFLAGS} "")
set(ENV{CPPFLAGS} "")
set(ENV{LDFLAGS} "")

set(_IREE_RUNTIME_INC
    "/scratch2/agustin/merlin/third_party/iree_bar/runtime/src")
set(_MERLIN_SAMPLES_COMMON "/scratch2/agustin/merlin/samples/common")
set(_INCS
    "-isystem ${_BOARD_SYSROOT}/usr/include/c++/9"
    "-isystem ${_BOARD_SYSROOT}/usr/include/c++/9/aarch64-linux-gnu"
    "-isystem ${_BOARD_SYSROOT}/usr/include/aarch64-linux-gnu"
    "-isystem ${_BOARD_SYSROOT}/usr/include"
    "-I${_IREE_RUNTIME_INC}"
    "-I${_MERLIN_SAMPLES_COMMON}")
string(JOIN " " _INC_STR ${_INCS})
# -Wno-error: gcc-12 toolchain compiling against gcc-9 stdlib emits spurious
# -Wmaybe-uninitialized in <regex>. We don't gain anything by treating that as
# fatal in a vendor cross-build.
set(_WARN_FLAGS "-Wno-error -w")
set(CMAKE_C_FLAGS_INIT "${_INC_STR} ${_WARN_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${_INC_STR} ${_WARN_FLAGS}")

set(_LDFLAGS
    "-B${_BOARD_SYSROOT}/usr/lib/aarch64-linux-gnu"
    "-B${_BOARD_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9"
    "-L${_BOARD_SYSROOT}/usr/lib/aarch64-linux-gnu"
    "-L${_BOARD_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9"
    "-L${_BOARD_SYSROOT}/lib/aarch64-linux-gnu"
    "-Wl,--dynamic-linker=/lib/ld-linux-aarch64.so.1"
    "-Wl,-rpath-link=${_BOARD_SYSROOT}/usr/lib/aarch64-linux-gnu:${_BOARD_SYSROOT}/lib/aarch64-linux-gnu"
)
string(JOIN " " _LD_STR ${_LDFLAGS})
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_LD_STR}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_LD_STR}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_LD_STR}")
