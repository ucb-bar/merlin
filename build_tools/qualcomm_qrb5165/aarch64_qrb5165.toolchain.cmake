# Cross-toolchain for the QRB5165 (aarch64-linux-gnu).
#
# Targets the board's glibc 2.31 (Ubuntu 20.04) by combining: - The Ubuntu-stock
# `gcc-10-aarch64-linux-gnu` cross compiler (GCC 10 ships a libgcc that does NOT
# reference glibc-2.32+ symbols like `_dl_find_object`, so its emitted binaries
# link cleanly against the board's older libc). - A board-rsync'd sysroot at
# /scratch2/agustin/qrb5165_board_sysroot for headers + crt0 + libc.so.6 +
# libstdc++.so.6 (Debian/Ubuntu multiarch layout,
# /usr/lib/aarch64-linux-gnu/...).
#
# This combination is what unblocks the QNN HAL: the merlin scheduler dlopens
# libQnn{Gpu,Htp}.so on board, those .sos pull in the board's
# /lib/aarch64-linux-gnu/libc.so.6, and our binary's dynamic libc must match
# (otherwise the loader trips _dl_call_libc_early_init on `__libc_early_init`
# lookup).
#
# Earlier this build used the Xilinx GCC 12.2.0 cross compiler with `-static`,
# but its libgcc references _dl_find_object@GLIBC_2.35 — fine for static-link
# CPU-only schedules, fatal once dlopen pulls in the board's older libc. See
# `project_qnn_glibc_skew.md` for the diagnosis.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc-10)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++-10)
set(CMAKE_AR /usr/bin/aarch64-linux-gnu-ar)
set(CMAKE_RANLIB /usr/bin/aarch64-linux-gnu-ranlib)
set(CMAKE_STRIP /usr/bin/aarch64-linux-gnu-strip)

# Board sysroot: provides libc.so.6, libstdc++.so.6, ld-linux, /usr/include,
# multiarch lib paths.
set(CMAKE_SYSROOT /scratch2/agustin/qrb5165_board_sysroot)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

# Conda's activate script seeds CFLAGS / CXXFLAGS / LDFLAGS with x86_64
# host-targeted flags (`-march=nocona`, `-mtune=haswell`...). aarch64 GCC
# rejects them. Wipe the env vars before CMake captures them via *_INIT seeding.
set(ENV{CFLAGS} "")
set(ENV{CXXFLAGS} "")
set(ENV{CPPFLAGS} "")
set(ENV{LDFLAGS} "")

# IREE's runtime headers + the merlin samples include root.
set(_IREE_RUNTIME_INC
    "/scratch2/agustin/merlin/third_party/iree_bar/runtime/src")
set(_MERLIN_SAMPLES_COMMON "/scratch2/agustin/merlin/samples/common")

# Ubuntu 24.04's `g++-10-aarch64-linux-gnu` package ships its own C++ headers at
# `/usr/aarch64-linux-gnu/include/c++/10/` — but those headers expect the host's
# glibc 2.39 cdefs.h macros (`__attr_access`...). The board has glibc 2.31. So
# we use `-nostdinc -nostdinc++` and explicitly layer: 1. GCC10's own internal
# headers (stddef.h, stdint.h, ...). 2. GCC10's C++ headers (libstdc++ headers
# from the cross package). 3. Board sysroot multiarch headers
# (/usr/include/aarch64-linux-gnu/). 4. Board sysroot bare /usr/include/. This
# way the C/C++ stdlib headers are GCC10's (matching the libgcc / libstdc++.a it
# links), but the system glibc headers are the board's.
set(_GCC10_INTERNAL_INC "/usr/lib/gcc-cross/aarch64-linux-gnu/10/include")
set(_GCC10_CXX_INC "/usr/aarch64-linux-gnu/include/c++/10")

set(_INC_FLAGS
    "-nostdinc -nostdinc++ \
-I${_GCC10_INTERNAL_INC} \
-I${_GCC10_CXX_INC} \
-I${_GCC10_CXX_INC}/aarch64-linux-gnu \
-I${_GCC10_CXX_INC}/backward \
-I${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu \
-I${CMAKE_SYSROOT}/usr/include \
-I${_IREE_RUNTIME_INC} -I${_MERLIN_SAMPLES_COMMON}")

# C uses only `-nostdinc`; `-nostdinc++` is C++-only and emits a warning under C
# compilation that becomes fatal under `-Werror` (which IREE's build sets by
# default). Filter it out of the C flags.
string(REPLACE "-nostdinc++ " "" _INC_FLAGS_C "${_INC_FLAGS}")
# GCC 10's aarch64 optimizer reports a false-positive array-bounds warning in
# upstream IREE runtime printf.c; IREE builds with -Werror, so demote it here
# for this cross toolchain rather than weakening source-level diagnostics.
set(_QRB5165_WARN_FLAGS
    "-Wno-error=array-bounds -Wno-error=missing-braces -Wno-error=attributes -Wno-error=inline"
)
set(CMAKE_C_FLAGS_INIT "${_INC_FLAGS_C} ${_QRB5165_WARN_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${_INC_FLAGS} ${_QRB5165_WARN_FLAGS}")

# CRT0 / Scrt1.o / crti.o / crtn.o sit in the board's multiarch lib dir.
set(_BOARD_MULTIARCH_LIB "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")
set(_BOARD_BASE_LIB "${CMAKE_SYSROOT}/lib/aarch64-linux-gnu")
add_compile_options("-B${_BOARD_MULTIARCH_LIB}")

# Dynamic link against the board's glibc 2.31 (via CMAKE_SYSROOT). Pin the
# loader to the board's path explicitly. -rpath ensures runtime library lookup
# hits the board's multiarch dirs even when LD_LIBRARY_PATH is set to QNN's
# runtime libs.
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-Wl,--dynamic-linker=/lib/ld-linux-aarch64.so.1 \
-B${_BOARD_MULTIARCH_LIB} \
-L${_BOARD_BASE_LIB} -L${_BOARD_MULTIARCH_LIB} \
-Wl,-rpath-link,${_BOARD_BASE_LIB} \
-Wl,-rpath-link,${_BOARD_MULTIARCH_LIB} \
-Wl,-rpath,/lib/aarch64-linux-gnu \
-Wl,-rpath,/usr/lib/aarch64-linux-gnu")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "")
