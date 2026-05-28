# riscv_firesim.toolchain.cmake Combined Bare-Metal Toolchain: Clang (Compile) +
# GCC (Link via Specs)

if(RISCV_TOOLCHAIN_INCLUDED)
  return()
endif(RISCV_TOOLCHAIN_INCLUDED)
set(RISCV_TOOLCHAIN_INCLUDED true)

# --- 1. Target System Configuration ---
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# --- 2. Toolchain Paths ---
if(DEFINED ENV{RISCV_TOOLCHAIN_ROOT})
  set(RISCV_TOOLCHAIN_ROOT
      "$ENV{RISCV_TOOLCHAIN_ROOT}"
      CACHE PATH "RISC-V compiler path" FORCE)
elseif(DEFINED ENV{RISCV})
  set(RISCV_TOOLCHAIN_ROOT
      "$ENV{RISCV}"
      CACHE PATH "RISC-V compiler path" FORCE)
else()
  message(
    FATAL_ERROR
      "RISCV_TOOLCHAIN_ROOT (or RISCV) environment variable must be set.")
endif()

# Sysroot setup for Clang Cascade: RISCV_NEWLIB_SYSROOT env →
# CHIPYARD_ROOT-derived → FATAL_ERROR
if(DEFINED ENV{RISCV_NEWLIB_SYSROOT})
  set(RISCV_NEWLIB_SYSROOT
      "$ENV{RISCV_NEWLIB_SYSROOT}"
      CACHE PATH "Newlib sysroot for bare-metal RISC-V" FORCE)
elseif(DEFINED ENV{CHIPYARD_ROOT})
  set(RISCV_NEWLIB_SYSROOT
      "$ENV{CHIPYARD_ROOT}/.conda-env/riscv-tools/riscv64-unknown-elf"
      CACHE PATH "Newlib sysroot for bare-metal RISC-V" FORCE)
else()
  message(
    FATAL_ERROR
      "CHIPYARD_ROOT (or RISCV_NEWLIB_SYSROOT) environment variable must be set.\n"
      "  export CHIPYARD_ROOT=/path/to/chipyard\n"
      "Or override the sysroot directly:\n"
      "  export RISCV_NEWLIB_SYSROOT=/path/to/riscv64-unknown-elf")
endif()
get_filename_component(RISCV_GCC_ROOT "${RISCV_NEWLIB_SYSROOT}" DIRECTORY)

# Define Tools
set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang++")
set(CMAKE_ASM_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang")
set(CMAKE_LINKER "${RISCV_GCC_ROOT}/bin/riscv64-unknown-elf-gcc") # Use GCC for
                                                                  # linking

set(CMAKE_AR "${RISCV_TOOLCHAIN_ROOT}/bin/llvm-ar")
set(CMAKE_RANLIB "${RISCV_TOOLCHAIN_ROOT}/bin/llvm-ranlib")
set(CMAKE_STRIP "${RISCV_TOOLCHAIN_ROOT}/bin/llvm-strip")
set(CMAKE_SYSROOT "${RISCV_NEWLIB_SYSROOT}")

# --- 3. Find C++ Headers for Clang ---
file(GLOB CPP_INCLUDE_DIRS "${RISCV_NEWLIB_SYSROOT}/include/c++/*")
if(CPP_INCLUDE_DIRS)
  list(GET CPP_INCLUDE_DIRS 0 CPP_INCLUDE_DIR)
else()
  message(
    WARNING "Could not find C++ headers in ${RISCV_NEWLIB_SYSROOT}/include/c++")
endif()

# --- 4. Flag Definitions ---

# Paths to linker scripts (co-located with this toolchain file)
set(SCRIPTS_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(SPECS_FILE "${RISCV_NEWLIB_SYSROOT}/lib/htif.specs")
set(LINKER_SCRIPT "${SCRIPTS_DIR}/htif.ld")

# --- 3a. Toolchain profile ---
#
# `bare-metal` (default): builds a self-contained ELF for Chipyard VCS / FireSim
# under IREE_PLATFORM_GENERIC. Uses htif.ld + htif-nano.spec for the link.
#
# `zephyr`: builds IREE static archives (still under IREE_PLATFORM_GENERIC -- we
# do *not* fork IREE's platform layer for Zephyr) for consumption by a Zephyr
# application on chipyard_riscv64. Skips htif.ld + htif-nano.spec (Zephyr
# supplies its own linker script and crt) but keeps the IREE_PLATFORM_GENERIC
# defines so iree_bar compiles unmodified. Multi-hart parallelism is handled at
# the Zephyr-app layer (multiple k_threads each driving their own
# iree_vm_context); the IREE runtime itself is the same single-threaded
# local-sync executor used by the bare-metal flow.
set(MERLIN_TOOLCHAIN_PROFILE
    "bare-metal"
    CACHE STRING "Merlin toolchain profile (bare-metal | zephyr)")
set_property(CACHE MERLIN_TOOLCHAIN_PROFILE PROPERTY STRINGS bare-metal zephyr)

if(NOT MERLIN_TOOLCHAIN_PROFILE STREQUAL "bare-metal"
   AND NOT MERLIN_TOOLCHAIN_PROFILE STREQUAL "zephyr")
  message(
    FATAL_ERROR
      "MERLIN_TOOLCHAIN_PROFILE must be 'bare-metal' or 'zephyr', got "
      "'${MERLIN_TOOLCHAIN_PROFILE}'.")
endif()

set(ARCH_FLAGS "-march=rv64imafdc -mabi=lp64d -mcmodel=medany -mstrict-align")

# --- 4a. Bare-metal CPU feature detection ---
# On bare-metal there is no OS to query CPU features at runtime. Set
# IREE_BARE_METAL_CPU_DATA0 so the IREE runtime knows which ukernel
# implementations to select. Each hardware target sets
# IREE_RISCV_BARE_METAL_FEATURES via cmake cache (from build.py or recipe).
#
# Bit definitions (from cpu_feature_bits.inl): V        = 1 << 0 = 0x01 ZVFHMIN
# = 1 << 1 = 0x02 ZVFH     = 1 << 2 = 0x04 XSMTVDOT = 1 << 3 = 0x08 XOPU     = 1
# << 4 = 0x10
set(IREE_RISCV_BARE_METAL_FEATURES
    "0x01"
    CACHE STRING
          "Bitmask of IREE_CPU_DATA0_RISCV_64_* features for bare-metal targets"
)
set(IREE_BARE_METAL_CPU_FLAGS
    "-DIREE_BARE_METAL_CPU_DATA0=${IREE_RISCV_BARE_METAL_FEATURES}")

# 4b. Clang Compile Flags CRITICAL FIX: We use -Wno-error=... to ensure these
# specific warnings never stop the build, even if -Wall -Werror is appended
# later by IREE.
set(CLANG_COMPILE_FLAGS
    "\
--target=riscv64-unknown-elf \
--sysroot=${RISCV_NEWLIB_SYSROOT} \
-I${CPP_INCLUDE_DIR} \
-I${CPP_INCLUDE_DIR}/riscv64-unknown-elf \
-I${RISCV_NEWLIB_SYSROOT}/include \
${ARCH_FLAGS} \
-fno-pic \
-fno-plt \
-fno-common \
-fno-builtin-printf \
-Wno-error=unused-command-line-argument \
-Wno-error=unused-parameter \
-Wno-error=sign-compare \
-Wno-error=missing-field-initializers \
-Wno-error=pointer-sign \
-Wno-error=char-subscripts \
-Wno-error=type-limits \
-Daligned_alloc=memalign \
-DIREE_DEVICE_SIZE_T=uint64_t \
-DPRIdsz=PRIu64 \
-DIREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED \
${IREE_BARE_METAL_CPU_FLAGS}")

# Profile-conditional defines.
#
# Both profiles use IREE_PLATFORM_GENERIC -- we never modify iree_bar to add a
# Zephyr-specific platform. Differences:
#
# bare-metal: IREE_TIME_NOW_FN / IREE_WAIT_UNTIL_FN stub out clock + wait
# surfaces (newlib bare-metal has no clock_gettime / no wait primitives).
#
# zephyr: Real clock_gettime is available via Zephyr's POSIX subsystem (when the
# consuming app enables CONFIG_POSIX_API), so we let IREE's default time path
# (clock_gettime(CLOCK_MONOTONIC)) take effect by *not* defining
# IREE_TIME_NOW_FN. IREE_WAIT_UNTIL_FN remains stubbed: the local-sync HAL never
# blocks on a wait, so this is dead code.
if(MERLIN_TOOLCHAIN_PROFILE STREQUAL "bare-metal")
  # IREE_ASYNC_HAVE_FD=1: upstream socket.c references primitive.value.fd
  # unconditionally; the union member is gated on IREE_ASYNC_HAVE_FD. The
  # local-sync HAL never instantiates an iree_async_proactor_t so the socket
  # path is dead code that --gc-sections drops at link time. Same override used
  # by the zephyr profile.
  set(CLANG_COMPILE_FLAGS
      "${CLANG_COMPILE_FLAGS} \
-DIREE_PLATFORM_GENERIC=1 \
-DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1 \
-DIREE_FILE_IO_ENABLE=0 \
-DIREE_ASYNC_HAVE_FD=1 \
-DIREE_TIME_NOW_FN=\"{ return 0; }\" \
-DIREE_WAIT_UNTIL_FN=sizeof \
-isystem ${SCRIPTS_DIR}/zephyr_stubs")
else() # zephyr
  # Zephyr profile: build IREE archives under IREE_PLATFORM_ZEPHYR (not
  # GENERIC), with synchronization enabled — iree_slim_mutex and
  # iree_notification's futex code paths now resolve via stub futex symbols
  # provided by the Zephyr application (samples/merlin_model_runner/src/
  # iree_futex_zephyr.c, backed by k_mutex + k_condvar). Same for iree_thread_*
  # (iree_thread_zephyr.c, backed by k_thread). This is what lets
  # IREE_HAL_DRIVER_LOCAL_TASK execute on Zephyr SMP.
  #
  # IREE_ASYNC_HAVE_FD=1 is the upstream-supported override (see
  # iree/async/primitive.h:33-37 -- "Each IREE_ASYNC_HAVE_* define can be set
  # externally to override detection. This allows custom/embedded platforms to
  # opt-in to features their platform supports").
  set(CLANG_COMPILE_FLAGS
      "${CLANG_COMPILE_FLAGS} \
-DIREE_PLATFORM_ZEPHYR=1 \
-DIREE_FILE_IO_ENABLE=0 \
-DIREE_HAL_MODULE_LOAD_FROM_FILE_DISABLE=1 \
-DIREE_TIME_NOW_FN=\"{ return 0; }\" \
-DIREE_WAIT_UNTIL_FN=sizeof \
-DIREE_ASYNC_HAVE_FD=1 \
-D_POSIX_THREADS=200809L \
-D_POSIX_TIMEOUTS=200809L \
-D_UNIX98_THREAD_MUTEX_ATTRIBUTES=1 \
-isystem ${SCRIPTS_DIR}/zephyr_stubs")
endif()

# 4b. GCC Link Flags -specs=... handles system libs (libgloss, libc_nano, lgcc)
# automatically. -T ... handles the memory map.
#
# bare-metal: needs htif.specs + htif.ld so that printf/exit work via the HTIF
# bridge in Chipyard VCS / FireSim.
#
# zephyr: Zephyr's CMake supplies the linker script and crt itself; passing
# htif.ld here would conflict with Zephyr's `zephyr.lds`. We still pass
# ARCH_FLAGS + -static so the static archives the IREE build emits are
# ABI-compatible with the Zephyr application that links them.
if(MERLIN_TOOLCHAIN_PROFILE STREQUAL "bare-metal")
  set(GCC_LINK_FLAGS
      "\
${ARCH_FLAGS} \
-static \
-specs=${SPECS_FILE} \
-T${LINKER_SCRIPT}")
else() # zephyr
  set(GCC_LINK_FLAGS
      "\
${ARCH_FLAGS} \
-static")
endif()

# --- 5. Apply Flags to CMake Variables ---

set(CMAKE_C_FLAGS
    "${CLANG_COMPILE_FLAGS} -std=gnu11 -O2"
    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS
    "${CLANG_COMPILE_FLAGS} -std=gnu++17 -O2 -stdlib=libstdc++"
    CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS
    "${CLANG_COMPILE_FLAGS}"
    CACHE STRING "" FORCE)

# Clear standard CMake linker flags
set(CMAKE_EXE_LINKER_FLAGS
    ""
    CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS
    ""
    CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS
    ""
    CACHE STRING "" FORCE)

# --- 6. Override Link Rule ---
# GCC Driver Link Rule: 1. <OBJECTS>: Your compiled C/C++ files. 2.
# <LINK_LIBRARIES>: The IREE static libraries (.a files). 3. ${GCC_LINK_FLAGS}:
# The specs file and linker script.
set(CMAKE_C_LINK_EXECUTABLE
    "<CMAKE_LINKER> <OBJECTS> <LINK_LIBRARIES> ${GCC_LINK_FLAGS} -o <TARGET>")
set(CMAKE_CXX_LINK_EXECUTABLE
    "<CMAKE_LINKER> <OBJECTS> <LINK_LIBRARIES> ${GCC_LINK_FLAGS} -o <TARGET>")

# --- 7. IREE Options ---
set(CMAKE_CROSSCOMPILING ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Bare-metal newlib has no libdl; force CMAKE_DL_LIBS empty so IREE's
# unconditional `${CMAKE_DL_LIBS}` (in runtime/src/iree/base/internal &
# threading) does not cascade into a `-ldl` at link time. The dynamic-library
# code paths are dead under IREE_PLATFORM_GENERIC anyway.
#
# CMakeGenericSystem.cmake sets CMAKE_DL_LIBS="dl" *after* the toolchain file
# runs (during enable_language), so a CACHE FORCE here gets shadowed by the
# non-cached variable in the parsing scope. We install a project include hook
# that resets it once the project() call has completed.
set(CMAKE_DL_LIBS
    ""
    CACHE STRING "" FORCE)
set(CMAKE_PROJECT_INCLUDE
    "${CMAKE_CURRENT_LIST_DIR}/clear_dl_libs.cmake"
    CACHE FILEPATH "" FORCE)

# Force disable Warnings-as-Errors for IREE targets
set(IREE_BUILD_WARNINGS_AS_ERRORS
    OFF
    CACHE BOOL "" FORCE)
set(IREE_ENABLE_COMPILER_WARNINGS
    OFF
    CACHE BOOL "" FORCE)

set(IREE_BUILD_BINDINGS_TFLITE
    OFF
    CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE_JAVA
    OFF
    CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_DEFAULTS
    OFF
    CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_SYNC
    ON
    CACHE BOOL "" FORCE)
# local-task is only meaningful when threading is on (i.e., the zephyr profile).
# For bare-metal we don't bring it in.
if(MERLIN_TOOLCHAIN_PROFILE STREQUAL "zephyr")
  set(IREE_HAL_DRIVER_LOCAL_TASK
      ON
      CACHE BOOL "" FORCE)
endif()
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS
    OFF
    CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF
    ON
    CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE
    ON
    CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS
    OFF
    CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF
    ON
    CACHE BOOL "" FORCE)
# Bare-metal stays single-threaded (no IREE thread library).
#
# For the `zephyr` profile, threading is driven by --profile picks (zephyr →
# OFF, zephyr-task → ON) in tools/build.py, which passes
# -DIREE_ENABLE_THREADING=... explicitly on the cmake command line. Respect that
# if set; otherwise fall through to platform defaults (OFF for bare-metal, OFF
# for zephyr lean variant by convention).
if(NOT DEFINED IREE_ENABLE_THREADING)
  set(IREE_ENABLE_THREADING
      OFF
      CACHE BOOL "" FORCE)
endif()
