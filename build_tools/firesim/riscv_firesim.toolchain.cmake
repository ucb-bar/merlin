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
-DIREE_PLATFORM_GENERIC=1 \
-DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1 \
-DIREE_FILE_IO_ENABLE=0 \
-DIREE_TIME_NOW_FN=\"{ return 0; }\" \
-DIREE_WAIT_UNTIL_FN=sizeof \
-DIREE_DEVICE_SIZE_T=uint64_t \
-DPRIdsz=PRIu64 \
-DIREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED \
${IREE_BARE_METAL_CPU_FLAGS}")

# 4b. GCC Link Flags -specs=... handles system libs (libgloss, libc_nano, lgcc)
# automatically. -T ... handles the memory map.
set(GCC_LINK_FLAGS
    "\
${ARCH_FLAGS} \
-static \
-specs=${SPECS_FILE} \
-T${LINKER_SCRIPT}")

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
set(IREE_ENABLE_THREADING
    OFF
    CACHE BOOL "" FORCE)
