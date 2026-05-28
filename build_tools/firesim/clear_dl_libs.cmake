# Re-clear CMAKE_DL_LIBS after CMakeGenericSystem.cmake set it to "dl".
# Bare-metal newlib has no libdl; the IREE dynamic-library code paths are dead
# under IREE_PLATFORM_GENERIC and --gc-sections drops them.
set(CMAKE_DL_LIBS "")
