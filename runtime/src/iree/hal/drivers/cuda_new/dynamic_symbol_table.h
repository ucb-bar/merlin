// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Error reporting — loaded first.
IREE_CU_PFN_DECL(cuGetErrorName, CUresult, const char **)
IREE_CU_PFN_DECL(cuGetErrorString, CUresult, const char **)

// Driver and version.
IREE_CU_PFN_DECL(cuInit, unsigned int)
IREE_CU_PFN_DECL(cuDriverGetVersion, int *)

// Device enumeration and properties.
IREE_CU_PFN_DECL(cuDeviceGet, CUdevice *, int)
IREE_CU_PFN_DECL(cuDeviceGetCount, int *)
IREE_CU_PFN_DECL(cuDeviceGetName, char *, int, CUdevice)
IREE_CU_PFN_DECL(cuDeviceGetAttribute, int *, CUdevice_attribute, CUdevice)
IREE_CU_PFN_DECL(cuDeviceGetUuid, CUuuid *, CUdevice)

// Context management (primary context model).
IREE_CU_PFN_DECL(cuDevicePrimaryCtxRetain, CUcontext *, CUdevice)
IREE_CU_PFN_DECL(cuDevicePrimaryCtxRelease, CUdevice)
IREE_CU_PFN_DECL(cuCtxSetCurrent, CUcontext)

// Memory allocation.
IREE_CU_PFN_DECL(cuMemAlloc, CUdeviceptr *, size_t)
IREE_CU_PFN_DECL(cuMemFree, CUdeviceptr)
IREE_CU_PFN_DECL(cuMemAllocManaged, CUdeviceptr *, size_t, unsigned int)
IREE_CU_PFN_DECL(cuMemHostAlloc, void **, size_t, unsigned int)
IREE_CU_PFN_DECL(cuMemFreeHost, void *)
IREE_CU_PFN_DECL(cuMemHostGetDevicePointer, CUdeviceptr *, void *, unsigned int)
IREE_CU_PFN_DECL(cuMemHostRegister, void *, size_t, unsigned int)
IREE_CU_PFN_DECL(cuMemHostUnregister, void *)
IREE_CU_PFN_DECL(cuMemPrefetchAsync, CUdeviceptr, size_t, CUdevice, CUstream)

// Module management.
IREE_CU_PFN_DECL(cuModuleLoadDataEx, CUmodule *, const void *, unsigned int,
	CUjit_option *, void **)
IREE_CU_PFN_DECL(cuModuleUnload, CUmodule)
IREE_CU_PFN_DECL(cuModuleGetFunction, CUfunction *, CUmodule, const char *)

// Memory transfer.
IREE_CU_PFN_DECL(cuMemcpyAsync, CUdeviceptr, CUdeviceptr, size_t, CUstream)
IREE_CU_PFN_DECL(cuMemcpyHtoDAsync, CUdeviceptr, const void *, size_t,
	CUstream)
IREE_CU_PFN_DECL(cuMemsetD8Async, CUdeviceptr, unsigned char, size_t,
	CUstream)
IREE_CU_PFN_DECL(cuMemsetD16Async, CUdeviceptr, unsigned short, size_t,
	CUstream)
IREE_CU_PFN_DECL(cuMemsetD32Async, CUdeviceptr, unsigned int, size_t,
	CUstream)

// Kernel launch.
IREE_CU_PFN_DECL(cuLaunchKernel, CUfunction, unsigned int, unsigned int,
	unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
	CUstream, void **, void **)

// Stream management.
IREE_CU_PFN_DECL(cuStreamCreate, CUstream *, unsigned int)
IREE_CU_PFN_DECL(cuStreamDestroy, CUstream)
IREE_CU_PFN_DECL(cuStreamSynchronize, CUstream)

// Symbol resolution.
IREE_CU_PFN_DECL(cuGetProcAddress, const char *, void **, int, cuuint64_t)
