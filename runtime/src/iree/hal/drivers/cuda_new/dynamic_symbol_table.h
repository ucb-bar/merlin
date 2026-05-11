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

// Function management.
IREE_CU_PFN_DECL(cuFuncSetAttribute, CUfunction, CUfunction_attribute, int)

// Event management.
IREE_CU_PFN_DECL(cuEventCreate, CUevent *, unsigned int)
IREE_CU_PFN_DECL(cuEventDestroy, CUevent)
IREE_CU_PFN_DECL(cuEventRecord, CUevent, CUstream)
IREE_CU_PFN_DECL(cuEventSynchronize, CUevent)
IREE_CU_PFN_DECL(cuEventQuery, CUevent)
IREE_CU_PFN_DECL(cuEventElapsedTime, float *, CUevent, CUevent)
IREE_CU_PFN_DECL(cuGraphAddEventRecordNode, CUgraphNode *, CUgraph,
	const CUgraphNode *, size_t, CUevent)
IREE_CU_PFN_DECL(cuStreamWaitEvent, CUstream, CUevent, unsigned int)

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
IREE_CU_PFN_DECL(cuLaunchHostFunc, CUstream, CUhostFn, void *)

// Memory pool management.
IREE_CU_PFN_DECL(cuMemPoolCreate, CUmemoryPool *, const CUmemPoolProps *)
IREE_CU_PFN_DECL(cuMemPoolDestroy, CUmemoryPool)
IREE_CU_PFN_DECL(cuMemPoolSetAttribute, CUmemoryPool, CUmemPool_attribute,
	void *)
IREE_CU_PFN_DECL(cuMemPoolTrimTo, CUmemoryPool, size_t)
IREE_CU_PFN_DECL(cuMemAllocFromPoolAsync, CUdeviceptr *, size_t,
	CUmemoryPool, CUstream)
IREE_CU_PFN_DECL(cuMemFreeAsync, CUdeviceptr, CUstream)

// Graph capture and execution.
IREE_CU_PFN_DECL(cuStreamBeginCapture, CUstream, CUstreamCaptureMode)
IREE_CU_PFN_DECL(cuStreamEndCapture, CUstream, CUgraph *)
IREE_CU_PFN_DECL(cuGraphInstantiate, CUgraphExec *, CUgraph, unsigned long long)
IREE_CU_PFN_DECL(cuGraphLaunch, CUgraphExec, CUstream)
IREE_CU_PFN_DECL(cuGraphExecDestroy, CUgraphExec)
IREE_CU_PFN_DECL(cuGraphDestroy, CUgraph)

// Stream management.
IREE_CU_PFN_DECL(cuStreamCreate, CUstream *, unsigned int)
IREE_CU_PFN_DECL(cuStreamDestroy, CUstream)
IREE_CU_PFN_DECL(cuStreamSynchronize, CUstream)

// Symbol resolution.
IREE_CU_PFN_DECL(cuGetProcAddress, const char *, void **, int, cuuint64_t)
