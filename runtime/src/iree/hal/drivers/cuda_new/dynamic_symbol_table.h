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

// Symbol resolution.
IREE_CU_PFN_DECL(cuGetProcAddress, const char *, void **, int, cuuint64_t)
