// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Version and initialization.
IREE_NCCL_PFN_DECL(ncclGetVersion, int*)
IREE_NCCL_PFN_DECL(ncclGetUniqueId, ncclUniqueId*)
IREE_NCCL_PFN_DECL(ncclCommInitRank, ncclComm_t*, int, ncclUniqueId, int)
IREE_NCCL_PFN_DECL(ncclCommDestroy, ncclComm_t)

// Error reporting.
IREE_NCCL_PFN_DECL_STR_RETURN(ncclGetErrorString, ncclResult_t)

// Collective operations.
IREE_NCCL_PFN_DECL(ncclAllReduce, const void*, void*, size_t, ncclDataType_t,
	ncclRedOp_t, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclBroadcast, const void*, void*, size_t, ncclDataType_t,
	int, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclAllGather, const void*, void*, size_t, ncclDataType_t,
	ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclReduceScatter, const void*, void*, size_t,
	ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclSend, const void*, size_t, ncclDataType_t, int,
	ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclRecv, void*, size_t, ncclDataType_t, int, ncclComm_t,
	cudaStream_t)

// Group operations.
IREE_NCCL_PFN_DECL(ncclGroupStart)
IREE_NCCL_PFN_DECL(ncclGroupEnd)
