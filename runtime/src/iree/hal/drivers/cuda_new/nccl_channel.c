// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nccl_channel.h"

#include <pthread.h>
#include <stddef.h>
#include <string.h>
#include <time.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "buffer.h"
#include "nccl_status_util.h"
#include "status_util.h"

//===----------------------------------------------------------------------===//
// iree_hal_cuda_new_nccl_channel_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda_new_nccl_channel_t {
	iree_hal_resource_t resource;

	const iree_hal_cuda_new_dynamic_symbols_t *cuda_symbols;
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *nccl_symbols;

	iree_allocator_t host_allocator;

	// Parent channel this was split from, if any.
	iree_hal_channel_t *parent_channel;

	// This participant's rank in the communicator.
	int rank;
	// Total number of participants in the communicator.
	int count;

	// Communicator handle.
	ncclComm_t comm;
} iree_hal_cuda_new_nccl_channel_t;

static const iree_hal_channel_vtable_t iree_hal_cuda_new_nccl_channel_vtable;

static iree_status_t iree_hal_cuda_new_nccl_comm_init_rank(
	const iree_hal_cuda_new_dynamic_symbols_t *cuda_symbols,
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *symbols,
	CUcontext context, const iree_hal_cuda_new_nccl_id_t *id, int rank, int count,
	ncclComm_t *out_comm);

static iree_hal_cuda_new_nccl_channel_t *
iree_hal_cuda_new_nccl_channel_cast(iree_hal_channel_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_new_nccl_channel_vtable);
	return (iree_hal_cuda_new_nccl_channel_t *)base_value;
}

static const iree_hal_cuda_new_nccl_channel_t *
iree_hal_cuda_new_nccl_channel_const_cast(
	const iree_hal_channel_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_new_nccl_channel_vtable);
	return (const iree_hal_cuda_new_nccl_channel_t *)base_value;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cuda_new_nccl_get_unique_id(
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *symbols,
	iree_hal_cuda_new_nccl_id_t *out_id) {
	static_assert(sizeof(*out_id) == sizeof(ncclUniqueId),
		"NCCL ID size mismatch");

	IREE_ASSERT_ARGUMENT(symbols);
	IREE_ASSERT_ARGUMENT(out_id);
	IREE_TRACE_ZONE_BEGIN(z0);

	memset(out_id, 0, sizeof(*out_id));
	iree_status_t status = IREE_NCCL_RESULT_TO_STATUS_NEW(
		symbols, ncclGetUniqueId((ncclUniqueId *)out_id), "ncclGetUniqueId");

	IREE_TRACE_ZONE_END(z0);
	return status;
}

iree_status_t iree_hal_cuda_new_nccl_channel_create(
	const iree_hal_cuda_new_dynamic_symbols_t *cuda_symbols,
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *nccl_symbols,
	CUcontext context, const iree_hal_cuda_new_nccl_id_t *id, int rank, int count,
	iree_allocator_t host_allocator, iree_hal_channel_t **out_channel) {
	IREE_ASSERT_ARGUMENT(cuda_symbols);
	IREE_ASSERT_ARGUMENT(nccl_symbols);
	IREE_ASSERT_ARGUMENT(id);
	IREE_ASSERT_ARGUMENT(out_channel);
	IREE_TRACE_ZONE_BEGIN(z0);

	*out_channel = NULL;

	ncclComm_t comm = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_hal_cuda_new_nccl_comm_init_rank(cuda_symbols, nccl_symbols,
			context, id, rank, count, &comm));

	iree_hal_cuda_new_nccl_channel_t *channel = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(
		z0, iree_allocator_malloc(host_allocator, sizeof(*channel),
				(void **)&channel));

	iree_hal_resource_initialize(&iree_hal_cuda_new_nccl_channel_vtable,
		&channel->resource);
	channel->cuda_symbols = cuda_symbols;
	channel->nccl_symbols = nccl_symbols;
	channel->host_allocator = host_allocator;
	channel->parent_channel = NULL;
	channel->rank = rank;
	channel->count = count;
	channel->comm = comm;
	*out_channel = (iree_hal_channel_t *)channel;

	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Channel vtable implementation
//===----------------------------------------------------------------------===//

static void iree_hal_cuda_new_nccl_channel_destroy(
	iree_hal_channel_t *base_channel) {
	iree_hal_cuda_new_nccl_channel_t *channel =
		iree_hal_cuda_new_nccl_channel_cast(base_channel);
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_allocator_t host_allocator = channel->host_allocator;

	IREE_NCCL_IGNORE_ERROR_NEW(channel->nccl_symbols,
		ncclCommDestroy(channel->comm));

	iree_hal_channel_release(channel->parent_channel);
	iree_allocator_free(host_allocator, channel);

	IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_new_nccl_channel_split(
	iree_hal_channel_t *base_channel, int32_t color, int32_t key,
	iree_hal_channel_flags_t flags,
	iree_hal_channel_t **out_split_channel) {
	// TODO: implement split using ncclCommSplit when needed.
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"NCCL channel split not yet implemented in cuda_new driver");
}

static void iree_hal_cuda_new_nccl_channel_query_rank_and_count(
	const iree_hal_channel_t *base_channel, int32_t *out_rank,
	int32_t *out_count) {
	IREE_ASSERT_ARGUMENT(base_channel);
	IREE_ASSERT_ARGUMENT(out_count);
	const iree_hal_cuda_new_nccl_channel_t *channel =
		iree_hal_cuda_new_nccl_channel_const_cast(base_channel);
	*out_rank = channel->rank;
	*out_count = channel->count;
}

//===----------------------------------------------------------------------===//
// Collective batch submission
//===----------------------------------------------------------------------===//

// Returns the NCCL communicator for the given |channel|.
static ncclComm_t iree_hal_cuda_new_nccl_channel_comm(
	iree_hal_channel_t *base_channel) {
	IREE_ASSERT_ARGUMENT(base_channel);
	iree_hal_cuda_new_nccl_channel_t *channel =
		iree_hal_cuda_new_nccl_channel_cast(base_channel);
	return channel->comm;
}

static iree_status_t iree_hal_cuda_new_nccl_submit_batch_entry(
	const iree_hal_collective_batch_entry_t *entry, CUstream stream);

typedef struct iree_hal_cuda_new_nccl_pending_entry_t {
	const iree_hal_collective_batch_entry_t *entry;
	CUcontext context;
	CUstream stream;
	iree_status_t status;
	bool completed;
} iree_hal_cuda_new_nccl_pending_entry_t;

typedef struct iree_hal_cuda_new_nccl_pending_group_t {
	bool active;
	int count;
	iree_hal_collective_op_t op;
	uint32_t param;
	iree_device_size_t element_count;
	iree_hal_cuda_new_nccl_pending_entry_t entries[16];
	int entry_count;
} iree_hal_cuda_new_nccl_pending_group_t;

static pthread_mutex_t iree_hal_cuda_new_nccl_group_mutex =
	PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t iree_hal_cuda_new_nccl_group_cond =
	PTHREAD_COND_INITIALIZER;
static iree_hal_cuda_new_nccl_pending_group_t
	iree_hal_cuda_new_nccl_pending_group;

typedef struct iree_hal_cuda_new_nccl_pending_init_entry_t {
	int rank;
	CUcontext context;
	ncclComm_t *out_comm;
	iree_status_t status;
	bool completed;
} iree_hal_cuda_new_nccl_pending_init_entry_t;

typedef struct iree_hal_cuda_new_nccl_pending_init_group_t {
	bool active;
	iree_hal_cuda_new_nccl_id_t id;
	int count;
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *symbols;
	iree_hal_cuda_new_nccl_pending_init_entry_t entries[16];
	int entry_count;
} iree_hal_cuda_new_nccl_pending_init_group_t;

// Group init globals — disabled while direct ncclCommInitRank path is used.
#if 0
static pthread_mutex_t iree_hal_cuda_new_nccl_init_mutex =
	PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t iree_hal_cuda_new_nccl_init_cond =
	PTHREAD_COND_INITIALIZER;
static iree_hal_cuda_new_nccl_pending_init_group_t
	iree_hal_cuda_new_nccl_pending_init_group;
#endif  // group init disabled

static bool iree_hal_cuda_new_nccl_group_matches(
	const iree_hal_cuda_new_nccl_pending_group_t *group,
	const iree_hal_collective_batch_entry_t *entry,
	const iree_hal_cuda_new_nccl_channel_t *channel) {
	return group->active && group->count == channel->count &&
		   group->op.packed == entry->op.packed &&
		   group->param == entry->param &&
		   group->element_count == entry->element_count;
}

static void iree_hal_cuda_new_nccl_deadline_from_now(
	int milliseconds, struct timespec *out_time) {
	clock_gettime(CLOCK_REALTIME, out_time);
	out_time->tv_nsec += (long)milliseconds * 1000 * 1000;
	out_time->tv_sec += out_time->tv_nsec / (1000 * 1000 * 1000);
	out_time->tv_nsec %= 1000 * 1000 * 1000;
}

static bool iree_hal_cuda_new_nccl_init_group_matches(
	const iree_hal_cuda_new_nccl_pending_init_group_t *group,
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *symbols,
	const iree_hal_cuda_new_nccl_id_t *id, int count) {
	return group->active && group->symbols == symbols &&
		   group->count == count && memcmp(&group->id, id, sizeof(*id)) == 0;
}

static iree_status_t iree_hal_cuda_new_nccl_comm_init_rank(
	const iree_hal_cuda_new_dynamic_symbols_t *cuda_symbols,
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *symbols,
	CUcontext context, const iree_hal_cuda_new_nccl_id_t *id, int rank, int count,
	ncclComm_t *out_comm) {
	// Multi-process (MPI) path: each process calls ncclCommInitRank directly.
	// NCCL handles the cross-process rendezvous internally via the unique ID.
	IREE_RETURN_IF_ERROR(IREE_CURESULT_TO_STATUS_NEW(
		cuda_symbols, cuCtxSetCurrent(context), "cuCtxSetCurrent"));
	return IREE_NCCL_RESULT_TO_STATUS_NEW(
		symbols,
		ncclCommInitRank(out_comm, count, *((const ncclUniqueId *)id),
			rank),
		"ncclCommInitRank");

	// The group init path below is for intra-process multi-thread usage
	// (e.g. the direct HAL test). It batches ncclCommInitRank calls inside
	// ncclGroupStart/End. Currently disabled; re-enable when needed.
#if 0
	pthread_mutex_lock(&iree_hal_cuda_new_nccl_init_mutex);
	iree_hal_cuda_new_nccl_pending_init_group_t *group =
		&iree_hal_cuda_new_nccl_pending_init_group;
	if (!group->active) {
		memset(group, 0, sizeof(*group));
		group->active = true;
		group->id = *id;
		group->count = count;
		group->symbols = symbols;
	}
	if (!iree_hal_cuda_new_nccl_init_group_matches(group, symbols, id,
			count)) {
		pthread_mutex_unlock(&iree_hal_cuda_new_nccl_init_mutex);
		IREE_RETURN_IF_ERROR(IREE_CURESULT_TO_STATUS_NEW(
			cuda_symbols, cuCtxSetCurrent(context), "cuCtxSetCurrent"));
		return IREE_NCCL_RESULT_TO_STATUS_NEW(
			symbols,
			ncclCommInitRank(out_comm, count, *((const ncclUniqueId *)id),
				rank),
			"ncclCommInitRank");
	}
	int slot = group->entry_count++;
	group->entries[slot].rank = rank;
	group->entries[slot].context = context;
	group->entries[slot].out_comm = out_comm;
	group->entries[slot].status = iree_ok_status();
	group->entries[slot].completed = false;

	if (group->entry_count == group->count) {
		iree_status_t status = IREE_NCCL_RESULT_TO_STATUS_NEW(
			symbols, ncclGroupStart(), "ncclGroupStart");
		for (int i = 0; i < group->entry_count && iree_status_is_ok(status);
			 ++i) {
			status = IREE_CURESULT_TO_STATUS_NEW(
				cuda_symbols, cuCtxSetCurrent(group->entries[i].context),
				"cuCtxSetCurrent");
			if (!iree_status_is_ok(status)) break;
			status = IREE_NCCL_RESULT_TO_STATUS_NEW(
				symbols,
				ncclCommInitRank(group->entries[i].out_comm, count,
					*((const ncclUniqueId *)id), group->entries[i].rank),
				"ncclCommInitRank");
		}
		if (iree_status_is_ok(status)) {
			status = IREE_NCCL_RESULT_TO_STATUS_NEW(
				symbols, ncclGroupEnd(), "ncclGroupEnd");
		} else {
			IREE_NCCL_IGNORE_ERROR_NEW(symbols, ncclGroupEnd());
		}
		for (int i = 0; i < group->entry_count; ++i) {
			group->entries[i].status =
				i == slot ? status : iree_status_clone(status);
			group->entries[i].completed = true;
		}
		group->active = false;
		pthread_cond_broadcast(&iree_hal_cuda_new_nccl_init_cond);
		pthread_mutex_unlock(&iree_hal_cuda_new_nccl_init_mutex);
		return status;
	}

	struct timespec deadline;
	iree_hal_cuda_new_nccl_deadline_from_now(250, &deadline);
	while (!group->entries[slot].completed && group->active) {
		int wait_result = pthread_cond_timedwait(
			&iree_hal_cuda_new_nccl_init_cond,
			&iree_hal_cuda_new_nccl_init_mutex, &deadline);
		if (wait_result != 0) break;
	}
	if (group->entries[slot].completed) {
		iree_status_t status = group->entries[slot].status;
		pthread_mutex_unlock(&iree_hal_cuda_new_nccl_init_mutex);
		return status;
	}
	for (int i = slot; i + 1 < group->entry_count; ++i) {
		group->entries[i] = group->entries[i + 1];
	}
	--group->entry_count;
	if (group->entry_count == 0) group->active = false;
	pthread_mutex_unlock(&iree_hal_cuda_new_nccl_init_mutex);
	IREE_RETURN_IF_ERROR(IREE_CURESULT_TO_STATUS_NEW(
		cuda_symbols, cuCtxSetCurrent(context), "cuCtxSetCurrent"));
	return IREE_NCCL_RESULT_TO_STATUS_NEW(
		symbols,
		ncclCommInitRank(out_comm, count, *((const ncclUniqueId *)id),
			rank),
		"ncclCommInitRank");
#endif  // group init disabled
}

static iree_status_t iree_hal_cuda_new_nccl_get_data_type(
	iree_hal_collective_element_type_t in, ncclDataType_t *out) {
	switch (in) {
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_8:
			*out = ncclInt8;
			break;
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_8:
			*out = ncclUint8;
			break;
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_16:
			return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
				"SINT16 is not supported for collective op");
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_16:
			return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
				"UINT16 is not supported for collective op");
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_32:
			*out = ncclInt32;
			break;
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_32:
			*out = ncclUint32;
			break;
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_64:
			*out = ncclInt64;
			break;
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_64:
			*out = ncclUint64;
			break;
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_16:
			*out = ncclFloat16;
			break;
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_32:
			*out = ncclFloat32;
			break;
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_64:
			*out = ncclFloat64;
			break;
		case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_BFLOAT_16:
			return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
				"bfloat16 collectives not supported by this NCCL build");
			break;
		default:
			return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
				"unhandled element type for collective op");
	}
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_nccl_get_reduction_type(
	iree_hal_collective_reduction_t in, ncclRedOp_t *out) {
	switch (in) {
		case IREE_HAL_COLLECTIVE_REDUCTION_SUM:
			*out = ncclSum;
			break;
		case IREE_HAL_COLLECTIVE_REDUCTION_PRODUCT:
			*out = ncclProd;
			break;
		case IREE_HAL_COLLECTIVE_REDUCTION_MINIMUM:
			*out = ncclMin;
			break;
		case IREE_HAL_COLLECTIVE_REDUCTION_MAXIMUM:
			*out = ncclMax;
			break;
		case IREE_HAL_COLLECTIVE_REDUCTION_AVERAGE:
			*out = ncclAvg;
			break;
		default:
			return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
				"unhandled reduction type for collective op");
	}
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_nccl_submit_batch_entry(
	const iree_hal_collective_batch_entry_t *entry, CUstream stream) {
	IREE_ASSERT_ARGUMENT(entry);

	iree_hal_cuda_new_nccl_channel_t *channel =
		iree_hal_cuda_new_nccl_channel_cast(entry->channel);
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *symbols =
		channel->nccl_symbols;
	ncclComm_t comm = iree_hal_cuda_new_nccl_channel_comm(entry->channel);

	ncclDataType_t datatype;
	IREE_RETURN_IF_ERROR(
		iree_hal_cuda_new_nccl_get_data_type(entry->op.element_type,
			&datatype));

	switch (entry->op.kind) {
		case IREE_HAL_COLLECTIVE_KIND_ALL_GATHER: {
			CUdeviceptr sendbuff =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(
						entry->send_binding.buffer)) +
				iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
				entry->send_binding.offset;
			CUdeviceptr recvbuff =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(
						entry->recv_binding.buffer)) +
				iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
				entry->recv_binding.offset;
			IREE_NCCL_RETURN_IF_ERROR_NEW(
				symbols,
				ncclAllGather((const void *)sendbuff, (void *)recvbuff,
					entry->element_count, datatype, comm, stream),
				"ncclAllGather");
			break;
		}
		case IREE_HAL_COLLECTIVE_KIND_ALL_REDUCE: {
			CUdeviceptr sendbuff =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(
						entry->send_binding.buffer)) +
				iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
				entry->send_binding.offset;
			CUdeviceptr recvbuff =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(
						entry->recv_binding.buffer)) +
				iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
				entry->recv_binding.offset;
			ncclRedOp_t redop;
			IREE_RETURN_IF_ERROR(
				iree_hal_cuda_new_nccl_get_reduction_type(
					entry->op.reduction, &redop));
			IREE_NCCL_RETURN_IF_ERROR_NEW(
				symbols,
				ncclAllReduce((const void *)sendbuff, (void *)recvbuff,
					entry->element_count, datatype, redop, comm, stream),
				"ncclAllReduce");
			break;
		}
		case IREE_HAL_COLLECTIVE_KIND_BROADCAST: {
			CUdeviceptr sendbuff =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(
						entry->send_binding.buffer)) +
				iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
				entry->send_binding.offset;
			CUdeviceptr recvbuff =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(
						entry->recv_binding.buffer)) +
				iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
				entry->recv_binding.offset;
			IREE_NCCL_RETURN_IF_ERROR_NEW(
				symbols,
				ncclBroadcast((const void *)sendbuff, (void *)recvbuff,
					entry->element_count, datatype, entry->param, comm,
					stream),
				"ncclBroadcast");
			break;
		}
		case IREE_HAL_COLLECTIVE_KIND_REDUCE_SCATTER: {
			CUdeviceptr sendbuff =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(
						entry->send_binding.buffer)) +
				iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
				entry->send_binding.offset;
			CUdeviceptr recvbuff =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(
						entry->recv_binding.buffer)) +
				iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
				entry->recv_binding.offset;
			ncclRedOp_t redop;
			IREE_RETURN_IF_ERROR(
				iree_hal_cuda_new_nccl_get_reduction_type(
					entry->op.reduction, &redop));
			IREE_NCCL_RETURN_IF_ERROR_NEW(
				symbols,
				ncclReduceScatter((const void *)sendbuff, (void *)recvbuff,
					entry->element_count, datatype, redop, comm, stream),
				"ncclReduceScatter");
			break;
		}
		case IREE_HAL_COLLECTIVE_KIND_SEND: {
			CUdeviceptr sendbuff =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(
						entry->send_binding.buffer)) +
				iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
				entry->send_binding.offset;
			IREE_NCCL_RETURN_IF_ERROR_NEW(
				symbols,
				ncclSend((const void *)sendbuff, entry->element_count,
					datatype, entry->param, comm, stream),
				"ncclSend");
			break;
		}
		case IREE_HAL_COLLECTIVE_KIND_RECV: {
			CUdeviceptr recvbuff =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(
						entry->recv_binding.buffer)) +
				iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
				entry->recv_binding.offset;
			IREE_NCCL_RETURN_IF_ERROR_NEW(
				symbols,
				ncclRecv((void *)recvbuff, entry->element_count, datatype,
					entry->param, comm, stream),
				"ncclRecv");
			break;
		}
		default:
			return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
				"unhandled collective op kind %d", (int)entry->op.kind);
	} // switch
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_nccl_submit_batch_same_process(
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *nccl_symbols,
	const iree_hal_collective_batch_t *batch, CUcontext context,
	CUstream stream,
	bool *out_submitted) {
	*out_submitted = false;
	if (batch->count != 1) return iree_ok_status();

	const iree_hal_collective_batch_entry_t *entry = &batch->entries[0];
	iree_hal_cuda_new_nccl_channel_t *channel =
		iree_hal_cuda_new_nccl_channel_cast(entry->channel);
	if (channel->count <= 1 ||
		channel->count >
			(int)IREE_ARRAYSIZE(iree_hal_cuda_new_nccl_pending_group.entries)) {
		return iree_ok_status();
	}

	pthread_mutex_lock(&iree_hal_cuda_new_nccl_group_mutex);
	iree_hal_cuda_new_nccl_pending_group_t *group =
		&iree_hal_cuda_new_nccl_pending_group;
	if (!group->active) {
		memset(group, 0, sizeof(*group));
		group->active = true;
		group->count = channel->count;
		group->op = entry->op;
		group->param = entry->param;
		group->element_count = entry->element_count;
	}

	if (!iree_hal_cuda_new_nccl_group_matches(group, entry, channel)) {
		pthread_mutex_unlock(&iree_hal_cuda_new_nccl_group_mutex);
		return iree_ok_status();
	}

	int slot = group->entry_count++;
	group->entries[slot].entry = entry;
	group->entries[slot].context = context;
	group->entries[slot].stream = stream;
	group->entries[slot].status = iree_ok_status();
	group->entries[slot].completed = false;

	if (group->entry_count == group->count) {
		iree_status_t status = IREE_NCCL_RESULT_TO_STATUS_NEW(
			nccl_symbols, ncclGroupStart(), "ncclGroupStart");
		for (int i = 0; i < group->entry_count && iree_status_is_ok(status);
			 ++i) {
			status = IREE_CURESULT_TO_STATUS_NEW(
				channel->cuda_symbols,
				cuCtxSetCurrent(group->entries[i].context),
				"cuCtxSetCurrent");
			if (!iree_status_is_ok(status)) break;
			status = iree_hal_cuda_new_nccl_submit_batch_entry(
				group->entries[i].entry, group->entries[i].stream);
		}
		if (iree_status_is_ok(status)) {
			status = IREE_NCCL_RESULT_TO_STATUS_NEW(
				nccl_symbols, ncclGroupEnd(), "ncclGroupEnd");
		} else {
			IREE_NCCL_IGNORE_ERROR_NEW(nccl_symbols, ncclGroupEnd());
		}
		for (int i = 0; i < group->entry_count; ++i) {
			group->entries[i].status =
				i == slot ? status : iree_status_clone(status);
			group->entries[i].completed = true;
		}
		group->active = false;
		pthread_cond_broadcast(&iree_hal_cuda_new_nccl_group_cond);
		pthread_mutex_unlock(&iree_hal_cuda_new_nccl_group_mutex);
		*out_submitted = true;
		return status;
	}

	struct timespec deadline;
	iree_hal_cuda_new_nccl_deadline_from_now(250, &deadline);
	while (!group->entries[slot].completed && group->active) {
		int wait_result = pthread_cond_timedwait(
			&iree_hal_cuda_new_nccl_group_cond,
			&iree_hal_cuda_new_nccl_group_mutex, &deadline);
		if (wait_result != 0) break;
	}

	if (group->entries[slot].completed) {
		iree_status_t status = group->entries[slot].status;
		pthread_mutex_unlock(&iree_hal_cuda_new_nccl_group_mutex);
		*out_submitted = true;
		return status;
	}

	// No matching same-process peers arrived. Fall back to normal NCCL
	// submission; this preserves distributed multi-process behavior.
	for (int i = slot; i + 1 < group->entry_count; ++i) {
		group->entries[i] = group->entries[i + 1];
	}
	--group->entry_count;
	if (group->entry_count == 0) group->active = false;
	pthread_mutex_unlock(&iree_hal_cuda_new_nccl_group_mutex);
	return iree_ok_status();
}

iree_status_t iree_hal_cuda_new_nccl_submit_batch(
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *nccl_symbols,
	const iree_hal_collective_batch_t *batch, CUcontext context,
	CUstream stream) {
	IREE_ASSERT_ARGUMENT(nccl_symbols);
	IREE_ASSERT_ARGUMENT(batch);
	IREE_TRACE_ZONE_BEGIN(z0);

	bool submitted = false;
	iree_status_t same_process_status =
		iree_hal_cuda_new_nccl_submit_batch_same_process(
			nccl_symbols, batch, context, stream, &submitted);
	if (submitted || !iree_status_is_ok(same_process_status)) {
		IREE_TRACE_ZONE_END(z0);
		return same_process_status;
	}

	// Issue all collective operations in the batch as part of a group.
	// NCCL may be able to fuse or reduce overheads by issuing like this.
	IREE_NCCL_RETURN_AND_END_ZONE_IF_ERROR_NEW(
		z0, nccl_symbols, ncclGroupStart(), "ncclGroupStart");
	for (iree_host_size_t i = 0; i < batch->count; ++i) {
		iree_status_t status = IREE_CURESULT_TO_STATUS_NEW(
			((iree_hal_cuda_new_nccl_channel_t *)batch->entries[i].channel)
				->cuda_symbols,
			cuCtxSetCurrent(context), "cuCtxSetCurrent");
		if (!iree_status_is_ok(status)) {
			IREE_NCCL_IGNORE_ERROR_NEW(nccl_symbols, ncclGroupEnd());
			IREE_TRACE_ZONE_END(z0);
			return status;
		}
		status = iree_hal_cuda_new_nccl_submit_batch_entry(
			&batch->entries[i], stream);
		if (!iree_status_is_ok(status)) {
			// Best-effort end the group before returning the error.
			IREE_NCCL_IGNORE_ERROR_NEW(nccl_symbols, ncclGroupEnd());
			IREE_TRACE_ZONE_END(z0);
			return status;
		}
	}
	IREE_NCCL_RETURN_AND_END_ZONE_IF_ERROR_NEW(
		z0, nccl_symbols, ncclGroupEnd(), "ncclGroupEnd");

	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Channel vtable
//===----------------------------------------------------------------------===//

static const iree_hal_channel_vtable_t
	iree_hal_cuda_new_nccl_channel_vtable = {
		.destroy = iree_hal_cuda_new_nccl_channel_destroy,
		.split = iree_hal_cuda_new_nccl_channel_split,
		.query_rank_and_count =
			iree_hal_cuda_new_nccl_channel_query_rank_and_count,
};
