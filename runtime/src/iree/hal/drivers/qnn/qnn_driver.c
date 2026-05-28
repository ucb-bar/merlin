// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// QNN runtime HAL driver. Wraps Qualcomm's QNN SDK so an IREE process can
// load pre-built QNN context binaries as `hal.executable.objects` and
// dispatch them via QnnGraph_execute. Runtime-only path: no IREE codegen
// for QNN.
//
// Status: structural scaffolding. The driver, device, executable, and
// command-buffer plumbing is in place and gated under
// MERLIN_RUNTIME_ENABLE_HAL_QNN. Real QNN integration (QnnSystem_loadInterface,
// QnnContext_createFromBinary, QnnGraph_execute, Qnn_Tensor_t binding) is
// implemented at the points marked TODO(qnn-integration); end-to-end
// validation requires on-board iteration on a QRB5165 + the QNN SDK at
// QNN_SDK_ROOT.

#include "qnn_driver.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#include "qnn_device.h"

#ifndef QNN_HAL_LIB_DIR
#define QNN_HAL_LIB_DIR ""
#endif

//===----------------------------------------------------------------------===//
// Backend descriptors
//===----------------------------------------------------------------------===//

typedef struct iree_hal_qnn_backend_descriptor_t {
	iree_hal_qnn_backend_t id;
	const char *name; // user-facing token: "cpu", "gpu", "htp", "dsp"
	const char *lib_name; // dlopen()'d at runtime: "libQnnHtp.so" etc.
} iree_hal_qnn_backend_descriptor_t;

static const iree_hal_qnn_backend_descriptor_t
	iree_hal_qnn_backend_descriptors_[] = {
		{IREE_HAL_QNN_BACKEND_CPU, "cpu", "libQnnCpu.so"},
		{IREE_HAL_QNN_BACKEND_GPU, "gpu", "libQnnGpu.so"},
		{IREE_HAL_QNN_BACKEND_HTP, "htp", "libQnnHtp.so"},
		{IREE_HAL_QNN_BACKEND_DSP, "dsp", "libQnnDsp.so"},
		{IREE_HAL_QNN_BACKEND_HTA, "hta", "libQnnHta.so"},
};

IREE_API_EXPORT void iree_hal_qnn_driver_options_initialize(
	iree_hal_qnn_driver_options_t *out_options) {
	memset(out_options, 0, sizeof(*out_options));
	// Default to "all backends enabled" so query_available_devices probes
	// each. Probe failures (missing .so on the target) are non-fatal.
	out_options->enabled_backends_mask = 0u;
}

//===----------------------------------------------------------------------===//
// iree_hal_qnn_driver_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_qnn_driver_t {
	iree_hal_resource_t resource;
	iree_string_view_t identifier;
	iree_allocator_t host_allocator;
	uint32_t enabled_backends_mask;
	// Per-backend dynamic-library handle (lazily populated on first use).
	iree_dynamic_library_t *backend_libs[IREE_HAL_QNN_BACKEND_COUNT];
	// libQnnSystem.so handle + interface pointer, lazily loaded on first
	// executable_create. Cached at driver level so subsequent
	// executable_creates don't repeat the dlopen + getProviders dance.
	iree_dynamic_library_t *system_lib;
	const void *system_interface; // const QnnSystemInterface_t*
	iree_slim_mutex_t system_init_mu;
	// Cached lib dir, used when dlopen()'ing per-backend libs.
	char lib_dir[256];
} iree_hal_qnn_driver_t;

extern const iree_hal_driver_vtable_t iree_hal_qnn_driver_vtable_;

static iree_hal_qnn_driver_t *iree_hal_qnn_driver_cast(
	iree_hal_driver_t *base) {
	IREE_HAL_ASSERT_TYPE(base, &iree_hal_qnn_driver_vtable_);
	return (iree_hal_qnn_driver_t *)base;
}

IREE_API_EXPORT iree_status_t iree_hal_qnn_driver_create(
	iree_string_view_t identifier, const iree_hal_qnn_driver_options_t *options,
	iree_allocator_t host_allocator, iree_hal_driver_t **out_driver) {
	IREE_ASSERT_ARGUMENT(options);
	IREE_ASSERT_ARGUMENT(out_driver);
	IREE_TRACE_ZONE_BEGIN(z0);
	*out_driver = NULL;

	iree_hal_qnn_driver_t *driver = NULL;
	iree_host_size_t identifier_offset = iree_sizeof_struct(*driver) +
		IREE_HAL_QNN_BACKEND_COUNT * sizeof(void *);
	iree_host_size_t total_size = identifier_offset + identifier.size;
	iree_status_t status =
		iree_allocator_malloc(host_allocator, total_size, (void **)&driver);
	if (!iree_status_is_ok(status)) {
		IREE_TRACE_ZONE_END(z0);
		return status;
	}
	iree_hal_resource_initialize(
		&iree_hal_qnn_driver_vtable_, &driver->resource);
	driver->host_allocator = host_allocator;
	driver->enabled_backends_mask = options->enabled_backends_mask
		? options->enabled_backends_mask
		: ((1u << IREE_HAL_QNN_BACKEND_COUNT) - 1u);
	iree_slim_mutex_initialize(&driver->system_init_mu);
	driver->system_lib = NULL;
	driver->system_interface = NULL;
	uint8_t *identifier_ptr = ((uint8_t *)driver) + identifier_offset;
	memcpy(identifier_ptr, identifier.data, identifier.size);
	driver->identifier =
		iree_make_string_view((const char *)identifier_ptr, identifier.size);
	// Pick up the build-time default for the lib dir; user override via
	// env var QNN_BACKEND_LIB_DIR takes precedence at runtime.
	const char *lib_env = getenv("QNN_BACKEND_LIB_DIR");
	const char *lib_path = (lib_env && lib_env[0]) ? lib_env : QNN_HAL_LIB_DIR;
	if (lib_path && lib_path[0]) {
		snprintf(driver->lib_dir, sizeof(driver->lib_dir), "%s", lib_path);
	}

	*out_driver = (iree_hal_driver_t *)driver;
	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static void iree_hal_qnn_driver_destroy(iree_hal_driver_t *base_driver) {
	iree_hal_qnn_driver_t *driver = iree_hal_qnn_driver_cast(base_driver);
	IREE_TRACE_ZONE_BEGIN(z0);
	for (int i = 0; i < IREE_HAL_QNN_BACKEND_COUNT; ++i) {
		if (driver->backend_libs[i]) {
			iree_dynamic_library_release(driver->backend_libs[i]);
		}
	}
	if (driver->system_lib) {
		iree_dynamic_library_release(driver->system_lib);
		driver->system_lib = NULL;
	}
	iree_slim_mutex_deinitialize(&driver->system_init_mu);
	iree_allocator_free(driver->host_allocator, driver);
	IREE_TRACE_ZONE_END(z0);
}

// Lazily loads libQnnSystem.so + binds QnnSystemInterface_t. Thread-safe;
// returns the same interface pointer for all callers. Failure leaves
// |*out_interface| == NULL and returns the underlying error so callers
// can fall back to a name-based lookup if libQnnSystem is missing.
iree_status_t iree_hal_qnn_driver_get_system_interface(
	iree_hal_driver_t *base_driver, const void **out_interface) {
	iree_hal_qnn_driver_t *driver = iree_hal_qnn_driver_cast(base_driver);
	iree_slim_mutex_lock(&driver->system_init_mu);
	iree_status_t status = iree_ok_status();
	if (driver->system_interface == NULL) {
		iree_dynamic_library_t *lib = NULL;
		status = iree_dynamic_library_load_from_file("libQnnSystem.so",
			IREE_DYNAMIC_LIBRARY_FLAG_NONE, driver->host_allocator, &lib);
		if (iree_status_is_ok(status)) {
			// Use uint64_t for the return because Qnn_ErrorHandle_t is
			// uint64_t; truncating to int silently mis-reads non-zero error
			// codes.
			typedef uint64_t (*get_providers_fn_t)(
				const void ***providers, uint32_t *num);
			get_providers_fn_t get_providers = NULL;
			status = iree_dynamic_library_lookup_symbol(lib,
				"QnnSystemInterface_getProviders", (void **)&get_providers);
			if (iree_status_is_ok(status) && get_providers) {
				const void **providers = NULL;
				uint32_t num = 0;
				uint64_t rc = get_providers(&providers, &num);
				if (rc == 0 && num > 0 && providers != NULL) {
					driver->system_lib = lib; // transfer ownership
					driver->system_interface = providers[0];
					lib = NULL;
				} else {
					status = iree_make_status(IREE_STATUS_INTERNAL,
						"QnnSystemInterface_getProviders rc=%llu num=%u",
						(unsigned long long)rc, num);
				}
			}
		}
		if (lib)
			iree_dynamic_library_release(lib);
	}
	*out_interface = driver->system_interface;
	iree_slim_mutex_unlock(&driver->system_init_mu);
	return status;
}

static iree_status_t iree_hal_qnn_driver_query_available_devices(
	iree_hal_driver_t *base_driver, iree_allocator_t host_allocator,
	iree_host_size_t *out_device_info_count,
	iree_hal_device_info_t **out_device_infos) {
	iree_hal_qnn_driver_t *driver = iree_hal_qnn_driver_cast(base_driver);

	// Probe each enabled backend's .so by trying to dlopen() it. A backend
	// is "available" only if the lib actually loads.
	iree_host_size_t available = 0;
	bool found[IREE_HAL_QNN_BACKEND_COUNT] = {0};
	for (int i = 0; i < IREE_HAL_QNN_BACKEND_COUNT; ++i) {
		if (!(driver->enabled_backends_mask & (1u << i)))
			continue;
		char path[512];
		snprintf(path, sizeof(path), "%s/%s", driver->lib_dir,
			iree_hal_qnn_backend_descriptors_[i].lib_name);
		iree_dynamic_library_t *lib = NULL;
		iree_status_t st = iree_dynamic_library_load_from_file(
			path, IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &lib);
		if (iree_status_is_ok(st)) {
			driver->backend_libs[i] = lib; // cache for create_device
			found[i] = true;
			++available;
		} else {
			iree_status_free(st);
		}
	}

	iree_host_size_t total_size = available * sizeof(iree_hal_device_info_t);
	for (int i = 0; i < IREE_HAL_QNN_BACKEND_COUNT; ++i) {
		if (!found[i])
			continue;
		total_size += iree_host_align(strlen("qnn-") + 4 + 1, iree_max_align_t);
	}
	iree_hal_device_info_t *infos = NULL;
	IREE_RETURN_IF_ERROR(
		iree_allocator_malloc(host_allocator, total_size, (void **)&infos));
	uint8_t *str_buf = (uint8_t *)infos + available * sizeof(*infos);
	iree_host_size_t idx = 0;
	for (int i = 0; i < IREE_HAL_QNN_BACKEND_COUNT; ++i) {
		if (!found[i])
			continue;
		char *name = (char *)str_buf;
		int n = snprintf(
			name, 32, "qnn-%s", iree_hal_qnn_backend_descriptors_[i].name);
		str_buf += iree_host_align(n + 1, iree_max_align_t);
		infos[idx].device_id = (iree_hal_device_id_t)i;
		infos[idx].name = iree_make_cstring_view(name);
		infos[idx].path = iree_string_view_empty();
		++idx;
	}
	*out_device_info_count = available;
	*out_device_infos = infos;
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_driver_dump_device_info(
	iree_hal_driver_t *base_driver, iree_hal_device_id_t device_id,
	iree_string_builder_t *builder) {
	if (device_id >= IREE_HAL_QNN_BACKEND_COUNT) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"QNN device id %" PRIu64 " out of range", (uint64_t)device_id);
	}
	return iree_string_builder_append_format(builder, "QNN backend: %s\n",
		iree_hal_qnn_backend_descriptors_[device_id].name);
}

// Probes the dlopen-able libQnn<Backend>.so on the host, calls
// QnnInterface_getProviders to confirm the SDK is functional, and emits a
// status describing what was found. This is the first half of full device
// creation: the second half (QnnContext_create + QnnDevice_create + the
// per-context resources iree_hal_qnn_device_t will own) lands once the
// command-buffer / executable wiring is filled in. Verified on QRB5165:
// libQnnCpu.so and libQnnHtp.so both return ret=0 num=1 from
// QnnInterface_getProviders.
typedef int (*iree_hal_qnn_get_providers_fn_t)(
	const void ***providers, uint32_t *num_providers);

static iree_status_t iree_hal_qnn_driver_create_device_by_id(
	iree_hal_driver_t *base_driver, iree_hal_device_id_t device_id,
	iree_host_size_t param_count, const iree_string_pair_t *params,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	iree_hal_qnn_driver_t *driver = iree_hal_qnn_driver_cast(base_driver);
	if (device_id >= IREE_HAL_QNN_BACKEND_COUNT) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"QNN device id %" PRIu64 " out of range", (uint64_t)device_id);
	}
	const iree_hal_qnn_backend_descriptor_t *desc =
		&iree_hal_qnn_backend_descriptors_[device_id];
	// dlopen libQnn<Backend>.so via IREE's dynamic-library loader so symbols
	// (and their dlclose ordering) follow IREE's lifetime model.
	iree_dynamic_library_t *lib = NULL;
	iree_status_t status = iree_dynamic_library_load_from_file(
		desc->lib_name, IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &lib);
	if (!iree_status_is_ok(status)) {
		return iree_status_annotate_f(status,
			"QNN backend '%s' library '%s' not loadable; ensure "
			"LD_LIBRARY_PATH includes <QNN_SDK>/lib/<arch>",
			desc->name, desc->lib_name);
	}
	driver->backend_libs[device_id] = lib;
	iree_hal_qnn_get_providers_fn_t get_providers = NULL;
	status = iree_dynamic_library_lookup_symbol(
		lib, "QnnInterface_getProviders", (void **)&get_providers);
	if (!iree_status_is_ok(status)) {
		iree_dynamic_library_release(lib);
		driver->backend_libs[device_id] = NULL;
		return iree_status_annotate(status,
			IREE_SV("QnnInterface_getProviders not exported by libQnn*.so; "
					"SDK version mismatch?"));
	}
	const void **providers = NULL;
	uint32_t num_providers = 0;
	int qnn_rc = get_providers(&providers, &num_providers);
	if (qnn_rc != 0 || num_providers == 0) {
		iree_dynamic_library_release(lib);
		driver->backend_libs[device_id] = NULL;
		return iree_make_status(IREE_STATUS_INTERNAL,
			"QnnInterface_getProviders returned rc=%d "
			"num_providers=%u for backend '%s'",
			qnn_rc, num_providers, desc->name);
	}
	// SDK probe succeeded — instantiate the device. The device retains its own
	// ref on the dlopen'd lib; the driver keeps its ref so subsequent device
	// creations on the same backend reuse the same load.
	char identifier_buf[64];
	int n =
		snprintf(identifier_buf, sizeof(identifier_buf), "qnn-%s", desc->name);
	iree_string_view_t identifier =
		iree_make_string_view(identifier_buf, (iree_host_size_t)n);
	return iree_hal_qnn_device_create(identifier,
		(iree_hal_qnn_backend_t)device_id, lib, base_driver, create_params,
		host_allocator, out_device);
}

static iree_status_t iree_hal_qnn_driver_create_device_by_path(
	iree_hal_driver_t *base_driver, iree_string_view_t driver_name,
	iree_string_view_t device_path, iree_host_size_t param_count,
	const iree_string_pair_t *params,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	// Map "qnn-htp" / "qnn-cpu" / etc to the corresponding device id.
	iree_hal_qnn_backend_t backend = IREE_HAL_QNN_BACKEND_HTP;
	if (iree_string_view_equal(device_path, IREE_SV("cpu"))) {
		backend = IREE_HAL_QNN_BACKEND_CPU;
	} else if (iree_string_view_equal(device_path, IREE_SV("gpu"))) {
		backend = IREE_HAL_QNN_BACKEND_GPU;
	} else if (iree_string_view_equal(device_path, IREE_SV("htp"))) {
		backend = IREE_HAL_QNN_BACKEND_HTP;
	} else if (iree_string_view_equal(device_path, IREE_SV("hta"))) {
		backend = IREE_HAL_QNN_BACKEND_HTA;
	} else if (iree_string_view_equal(device_path, IREE_SV("npu"))) {
		// "npu" — schedule-level alias. Map to whichever Hexagon backend
		// works on this chip (HTA on S865/QRB5165, HTP on newer parts). For
		// now route to HTA explicitly; future: probe both at create time.
		backend = IREE_HAL_QNN_BACKEND_HTA;
	} else if (iree_string_view_equal(device_path, IREE_SV("dsp"))) {
		backend = IREE_HAL_QNN_BACKEND_DSP;
	} else if (!iree_string_view_is_empty(device_path)) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"unknown QNN backend '%.*s'", (int)device_path.size,
			device_path.data);
	}
	return iree_hal_qnn_driver_create_device_by_id(base_driver, backend,
		param_count, params, create_params, host_allocator, out_device);
}

const iree_hal_driver_vtable_t iree_hal_qnn_driver_vtable_ = {
	.destroy = iree_hal_qnn_driver_destroy,
	.query_available_devices = iree_hal_qnn_driver_query_available_devices,
	.dump_device_info = iree_hal_qnn_driver_dump_device_info,
	.create_device_by_id = iree_hal_qnn_driver_create_device_by_id,
	.create_device_by_path = iree_hal_qnn_driver_create_device_by_path,
};
