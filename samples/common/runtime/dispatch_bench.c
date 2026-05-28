// dispatch_bench: minimal microbenchmark for measuring per-dispatch cost
// in the merlin runtime. Loads a VMFB + a HAL device once, then invokes
// the named function N times within a single process. Reports setup
// time + mean / median / p99 wall time per dispatch.
//
// This is the cross-toolchain-friendly substitute for upstream IREE's
// `iree-benchmark-module` which can't be built in our QRB5165 cross
// because google/benchmark hits the -nostdinc++ stdint.h issue. Same
// purpose, no third-party deps.
//
// Usage:
//   dispatch_bench --module=foo.vmfb --device=qnn://gpu --function=run
//                  [--zero-input=1x299x299x3xf32]
//                  [--input=1x299x299x3xf32=@input.bin]
//                  [--output-dump-dir=/tmp/out]
//                  --iterations=100 --warmup=5

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

static const char *opt_module = NULL;
static const char *opt_device = "local-task";
static const char *opt_function = "run";
#define MAX_INPUTS 16
typedef struct input_spec_t {
	const char *shape;
	const char *file_path;
	int zero_fill;
} input_spec_t;
static input_spec_t opt_inputs[MAX_INPUTS] = {0};
static int opt_input_count = 0;
static int opt_iterations = 100;
static int opt_warmup = 5;
static int opt_verbose = 0;
static const char *opt_output_dump_dir = NULL;

static void print_usage(const char *argv0) {
	fprintf(stderr,
		"Usage: %s --module=<vmfb> --device=<uri> --function=<name>\n"
		"          [--zero-input=<shape>]...\n"
		"          [--input=<shape>=@file.bin]...\n"
		"          [--output-dump-dir=<dir>] [--iterations=N] [--warmup=N]\n"
		"          [--verbose]\n"
		"\n"
		"When no input flags are given, the function must take zero inputs.\n"
		"Each input flag adds one buffer-view argument in order.\n"
		"Shape spec is e.g. 1x299x299x3xf32. Dtypes: f32, f16, i32, i8, ui8.\n"
		"\n"
		"Times one-time setup + mean/median/p99 wall over N invocations\n"
		"in a single process.\n",
		argv0);
}

static int parse_args(int argc, char **argv) {
	for (int i = 1; i < argc; ++i) {
		const char *a = argv[i];
		if (strncmp(a, "--module=", 9) == 0) {
			opt_module = a + 9;
		} else if (strncmp(a, "--device=", 9) == 0) {
			opt_device = a + 9;
		} else if (strncmp(a, "--function=", 11) == 0) {
			opt_function = a + 11;
		} else if (strncmp(a, "--zero-input=", 13) == 0) {
			if (opt_input_count >= MAX_INPUTS) {
				fprintf(stderr, "too many input flags (max %d)\n", MAX_INPUTS);
				return 1;
			}
			opt_inputs[opt_input_count++] = (input_spec_t){
				.shape = a + 13, .file_path = NULL, .zero_fill = 1};
		} else if (strncmp(a, "--input=", 8) == 0) {
			const char *spec = a + 8;
			const char *split = strstr(spec, "=@");
			size_t shape_len = 0;
			char *shape_copy = NULL;
			if (!split) {
				fprintf(stderr, "--input requires --input=<shape>=@file.bin\n");
				return 1;
			}
			if (opt_input_count >= MAX_INPUTS) {
				fprintf(stderr, "too many input flags (max %d)\n", MAX_INPUTS);
				return 1;
			}
			shape_len = (size_t)(split - spec);
			shape_copy = (char *)malloc(shape_len + 1);
			if (!shape_copy) {
				fprintf(stderr, "alloc failed for --input shape\n");
				return 1;
			}
			memcpy(shape_copy, spec, shape_len);
			shape_copy[shape_len] = '\0';
			opt_inputs[opt_input_count++] = (input_spec_t){
				.shape = shape_copy, .file_path = split + 2, .zero_fill = 0};
		} else if (strncmp(a, "--iterations=", 13) == 0) {
			opt_iterations = atoi(a + 13);
		} else if (strncmp(a, "--warmup=", 9) == 0) {
			opt_warmup = atoi(a + 9);
		} else if (strncmp(a, "--output-dump-dir=", 18) == 0) {
			opt_output_dump_dir = a + 18;
		} else if (strcmp(a, "--verbose") == 0) {
			opt_verbose = 1;
		} else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
			print_usage(argv[0]);
			return 1;
		} else {
			fprintf(stderr, "Unknown arg: %s\n", a);
			print_usage(argv[0]);
			return 1;
		}
	}
	if (!opt_module || !opt_function) {
		print_usage(argv[0]);
		return 1;
	}
	return 0;
}

static int64_t now_ns(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (int64_t)t.tv_sec * 1000000000LL + (int64_t)t.tv_nsec;
}

static int compare_int64(const void *a, const void *b) {
	int64_t va = *(const int64_t *)a;
	int64_t vb = *(const int64_t *)b;
	return (va > vb) - (va < vb);
}

static iree_status_t parse_shape_spec(const char *shape_spec,
	iree_hal_dim_t *shape, iree_host_size_t max_rank,
	iree_host_size_t *out_rank, iree_host_size_t *out_num_bytes,
	iree_hal_element_type_t *out_element_type) {
	iree_host_size_t shape_rank = 0;
	iree_host_size_t element_size = 0;
	iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
	const char *s = shape_spec;
	while (*s) {
		char *end = NULL;
		long v = strtol(s, &end, 10);
		if (end == s)
			break;
		if (shape_rank >= max_rank) {
			return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
				"shape rank exceeds %zu", (size_t)max_rank);
		}
		shape[shape_rank++] = (iree_hal_dim_t)v;
		s = end;
		if (*s == 'x') {
			++s;
		} else {
			break;
		}
	}
	if (strcmp(s, "f32") == 0) {
		element_size = 4;
		element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
	} else if (strcmp(s, "f16") == 0) {
		element_size = 2;
		element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
	} else if (strcmp(s, "i32") == 0) {
		element_size = 4;
		element_type = IREE_HAL_ELEMENT_TYPE_INT_32;
	} else if (strcmp(s, "i8") == 0) {
		element_size = 1;
		element_type = IREE_HAL_ELEMENT_TYPE_INT_8;
	} else if (strcmp(s, "ui8") == 0) {
		element_size = 1;
		element_type = IREE_HAL_ELEMENT_TYPE_UINT_8;
	} else {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"unknown dtype suffix '%s' in shape '%s'", s, shape_spec);
	}

	*out_num_bytes = element_size;
	for (iree_host_size_t i = 0; i < shape_rank; ++i) {
		*out_num_bytes *= (iree_host_size_t)shape[i];
	}
	*out_rank = shape_rank;
	*out_element_type = element_type;
	return iree_ok_status();
}

static iree_status_t read_exact_file(
	const char *path, uint8_t *dst, iree_host_size_t want_bytes) {
	FILE *f = fopen(path, "rb");
	size_t read_bytes = 0;
	long file_size = 0;
	if (!f) {
		return iree_make_status(IREE_STATUS_NOT_FOUND,
			"cannot open input file %s: %s", path, strerror(errno));
	}
	if (fseek(f, 0, SEEK_END) != 0) {
		fclose(f);
		return iree_make_status(
			IREE_STATUS_INTERNAL, "failed seeking input file %s", path);
	}
	file_size = ftell(f);
	if (file_size < 0) {
		fclose(f);
		return iree_make_status(
			IREE_STATUS_INTERNAL, "failed sizing input file %s", path);
	}
	rewind(f);
	if ((iree_host_size_t)file_size != want_bytes) {
		fclose(f);
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"input file %s has %ld bytes; expected %zu", path, file_size,
			(size_t)want_bytes);
	}
	read_bytes = fread(dst, 1, want_bytes, f);
	fclose(f);
	if (read_bytes != want_bytes) {
		return iree_make_status(
			IREE_STATUS_DATA_LOSS, "short read from %s", path);
	}
	return iree_ok_status();
}

static iree_status_t dump_buffer_view_to_file(
	iree_hal_device_t *device, iree_hal_buffer_view_t *view, const char *path) {
	iree_hal_buffer_t *buf = iree_hal_buffer_view_buffer(view);
	iree_device_size_t bytes = iree_hal_buffer_byte_length(buf);
	uint8_t *host = (uint8_t *)malloc((size_t)bytes);
	FILE *f = NULL;
	iree_status_t status = iree_ok_status();
	if (!host) {
		return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
			"allocating %" PRIu64 " bytes", (uint64_t)bytes);
	}
	status = iree_hal_device_transfer_d2h(device, buf, /*source_offset=*/0,
		host, bytes, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
		iree_infinite_timeout());
	if (!iree_status_is_ok(status)) {
		free(host);
		return status;
	}
	f = fopen(path, "wb");
	if (!f) {
		free(host);
		return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
			"cannot open %s for write: %s", path, strerror(errno));
	}
	if (fwrite(host, 1, (size_t)bytes, f) != (size_t)bytes) {
		status =
			iree_make_status(IREE_STATUS_DATA_LOSS, "short write to %s", path);
	}
	fclose(f);
	free(host);
	return status;
}

static iree_status_t make_device(iree_runtime_instance_t *instance,
	const char *uri, iree_hal_device_t **out_device) {
	// Try the simple driver path first ("qnn", "local-task", ...). Fall
	// back to driver://device-path syntax used by --device=qnn://gpu.
	iree_string_view_t uri_view = iree_make_cstring_view(uri);
	iree_string_view_t driver_name, rest;
	iree_string_view_split(uri_view, ':', &driver_name, &rest);
	if (iree_string_view_is_empty(rest)) {
		return iree_runtime_instance_try_create_default_device(
			instance, uri_view, out_device);
	}
	// Strip leading "//".
	iree_string_view_t device_path = rest;
	if (device_path.size >= 2 && device_path.data[0] == '/' &&
		device_path.data[1] == '/') {
		device_path =
			iree_make_string_view(device_path.data + 2, device_path.size - 2);
	}
	iree_hal_driver_registry_t *reg =
		iree_runtime_instance_driver_registry(instance);
	if (!reg) {
		return iree_make_status(
			IREE_STATUS_FAILED_PRECONDITION, "no driver registry on instance");
	}
	iree_hal_driver_t *driver = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create(
		reg, driver_name, iree_allocator_system(), &driver));

	// Backends like QNN assert on a non-NULL proactor_pool in their
	// device_create. Mirror what iree_runtime_instance_try_create_default
	// does: spin up a proactor pool here, attach to create_params, release
	// after device creation (the device retains its own ref).
	iree_async_proactor_pool_t *proactor_pool = NULL;
	iree_status_t s = iree_async_proactor_pool_create(iree_numa_node_count(),
		/*node_ids=*/NULL, iree_async_proactor_pool_options_default(),
		iree_allocator_system(), &proactor_pool);
	if (iree_status_is_ok(s)) {
		iree_hal_device_create_params_t cp =
			iree_hal_device_create_params_default();
		cp.proactor_pool = proactor_pool;
		s = iree_hal_driver_create_device_by_path(driver, driver_name,
			device_path,
			/*param_count=*/0, /*params=*/NULL, &cp, iree_allocator_system(),
			out_device);
	}
	iree_async_proactor_pool_release(proactor_pool);
	iree_hal_driver_release(driver);
	return s;
}

int main(int argc, char **argv) {
	if (parse_args(argc, argv))
		return 1;

	iree_status_t status = iree_ok_status();
	iree_runtime_instance_t *instance = NULL;
	iree_hal_device_t *device = NULL;
	iree_runtime_session_t *session = NULL;
	iree_runtime_call_t call;
	memset(&call, 0, sizeof(call));
	bool call_initialized = false;
	iree_hal_buffer_view_t *input_views[MAX_INPUTS] = {0};
	int input_view_count = 0;

	// ------- Phase A: one-time setup --------------------------------------
	const int64_t t_setup0 = now_ns();
	iree_runtime_instance_options_t opts;
	iree_runtime_instance_options_initialize(&opts);
	iree_runtime_instance_options_use_all_available_drivers(&opts);
	status =
		iree_runtime_instance_create(&opts, iree_allocator_system(), &instance);
	if (!iree_status_is_ok(status))
		goto cleanup;
	status = make_device(instance, opt_device, &device);
	if (!iree_status_is_ok(status))
		goto cleanup;

	iree_runtime_session_options_t sopts;
	iree_runtime_session_options_initialize(&sopts);
	status = iree_runtime_session_create_with_device(
		instance, &sopts, device, iree_allocator_system(), &session);
	if (!iree_status_is_ok(status))
		goto cleanup;
	status = iree_runtime_session_append_bytecode_module_from_file(
		session, opt_module);
	if (!iree_status_is_ok(status))
		goto cleanup;

	iree_vm_function_t function;
	status = iree_runtime_session_lookup_function(
		session, iree_make_cstring_view(opt_function), &function);
	if (!iree_status_is_ok(status))
		goto cleanup;

	status = iree_runtime_call_initialize(session, function, &call);
	if (!iree_status_is_ok(status))
		goto cleanup;
	call_initialized = true;

	// For each input flag, allocate one ordered buffer-view input from the
	// device allocator. Inputs may be zero-filled or file-backed.
	for (int input_idx = 0; input_idx < opt_input_count; ++input_idx) {
		const input_spec_t spec = opt_inputs[input_idx];
		enum { MAX_RANK = 8 };
		iree_hal_dim_t shape[MAX_RANK];
		iree_host_size_t shape_rank = 0;
		iree_host_size_t num_bytes = 0;
		iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
		iree_hal_buffer_view_t *input_view = NULL;
		status = parse_shape_spec(spec.shape, shape, MAX_RANK, &shape_rank,
			&num_bytes, &element_type);
		if (!iree_status_is_ok(status))
			goto cleanup;
		iree_hal_buffer_params_t buf_params = {
			.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
				IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
			.access = IREE_HAL_MEMORY_ACCESS_ALL,
			.usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
			.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
		};
		iree_hal_buffer_t *input_buf = NULL;
		status = iree_hal_allocator_allocate_buffer(
			iree_hal_device_allocator(device), buf_params, num_bytes,
			&input_buf);
		if (iree_status_is_ok(status)) {
			iree_hal_buffer_mapping_t mapping;
			status = iree_hal_buffer_map_range(input_buf,
				IREE_HAL_MAPPING_MODE_PERSISTENT, IREE_HAL_MEMORY_ACCESS_WRITE,
				/*offset=*/0, num_bytes, &mapping);
			if (iree_status_is_ok(status)) {
				if (spec.zero_fill) {
					memset(
						mapping.contents.data, 0, mapping.contents.data_length);
				} else {
					status = read_exact_file(spec.file_path,
						(uint8_t *)mapping.contents.data, num_bytes);
				}
				iree_hal_buffer_unmap_range(&mapping);
			}
		}
		if (iree_status_is_ok(status)) {
			status = iree_hal_buffer_view_create(input_buf, shape_rank, shape,
				element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
				iree_allocator_system(), &input_view);
		}
		iree_hal_buffer_release(input_buf);
		if (!iree_status_is_ok(status))
			goto cleanup;
		input_views[input_view_count++] = input_view;
	}
	if (!iree_status_is_ok(status))
		goto cleanup;

	const int64_t t_setup1 = now_ns();
	const double setup_ms = (double)(t_setup1 - t_setup0) / 1.0e6;

	// Push all inputs once at setup. iree_runtime_call_invoke documents
	// that "inputs list will remain unchanged" across invocations, so we
	// can reuse the same buffer views N times. Between invocations we just
	// clear the outputs list.
	for (int i = 0; i < input_view_count; ++i) {
		status = iree_runtime_call_inputs_push_back_buffer_view(
			&call, input_views[i]);
		if (!iree_status_is_ok(status))
			goto cleanup;
	}

	// ------- Phase B: warm-up + timed invocations --------------------------
	for (int i = 0; i < opt_warmup && iree_status_is_ok(status); ++i) {
		iree_vm_list_resize(call.outputs, 0);
		status = iree_runtime_call_invoke(&call, /*flags=*/0);
	}
	if (!iree_status_is_ok(status))
		goto cleanup;

	// After warmup: when the function takes inputs we expect it to also
	// produce outputs. Skip the check for void-returning functions (e.g.,
	// the per-dispatch wrappers from DumpExecutableDispatchModulesPass,
	// which take !hal.buffer args and return ()).
	if (input_view_count > 0) {
		iree_host_size_t out_count = iree_vm_list_size(call.outputs);
		if (out_count == 0 && opt_verbose) {
			fprintf(stderr,
				"note: function returned 0 outputs (in-place dispatch "
				"wrapper)\n");
		}
	}

	int64_t *samples =
		(int64_t *)calloc((size_t)opt_iterations, sizeof(int64_t));
	if (!samples) {
		status =
			iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "samples alloc");
		goto cleanup;
	}
	for (int i = 0; i < opt_iterations; ++i) {
		iree_vm_list_resize(call.outputs, 0);
		int64_t t0 = now_ns();
		status = iree_runtime_call_invoke(&call, /*flags=*/0);
		int64_t t1 = now_ns();
		if (!iree_status_is_ok(status))
			break;
		samples[i] = t1 - t0;
		if (opt_verbose) {
			fprintf(stderr, "  run %d: %.3f ms\n", i, samples[i] / 1.0e6);
		}
	}
	if (!iree_status_is_ok(status)) {
		free(samples);
		goto cleanup;
	}

	// ------- Phase C: stats ------------------------------------------------
	qsort(samples, (size_t)opt_iterations, sizeof(int64_t), compare_int64);
	int64_t total = 0;
	for (int i = 0; i < opt_iterations; ++i)
		total += samples[i];
	const double mean_ms = (double)total / opt_iterations / 1.0e6;
	const double med_ms = (double)samples[opt_iterations / 2] / 1.0e6;
	const int p99_idx = (opt_iterations * 99) / 100;
	const double p99_ms = (double)samples[p99_idx] / 1.0e6;
	const double min_ms = (double)samples[0] / 1.0e6;
	const double max_ms = (double)samples[opt_iterations - 1] / 1.0e6;
	printf("setup_ms=%.3f iters=%d warmup=%d "
		   "mean_ms=%.3f median_ms=%.3f p99_ms=%.3f "
		   "min_ms=%.3f max_ms=%.3f\n",
		setup_ms, opt_iterations, opt_warmup, mean_ms, med_ms, p99_ms, min_ms,
		max_ms);

	if (opt_output_dump_dir && iree_vm_list_size(call.outputs) > 0) {
		if (mkdir(opt_output_dump_dir, 0777) != 0 && errno != EEXIST) {
			status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
				"cannot create output dir %s: %s", opt_output_dump_dir,
				strerror(errno));
			free(samples);
			goto cleanup;
		}
		for (iree_host_size_t i = 0; i < iree_vm_list_size(call.outputs); ++i) {
			iree_hal_buffer_view_t *out_view =
				iree_vm_list_get_buffer_view_retain(call.outputs, i);
			char path[1024];
			if (!out_view)
				continue;
			snprintf(path, sizeof(path), "%s/output_%zu.bin",
				opt_output_dump_dir, (size_t)i);
			status = dump_buffer_view_to_file(device, out_view, path);
			iree_hal_buffer_view_release(out_view);
			if (!iree_status_is_ok(status)) {
				free(samples);
				goto cleanup;
			}
		}
	}
	free(samples);

cleanup:
	if (!iree_status_is_ok(status)) {
		iree_status_fprint(stderr, status);
		iree_status_ignore(status);
	}
	for (int i = 0; i < input_view_count; ++i) {
		iree_hal_buffer_view_release(input_views[i]);
	}
	for (int i = 0; i < opt_input_count; ++i) {
		if (!opt_inputs[i].zero_fill && opt_inputs[i].shape) {
			free((void *)opt_inputs[i].shape);
			opt_inputs[i].shape = NULL;
		}
	}
	if (call_initialized)
		iree_runtime_call_deinitialize(&call);
	iree_runtime_session_release(session);
	iree_hal_device_release(device);
	iree_runtime_instance_release(instance);
	return iree_status_is_ok(status) ? 0 : 1;
}
