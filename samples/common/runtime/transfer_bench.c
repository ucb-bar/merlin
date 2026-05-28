// transfer_bench: measures the cost of moving a buffer between two
// HAL devices in a single process. This is what the heterogeneous
// scheduler pays at every chunk boundary that crosses device affinity.
//
// On UMA architectures like the QRB5165 (CPU + Adreno GPU + Hexagon NPU
// share DRAM), 'transfer' is dominated by:
//   1. iree_hal_allocator_allocate_buffer on the destination device
//   2. iree_hal_buffer_map_range to read the source side
//   3. memcpy under the hood
//   4. iree_hal_buffer_unmap_range + cache flush
//
// We measure this directly by allocating on dev_a, filling it, then
// allocating on dev_b and copying. The harness times only the copy +
// bookkeeping, not the initial fill.
//
// Usage:
//   transfer_bench --src-device=local-task --dst-device=qnn://gpu \
//                  [--bytes=4096,16384,65536,262144,1048576,4194304] \
//                  [--iterations=200] [--warmup=20]

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/runtime/api.h"

static const char *opt_src_device = "local-task";
static const char *opt_dst_device = "qnn://gpu";
static const char *opt_bytes_csv =
	"4096,16384,65536,262144,1048576,4194304,16777216";
static int opt_iterations = 200;
static int opt_warmup = 20;

static int parse_args(int argc, char **argv) {
	for (int i = 1; i < argc; ++i) {
		const char *a = argv[i];
		if (strncmp(a, "--src-device=", 13) == 0)
			opt_src_device = a + 13;
		else if (strncmp(a, "--dst-device=", 13) == 0)
			opt_dst_device = a + 13;
		else if (strncmp(a, "--bytes=", 8) == 0)
			opt_bytes_csv = a + 8;
		else if (strncmp(a, "--iterations=", 13) == 0)
			opt_iterations = atoi(a + 13);
		else if (strncmp(a, "--warmup=", 9) == 0)
			opt_warmup = atoi(a + 9);
		else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
			fprintf(stderr,
				"Usage: %s --src-device=<uri> --dst-device=<uri>\n"
				"          [--bytes=N1,N2,...] [--iterations=N] [--warmup=N]\n",
				argv[0]);
			return 1;
		} else {
			fprintf(stderr, "Unknown arg: %s\n", a);
			return 1;
		}
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

static iree_status_t make_device(iree_runtime_instance_t *instance,
	const char *uri, iree_hal_device_t **out_device) {
	iree_string_view_t uri_view = iree_make_cstring_view(uri);
	iree_string_view_t driver_name, rest;
	iree_string_view_split(uri_view, ':', &driver_name, &rest);
	if (iree_string_view_is_empty(rest)) {
		return iree_runtime_instance_try_create_default_device(
			instance, uri_view, out_device);
	}
	iree_string_view_t device_path = rest;
	if (device_path.size >= 2 && device_path.data[0] == '/' &&
		device_path.data[1] == '/') {
		device_path =
			iree_make_string_view(device_path.data + 2, device_path.size - 2);
	}
	iree_hal_driver_registry_t *reg =
		iree_runtime_instance_driver_registry(instance);
	iree_hal_driver_t *driver = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create(
		reg, driver_name, iree_allocator_system(), &driver));
	iree_async_proactor_pool_t *pp = NULL;
	iree_status_t s = iree_async_proactor_pool_create(iree_numa_node_count(),
		NULL, iree_async_proactor_pool_options_default(),
		iree_allocator_system(), &pp);
	if (iree_status_is_ok(s)) {
		iree_hal_device_create_params_t cp =
			iree_hal_device_create_params_default();
		cp.proactor_pool = pp;
		s = iree_hal_driver_create_device_by_path(driver, driver_name,
			device_path, 0, NULL, &cp, iree_allocator_system(), out_device);
	}
	iree_async_proactor_pool_release(pp);
	iree_hal_driver_release(driver);
	return s;
}

// Time one round trip: allocate src on dev_src, fill with pattern,
// allocate dst on dev_dst, copy via map/unmap, return total ns.
static iree_status_t measure_one(iree_hal_device_t *dev_src,
	iree_hal_device_t *dev_dst, size_t bytes, int64_t *samples, int n) {
	iree_hal_buffer_params_t params = {
		.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
			IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
		.access = IREE_HAL_MEMORY_ACCESS_ALL,
		.usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
		.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
	};
	iree_hal_buffer_t *src = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
		iree_hal_device_allocator(dev_src), params, bytes, &src));
	// One-time fill.
	iree_hal_buffer_mapping_t m_src;
	IREE_RETURN_IF_ERROR(
		iree_hal_buffer_map_range(src, IREE_HAL_MAPPING_MODE_PERSISTENT,
			IREE_HAL_MEMORY_ACCESS_WRITE, 0, bytes, &m_src));
	memset(m_src.contents.data, 0xa5, bytes);
	iree_hal_buffer_unmap_range(&m_src);

	for (int i = 0; i < n; ++i) {
		int64_t t0 = now_ns();
		// Allocate destination buffer on dev_dst.
		iree_hal_buffer_t *dst = NULL;
		iree_status_t s = iree_hal_allocator_allocate_buffer(
			iree_hal_device_allocator(dev_dst), params, bytes, &dst);
		if (!iree_status_is_ok(s)) {
			iree_hal_buffer_release(src);
			return s;
		}
		// Map src for read, dst for write, memcpy, unmap both.
		iree_hal_buffer_mapping_t mr, mw;
		s = iree_hal_buffer_map_range(src, IREE_HAL_MAPPING_MODE_PERSISTENT,
			IREE_HAL_MEMORY_ACCESS_READ, 0, bytes, &mr);
		if (iree_status_is_ok(s)) {
			s = iree_hal_buffer_map_range(dst, IREE_HAL_MAPPING_MODE_PERSISTENT,
				IREE_HAL_MEMORY_ACCESS_WRITE, 0, bytes, &mw);
			if (iree_status_is_ok(s)) {
				memcpy(mw.contents.data, mr.contents.data, bytes);
				iree_hal_buffer_unmap_range(&mw);
			}
			iree_hal_buffer_unmap_range(&mr);
		}
		iree_hal_buffer_release(dst);
		int64_t t1 = now_ns();
		if (!iree_status_is_ok(s)) {
			iree_hal_buffer_release(src);
			return s;
		}
		samples[i] = t1 - t0;
	}
	iree_hal_buffer_release(src);
	return iree_ok_status();
}

int main(int argc, char **argv) {
	if (parse_args(argc, argv))
		return 1;
	iree_runtime_instance_t *instance = NULL;
	iree_runtime_instance_options_t opts;
	iree_runtime_instance_options_initialize(&opts);
	iree_runtime_instance_options_use_all_available_drivers(&opts);
	iree_status_t s =
		iree_runtime_instance_create(&opts, iree_allocator_system(), &instance);
	if (!iree_status_is_ok(s))
		goto done;

	iree_hal_device_t *dev_src = NULL;
	iree_hal_device_t *dev_dst = NULL;
	s = make_device(instance, opt_src_device, &dev_src);
	if (!iree_status_is_ok(s))
		goto done;
	s = make_device(instance, opt_dst_device, &dev_dst);
	if (!iree_status_is_ok(s))
		goto done;

	printf("# bytes,iters,setup_us,mean_us,median_us,p99_us,min_us,max_us,"
		   "mb_per_s\n");
	// Parse bytes_csv.
	char buf[2048];
	strncpy(buf, opt_bytes_csv, sizeof(buf) - 1);
	buf[sizeof(buf) - 1] = '\0';
	char *save = NULL;
	for (char *tok = strtok_r(buf, ",", &save); tok;
		 tok = strtok_r(NULL, ",", &save)) {
		size_t n = (size_t)atoll(tok);
		if (!n)
			continue;
		int64_t *warmup_samples =
			(int64_t *)calloc((size_t)opt_warmup, sizeof(int64_t));
		int64_t *samples =
			(int64_t *)calloc((size_t)opt_iterations, sizeof(int64_t));
		int64_t t0 = now_ns();
		s = measure_one(dev_src, dev_dst, n, warmup_samples, opt_warmup);
		int64_t t1 = now_ns();
		double setup_us = (double)(t1 - t0) / 1e3;
		if (iree_status_is_ok(s)) {
			s = measure_one(dev_src, dev_dst, n, samples, opt_iterations);
		}
		if (!iree_status_is_ok(s)) {
			iree_status_fprint(stderr, s);
			iree_status_ignore(s);
			free(warmup_samples);
			free(samples);
			break;
		}
		qsort(samples, (size_t)opt_iterations, sizeof(int64_t), compare_int64);
		int64_t total = 0;
		for (int i = 0; i < opt_iterations; ++i)
			total += samples[i];
		double mean_us = (double)total / opt_iterations / 1e3;
		double med_us = (double)samples[opt_iterations / 2] / 1e3;
		double p99_us = (double)samples[(opt_iterations * 99) / 100] / 1e3;
		double min_us = (double)samples[0] / 1e3;
		double max_us = (double)samples[opt_iterations - 1] / 1e3;
		double mb_per_s = (n / 1e6) / (med_us / 1e6);
		printf("%zu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f\n", n, opt_iterations,
			setup_us, mean_us, med_us, p99_us, min_us, max_us, mb_per_s);
		fflush(stdout);
		free(warmup_samples);
		free(samples);
	}

done:
	if (!iree_status_is_ok(s)) {
		iree_status_fprint(stderr, s);
		iree_status_ignore(s);
	}
	iree_hal_device_release(dev_dst);
	iree_hal_device_release(dev_src);
	iree_runtime_instance_release(instance);
	return iree_status_is_ok(s) ? 0 : 1;
}
