// RVV variant loader. Mirrors tinyllama_vmfb_loader.c but for the RVV blob.

#include <stddef.h>
#include <stdint.h>

extern const uint8_t _binary_tinyllama_rvv_vmfb_start[];
extern const uint8_t _binary_tinyllama_rvv_vmfb_end[];

struct iree_file_toc_t {
	const char *name;
	const uint8_t *data;
	size_t size;
};

const struct iree_file_toc_t *iree_samples_model_tinyllama_rvv_create(void) {
	static struct iree_file_toc_t toc;
	toc.name = "tinyllama_rvv";
	toc.data = _binary_tinyllama_rvv_vmfb_start;
	toc.size = (size_t)(_binary_tinyllama_rvv_vmfb_end -
		_binary_tinyllama_rvv_vmfb_start);
	return &toc;
}
