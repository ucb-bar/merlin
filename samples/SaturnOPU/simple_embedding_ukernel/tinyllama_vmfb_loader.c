// Loader for TinyLlama VMFB linked as a binary blob via .incbin assembly.
// Provides the same interface as iree-c-embed-data generated files.

#include <stddef.h>
#include <stdint.h>

// Symbols from tinyllama_vmfb_embed.S (.incbin)
extern const uint8_t _binary_tinyllama_vmfb_start[];
extern const uint8_t _binary_tinyllama_vmfb_end[];

struct iree_file_toc_t {
	const char *name;
	const uint8_t *data;
	size_t size;
};

const struct iree_file_toc_t *iree_samples_model_tinyllama_opu_create(void) {
	static struct iree_file_toc_t toc;
	toc.name = "tinyllama_opu";
	toc.data = _binary_tinyllama_vmfb_start;
	toc.size =
		(size_t)(_binary_tinyllama_vmfb_end - _binary_tinyllama_vmfb_start);
	return &toc;
}
