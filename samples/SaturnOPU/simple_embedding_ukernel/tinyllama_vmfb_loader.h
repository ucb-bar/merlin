// Forward declaration for TinyLlama VMFB binary blob loader.
// Replaces the auto-generated iree-c-embed-data header.
#ifndef TINYLLAMA_VMFB_LOADER_H
#define TINYLLAMA_VMFB_LOADER_H

#include <stddef.h>
#include <stdint.h>

struct iree_file_toc_t {
	const char *name;
	const uint8_t *data;
	size_t size;
};

extern const struct iree_file_toc_t *iree_samples_model_tinyllama_opu_create(
	void);

#endif
