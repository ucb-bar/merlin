#ifndef SMOL_NATIVE_DEVICE_H
#define SMOL_NATIVE_DEVICE_H
#include <stddef.h>
#include <stdint.h>

typedef int (*smol_native_reply)(uint32_t, const uint32_t *, size_t, uint32_t, void *);
int smol_native_configure(uintptr_t, uintptr_t, size_t, smol_native_reply, void *);
int smol_native_map(uintptr_t, size_t);
int smol_native_unmap(void);
uint32_t smol_native_starts(void);
uint32_t smol_native_error(void);
const char *smol_native_output(void);
size_t smol_native_output_bytes(void);
uintptr_t smol_native_imem_base(void);
uintptr_t smol_native_csr_base(void);
size_t smol_native_imem_bytes(void);
void smol_native_putchar(char);
#endif
