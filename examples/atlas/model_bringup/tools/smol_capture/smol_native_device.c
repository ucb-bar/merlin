/* Synchronous captured replies for native host protocol checks, without device execution. */
#define _GNU_SOURCE
#include "smol_native_device.h"
#include "atlas_host.h"
#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>

static struct {
    uintptr_t imem_base, csr_base;
    size_t capacity, words, output_bytes;
    uint32_t imem[32768], starts, error, marker, sequence, status, counter;
    smol_native_reply reply;
    void *context, *arena;
    size_t arena_bytes;
    char output[262144];
} device;

int smol_native_configure(uintptr_t imem, uintptr_t csr, size_t capacity,
                          smol_native_reply reply, void *context) {
    if (!reply || !capacity || capacity > sizeof(device.imem) || capacity % 4 || imem % 4 || csr % 4)
        return EINVAL;
    device.imem_base = imem; device.csr_base = csr; device.capacity = capacity;
    device.reply = reply; device.context = context; device.status = 5;
    device.words = device.starts = device.error = device.marker = device.sequence = device.counter = 0;
    device.output_bytes = 0; device.output[0] = 0;
    return 0;
}

int smol_native_map(uintptr_t address, size_t bytes) {
    long page = sysconf(_SC_PAGESIZE);
    if (device.arena) return EBUSY;
    if (page <= 0 || !bytes || address % (uintptr_t)page || bytes > SIZE_MAX-(size_t)page)
        return EINVAL;
    size_t rounded = (bytes+(size_t)page-1)/(size_t)page*(size_t)page;
    void *mapped = mmap((void *)address, rounded, PROT_READ|PROT_WRITE,
                        MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE, -1, 0);
    if (mapped == MAP_FAILED) return errno;
    if (mapped != (void *)address) { munmap(mapped, rounded); return EFAULT; }
    device.arena = mapped; device.arena_bytes = rounded;
    return 0;
}

int smol_native_unmap(void) {
    if (!device.arena) return EINVAL;
    int result = munmap(device.arena, device.arena_bytes);
    device.arena = 0; device.arena_bytes = 0;
    return result ? errno : 0;
}

uint32_t __wrap_atlas_host_direct_read32(void *context, uintptr_t address) {
    (void)context;
    ++device.counter;
    if (address == device.csr_base+ATLAS_CSR_CYCLES) return device.counter;
    if (address == device.csr_base+ATLAS_CSR_STATUS) return device.status;
    if (address == device.csr_base+ATLAS_CSR_DBG0) return device.marker;
    if (address == device.csr_base+ATLAS_CSR_DBG1) return device.sequence;
    device.error = 1; return 0;
}

void __wrap_atlas_host_direct_write32(void *context, uintptr_t address, uint32_t word) {
    (void)context;
    ++device.counter;
    if (address >= device.imem_base && address-device.imem_base < device.capacity) {
        size_t index = (address-device.imem_base)/4;
        if (address % 4 || (index && index != device.words)) { device.error = 2; return; }
        if (!index) device.words = 0;
        device.imem[device.words++] = word; return;
    }
    if (address == device.csr_base+ATLAS_CSR_DBG0) { device.marker = word; return; }
    if (address == device.csr_base+ATLAS_CSR_DBG1) { device.sequence = word; return; }
    if (address == device.csr_base+ATLAS_CSR_EXEC_CONTROL && word == 1) {
        uint32_t index = device.starts++;
        int result = 1;
        if (!device.error && device.status == 5 && !device.marker && device.sequence == index+1)
            result = device.reply(index, device.imem, device.words, device.sequence, device.context);
        else device.error = 3;
        device.marker = device.sequence;
        device.status = result ? 3 : 5;
        return;
    }
    device.error = 4;
}

void smol_native_putchar(char value) {
    if (device.output_bytes+1 >= sizeof(device.output)) { device.error = 5; return; }
    device.output[device.output_bytes++] = value;
    device.output[device.output_bytes] = 0;
}
const char *smol_native_output(void) { return device.output; }
size_t smol_native_output_bytes(void) { return device.output_bytes; }
uint32_t smol_native_starts(void) { return device.starts; }
uint32_t smol_native_error(void) { return device.error; }
uintptr_t smol_native_imem_base(void) { return ATLAS_IMEM_BASE; }
uintptr_t smol_native_csr_base(void) { return ATLAS_CSR_BASE; }
size_t smol_native_imem_bytes(void) { return ATLAS_IMEM_BYTES; }
