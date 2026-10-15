/* Bounded bare-metal model allocator.
 *
 * Bufferization emits a malloc/free pair for each temporary. A bump allocator
 * that ignores free needs the sum of every temporary in a whole model; the
 * declared activation arena instead budgets the peak live set. Each chunk has
 * a fixed 64-byte header followed by a 64-byte-aligned payload. Free chunks
 * stay address-sorted, are split on reuse and coalesced on release, so a later
 * large tensor can reuse space held by several smaller dead tensors. The arena
 * remains at an absolute address: its location may be beyond medany's reach
 * from model code and is part of the selected board's memory admission.
 *
 * Exhaustion and invalid frees fail loudly before generated code can use a
 * null or stale pointer. No memory outside MERLIN_ARENA_SIZE_BYTES is served.
 */
#include <stddef.h>
#include <stdint.h>

#include "htif.h"
#ifndef MERLIN_ARENA_BASE_ADDR
#define MERLIN_ARENA_BASE_ADDR 0xC0000000ULL
#endif
#ifndef MERLIN_ARENA_SIZE_BYTES
#define MERLIN_ARENA_SIZE_BYTES 0x10000000ULL
#endif
#define MAGIC 0x4d45524c494e414cULL
#define HEADER 64ULL
typedef struct Block {
	struct Block *next;
	size_t span;
	uint64_t magic;
	uint64_t live;
	size_t requested;
	size_t alignment;
	uint64_t reserved[2];
} Block;
_Static_assert(sizeof(Block) == HEADER, "fixed header size");
static uintptr_t brk;
static Block *free_list;
static unsigned long long seq, freed, reused, split, merged;
static size_t largest;
static uintptr_t arena_base(void) {
	return (uintptr_t)MERLIN_ARENA_BASE_ADDR;
}
static uintptr_t arena_end(void) {
	return arena_base() + (uintptr_t)MERLIN_ARENA_SIZE_BYTES;
}
static uintptr_t up(uintptr_t n, uintptr_t a) {
	return (n + a - 1) & ~(a - 1);
}
void merlin_arena_reset(void) {
	brk = arena_base();
	free_list = 0;
	seq = freed = reused = split = merged = 0;
	largest = 0;
}
static void fatal(const char *why, size_t n, uintptr_t align) {
	size_t total = 0, big = 0;
	for (Block *b = free_list; b; b = b->next) {
		total += b->span;
		if (b->span > big)
			big = b->span;
	}
	htif_puts("\nFATAL: model arena -- ");
	htif_puts(why);
	htif_puts("\n  requested ");
	htif_puthex(n);
	htif_puts(" align ");
	htif_putd((long)align);
	htif_puts("\n  arena ");
	htif_puthex(arena_base());
	htif_puts(" .. ");
	htif_puthex(arena_end());
	htif_puts(" brk ");
	htif_puthex(brk);
	htif_puts("\n  free span ");
	htif_puthex(total);
	htif_puts(" largest free ");
	htif_puthex(big);
	htif_puts(" largest request ");
	htif_puthex(largest);
	htif_puts("\n  allocations ");
	htif_putd((long)seq);
	htif_puts(" frees ");
	htif_putd((long)freed);
	htif_puts(" reused ");
	htif_putd((long)reused);
	htif_puts(" split ");
	htif_putd((long)split);
	htif_puts(" merged ");
	htif_putd((long)merged);
	htif_putc('\n');
	htif_exit(0x900);
}
static void *allocate(size_t n, uintptr_t align) {
	if (!brk)
		brk = arena_base();
	++seq;
	if (align < 64)
		align = 64;
	if ((align & (align - 1)) || align > MERLIN_ARENA_SIZE_BYTES)
		fatal("invalid alignment", n, align);
	if (MERLIN_ARENA_SIZE_BYTES <= HEADER + 64 ||
		n > MERLIN_ARENA_SIZE_BYTES - HEADER - 64)
		fatal("request exceeds whole arena", n, align);
	if (!n)
		n = 1;
	uintptr_t need = up(HEADER + n, 64);
	Block *best = 0, *best_prev = 0, *prev = 0;
	for (Block *b = free_list; b; prev = b, b = b->next)
		if (b->span >= need && (((uintptr_t)b + HEADER) & (align - 1)) == 0 &&
			(!best || b->span < best->span)) {
			best = b;
			best_prev = prev;
		}
	if (best) {
		uintptr_t remain = best->span - need;
		if (remain >= HEADER + 64) {
			Block *tail = (Block *)((uintptr_t)best + need);
			tail->next = best->next;
			tail->span = remain;
			tail->magic = MAGIC;
			tail->live = 0;
			tail->requested = 0;
			tail->alignment = 64;
			if (best_prev)
				best_prev->next = tail;
			else
				free_list = tail;
			best->span = need;
			++split;
		} else {
			if (best_prev)
				best_prev->next = best->next;
			else
				free_list = best->next;
		}
		best->next = 0;
		best->live = 1;
		best->requested = n;
		best->alignment = align;
		++reused;
		if (n > largest)
			largest = n;
		return (void *)((uintptr_t)best + HEADER);
	}
	uintptr_t start = up(brk + HEADER, align) - HEADER;
	if (start > arena_end() || need > arena_end() - start)
		fatal("exhausted", n, align);
	if (start > brk && start - brk >= HEADER + 64) {
		Block *gap = (Block *)brk;
		gap->span = start - brk;
		gap->magic = MAGIC;
		gap->live = 0;
		gap->requested = 0;
		gap->alignment = 64;
		Block **link = &free_list;
		while (*link && (uintptr_t)*link < (uintptr_t)gap)
			link = &(*link)->next;
		gap->next = *link;
		*link = gap;
	}
	Block *b = (Block *)start;
	brk = start + need;
	b->next = 0;
	b->span = need;
	b->magic = MAGIC;
	b->live = 1;
	b->requested = n;
	b->alignment = align;
	if (n > largest)
		largest = n;
	return (void *)(start + HEADER);
}
void *malloc(size_t n) {
	return allocate(n, 64);
}
void *aligned_alloc(size_t alignment, size_t size) {
	return allocate(size, alignment);
}
void *calloc(size_t nmemb, size_t size) {
	if (size && nmemb > SIZE_MAX / size)
		fatal("calloc size multiplication overflow", nmemb, size);
	size_t n = nmemb * size;
	unsigned char *p = allocate(n, 64);
	for (size_t i = 0; i < n; i++)
		p[i] = 0;
	return p;
}
void free(void *p) {
	if (!p)
		return;
	uintptr_t address = (uintptr_t)p;
	if (address < arena_base() + HEADER || address >= brk || (address & 63))
		fatal("invalid free pointer", 0, 0);
	Block *b = (Block *)(address - HEADER);
	if (b->magic != MAGIC || b->live != 1 || b->span < HEADER + 64 ||
		(b->span & 63) || b->span > brk - (uintptr_t)b)
		fatal("invalid or duplicate free", 0, 0);
	b->live = 0;
	++freed;
	Block *prev = 0, *next = free_list;
	while (next && (uintptr_t)next < (uintptr_t)b) {
		prev = next;
		next = next->next;
	}
	b->next = next;
	if (prev)
		prev->next = b;
	else
		free_list = b;
	if (prev && (uintptr_t)prev + prev->span == (uintptr_t)b) {
		prev->span += b->span;
		prev->next = b->next;
		b = prev;
		++merged;
	}
	next = b->next;
	if (next && (uintptr_t)b + b->span == (uintptr_t)next) {
		b->span += next->span;
		b->next = next->next;
		++merged;
	}
	if ((uintptr_t)b + b->span == brk) {
		if (b == free_list)
			free_list = b->next;
		else {
			prev = free_list;
			while (prev && prev->next != b)
				prev = prev->next;
			if (prev)
				prev->next = b->next;
		}
		brk = (uintptr_t)b;
	}
}
