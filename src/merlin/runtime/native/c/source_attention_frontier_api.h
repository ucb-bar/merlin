#ifndef MERLIN_SOURCE_ATTENTION_FRONTIER_API_H
#define MERLIN_SOURCE_ATTENTION_FRONTIER_API_H
#include <stdint.h>
#include <stddef.h>
/* Host semantic view, not a target memref calling convention. Offset/strides
 * are element counts. Compiler/caller owns allocation, lifetime, disjoint
 * writable destination and workspace proofs. */
typedef struct {void *data;int64_t offset,sizes[4],strides[4];} merlin_attention_view;
/* Complete signed-i8 grouped dot into every m*n i32 output. Implementations
 * must preserve host FENV, synchronize before return and have independently
 * proved radix pairs, overflow, storage and full-write contracts. */
typedef int (*merlin_attention_product)(void*,const int8_t*,const int8_t*,int32_t*,int,int,int,int);
#endif
