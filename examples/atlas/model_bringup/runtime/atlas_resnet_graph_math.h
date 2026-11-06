#ifndef ATLAS_RESNET_GRAPH_MATH_H
#define ATLAS_RESNET_GRAPH_MATH_H

#include <stddef.h>
#include <stdint.h>

int atlas_graph_pack(const float *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, uint8_t *, size_t, int *);
int atlas_graph_decode(const uint8_t *, size_t, float *, unsigned, unsigned, unsigned);
int atlas_graph_affine(const float *, float *, const float *, const float *, unsigned, size_t);
int atlas_graph_relu(const float *, float *, size_t);
int atlas_graph_add(const float *, const float *, float *, size_t);

#endif
