/* Generic retained-object chain driver. chain_data.h is derived from typed
 * captures/models and selected input/reference bytes, never from model names.
 * Counter windows bracket only the actual forward call, excluding transfers,
 * comparisons and output transport. All declared steps run even on mismatch. */
#include "merlin_model.h"
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__ || FLT_RADIX != 2 ||             \
	FLT_MANT_DIG != 24 || DBL_MANT_DIG != 53
#error                                                                         \
	"chain comparison requires the selected little-endian IEEE binary32/binary64 ABI"
#endif
_Static_assert(
	sizeof(float) == 4 && sizeof(double) == 8, "selected IEEE storage ABI");

typedef struct {
	int rank, width, kind;
	long dims[MERLIN_MAX_RANK];
	uint64_t elements;
} chain_spec;
typedef struct {
	const merlin_arg_t *args;
	int n_args, n_outputs;
	void *const *inputs;
	void *const *outputs;
	const void *weights;
	uint64_t weight_bytes;
	merlin_invoke_fn_t invoke;
	const chain_spec *specs;
	void (*profile_reset)(void);
	void (*profile_dump)(uint64_t, uint64_t);
} chain_program;
typedef struct {
	int argument, source_program, source_output;
	uint64_t offset, bytes;
} chain_transfer;
typedef struct {
	int program, step, n_transfers;
	const chain_transfer *transfers;
	const uint64_t *references;
} chain_step;

void console_init(void);
void htif_puts(const char *);
void htif_putc(char);
void htif_line_flush(int);
void htif_exit(int);
void merlin_arena_reset(void);
unsigned long long merlin_memref_rank_mismatches(void);
#include "chain_data.h"

static void decimal(uint64_t n) {
	char text[24];
	int i = 0;
	do {
		text[i++] = (char)('0' + n % 10);
		n /= 10;
	} while (n);
	while (i)
		htif_putc(text[--i]);
}
static void field(uint64_t n) {
	htif_putc(' ');
	decimal(n);
}
static void hex(uint64_t n) {
	for (int shift = 60; shift >= 0; shift -= 4)
		htif_putc("0123456789abcdef"[(n >> shift) & 15]);
}
static uint64_t bits(const void *p, int width) {
	/* Explicit constant-size builtins still become unaligned-safe scalar loads
	 * under the retained freestanding/-fno-builtin compilation policy. */
	switch (width) {
	case 1:
		return *(const unsigned char *)p;
	case 2: {
		uint16_t n;
		__builtin_memcpy(&n, p, sizeof(n));
		return n;
	}
	case 4: {
		uint32_t n;
		__builtin_memcpy(&n, p, sizeof(n));
		return n;
	}
	case 8: {
		uint64_t n;
		__builtin_memcpy(&n, p, sizeof(n));
		return n;
	}
	default: {
		uint64_t n = 0;
		memcpy(&n, p, (size_t)width);
		return n;
	}
	}
}
static uint64_t hash(const unsigned char *p, uint64_t n) {
	uint64_t h = UINT64_C(0xcbf29ce484222325);
	for (uint64_t i = 0; i < n; ++i)
		h = (h ^ p[i]) * UINT64_C(0x100000001b3);
	return h;
}
static uint64_t extend_signed(uint64_t word, int width) {
	if (width < 8 && (word & (UINT64_C(1) << (width * 8 - 1))))
		word |= UINT64_MAX << (width * 8);
	return word;
}
static double integer_delta(uint64_t a, uint64_t b, int width) {
	uint64_t ua = extend_signed(a, width), ub = extend_signed(b, width);
	int64_t sa, sb;
	memcpy(&sa, &ua, 8);
	memcpy(&sb, &ub, 8);
	/* Subtract as unsigned only after signed ordering: even INT64_MIN versus
	 * INT64_MAX has an exact representable uint64 magnitude before conversion.
	 */
	return (double)(sa >= sb ? ua - ub : ub - ua);
}
static uint64_t extent(const merlin_arg_t *a) {
	if (a->rank < 0 || a->rank > MERLIN_MAX_RANK || a->elem_size <= 0 ||
		a->elem_size > 8)
		return 0;
	uint64_t n = (uint64_t)a->elem_size;
	for (int i = 0; i < a->rank; ++i) {
		if (a->dims[i] <= 0 || n > UINT64_MAX / (uint64_t)a->dims[i])
			return 0;
		n *= (uint64_t)a->dims[i];
	}
	return n;
}
static int validate(void) {
	for (int p = 0; p < CHAIN_PROGRAMS; ++p) {
		const chain_program *program = &PROGRAMS[p];
		int outputs = 0;
		if (program->n_args <= 0 || program->n_args > 4096)
			return 0;
		for (int a = 0; a < program->n_args; ++a) {
			const merlin_arg_t *arg = &program->args[a];
			uint64_t n = extent(arg);
			if (!n)
				return 0;
			if (arg->kind == MERLIN_WEIGHT) {
				if (arg->offset < 0 ||
					(uint64_t)arg->offset > program->weight_bytes ||
					n > program->weight_bytes - arg->offset)
					return 0;
			} else if (arg->kind == MERLIN_INPUT) {
				if (!program->inputs[a])
					return 0;
			} else if (arg->kind == MERLIN_OUTPUT) {
				if (outputs >= program->n_outputs || !program->outputs[outputs])
					return 0;
				const chain_spec *s = &program->specs[outputs++];
				if (arg->rank != s->rank || arg->elem_size != s->width ||
					n != s->elements * s->width)
					return 0;
				for (int d = 0; d < s->rank; ++d)
					if (arg->dims[d] != s->dims[d])
						return 0;
			} else
				return 0;
		}
		if (outputs != program->n_outputs)
			return 0;
	}
	return 1;
}
/* kind: 0=integer, 1=f32, 2=f64, 3=bf16, 4=f16, 5=canonical bool. */
static double real_value(uint64_t b, int kind, int *finite) {
	if (kind == 1 || kind == 3) {
		uint32_t word = kind == 3 ? (uint32_t)b << 16 : (uint32_t)b;
		float f;
		if ((word & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
			*finite = 0;
		__builtin_memcpy(&f, &word, sizeof(f));
		return (double)f;
	}
	if (kind == 2) {
		double d;
		if ((b & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000))
			*finite = 0;
		__builtin_memcpy(&d, &b, sizeof(d));
		return d;
	}
	unsigned sign = (unsigned)b >> 15, exponent = ((unsigned)b >> 10) & 31,
			 fraction = (unsigned)b & 1023;
	if (exponent == 31) {
		*finite = 0;
		return 0.0;
	}
	double value = exponent ? (double)(1024 + fraction) : (double)fraction;
	int power = exponent ? (int)exponent - 25 : -24;
	while (power > 0) {
		value *= 2.0;
		--power;
	}
	while (power < 0) {
		value *= 0.5;
		++power;
	}
	return sign ? -value : value;
}
static uint64_t compare_array(
	int p, int step, int output, uint64_t reference_offset) {
	const chain_program *program = &PROGRAMS[p];
	const chain_spec *s = &program->specs[output];
	const unsigned char *actual = program->outputs[output],
						*expected = CHAIN_REFERENCES + reference_offset;
	uint64_t mismatched = 0, sample_index[CHAIN_SAMPLES],
			 sample_a[CHAIN_SAMPLES], sample_b[CHAIN_SAMPLES];
	int samples = 0, all_finite = 1;
	double maximum = 0.0, square_sum = 0.0;
	for (uint64_t i = 0; i < s->elements; ++i) {
		uint64_t a = bits(actual + i * s->width, s->width),
				 b = bits(expected + i * s->width, s->width);
		int matching = a == b;
		if (s->kind >= 1 && s->kind <= 4) {
			int finite = 1;
			double av = real_value(a, s->kind, &finite),
				   bv = real_value(b, s->kind, &finite);
			double delta = av - bv;
			if (delta < 0.0)
				delta = -delta;
			double magnitude = bv < 0.0 ? -bv : bv;
			matching = finite && delta <= CHAIN_ATOL + CHAIN_RTOL * magnitude;
			if (!finite)
				all_finite = 0;
			else {
				if (delta > maximum)
					maximum = delta;
				square_sum += delta * delta;
			}
		} else {
			if (s->kind == 5)
				matching = matching && a <= 1 && b <= 1;
			double delta = integer_delta(a, b, s->width);
			if (delta > maximum)
				maximum = delta;
			square_sum += delta * delta;
		}
		if (!matching) {
			++mismatched;
			if (samples < CHAIN_SAMPLES) {
				sample_index[samples] = i;
				sample_a[samples] = a;
				sample_b[samples++] = b;
			}
		}
	}
	uint64_t maximum_bits, square_bits;
	memcpy(&maximum_bits, &maximum, 8);
	memcpy(&square_bits, &square_sum, 8);
	htif_puts("CHAIN_ARRAY v1");
	field(p);
	field(step);
	field(output);
	field(s->elements);
	field(mismatched);
	field(all_finite);
	htif_putc(' ');
	hex(maximum_bits);
	htif_putc(' ');
	hex(square_bits);
	htif_putc(' ');
	hex(hash(actual, s->elements * s->width));
	htif_putc(' ');
	hex(hash(expected, s->elements * s->width));
	field(samples);
	for (int i = 0; i < samples; ++i) {
		field(sample_index[i]);
		htif_putc(' ');
		hex(sample_a[i]);
		htif_putc(' ');
		hex(sample_b[i]);
	}
	htif_putc('\n');
	return mismatched;
}
static void full_array(int p, int step, int o) {
	const chain_spec *s = &PROGRAMS[p].specs[o];
	const unsigned char *data = PROGRAMS[p].outputs[o];
	uint64_t n = s->elements * s->width;
	htif_puts("OUT_BIN_BEGIN v1 p");
	decimal(p);
	htif_puts("_s");
	decimal(step);
	htif_puts("_out");
	decimal(o);
	uint64_t columns = s->rank ? (uint64_t)s->dims[s->rank - 1] : 1;
	field(s->elements / columns);
	field(columns);
	field(s->width);
	htif_putc(' ');
	htif_putc(s->kind == 0 ? 's' : 'u');
	field(n);
	htif_putc('\n');
	htif_line_flush(0);
	for (uint64_t i = 0; i < n; ++i)
		htif_putc((char)data[i]);
	htif_line_flush(1);
	htif_puts("OUT_BIN_END v1 ");
	hex(hash(data, n));
	htif_putc('\n');
}
/* Satisfy the retained runtime's legacy single-entry compatibility wrapper. */
void merlin_invoke(void **descs) {
	PROGRAMS[0].invoke(descs);
}
static merlin_invoke_fn_t active_invoke;
static uint64_t invoke_begin, invoke_end;
static unsigned invoke_calls;
static void timed_invoke(void **descs) {
	++invoke_calls;
	invoke_begin = chain_counter_now();
	active_invoke(descs);
	invoke_end = chain_counter_now();
}

static int run_chain(void) {
	static merlin_descriptor_t descriptors[CHAIN_MAX_ARGS];
	uint64_t arrays = 0, mismatches = 0, runtime_errors = 0;
	if (!validate()) {
		htif_puts("CHAIN_ERROR v1 compiled_argument_ABI\n");
		return 2;
	}
	for (int i = 0; i < CHAIN_STEPS; ++i) {
		const chain_step *step = &STEPS[i];
		const chain_program *program = &PROGRAMS[step->program];
		for (int j = 0; j < step->n_transfers; ++j) {
			const chain_transfer *t = &step->transfers[j];
			if (program->args[t->argument].kind != MERLIN_INPUT ||
				extent(&program->args[t->argument]) != t->bytes)
				return 2;
			const void *source = t->source_program < 0
				? CHAIN_INPUTS + t->offset
				: PROGRAMS[t->source_program].outputs[t->source_output];
			memmove(program->inputs[t->argument], source, (size_t)t->bytes);
		}
		merlin_arena_reset();
		if (program->profile_reset)
			program->profile_reset();
		active_invoke = program->invoke;
		invoke_calls = 0;
		merlin_run_multi_with(program->args, program->n_args, program->weights,
			program->inputs, program->outputs, descriptors, timed_invoke);
		uint64_t step_errors = merlin_memref_rank_mismatches();
		if (invoke_calls != 1)
			step_errors |= 1;
		runtime_errors |= step_errors;
		htif_puts("CHAIN_STEP v1");
		field(step->program);
		field(step->step);
		field(invoke_end - invoke_begin);
		field(step_errors);
		htif_putc('\n');
		/* Profiling transport and reset are outside the invocation window.
		 * Reset per step also bounds recurrent programs' trace storage. */
		if (program->profile_dump) {
			htif_puts("CHAIN_PROFILE_BEGIN v1");
			field(step->program);
			field(step->step);
			htif_putc('\n');
			program->profile_dump(invoke_begin, invoke_end);
			htif_puts("CHAIN_PROFILE_END v1");
			field(step->program);
			field(step->step);
			htif_putc('\n');
		}
		for (int o = 0; o < program->n_outputs; ++o) {
			if (CHAIN_COMPACT)
				mismatches += compare_array(
					step->program, step->step, o, step->references[o]);
			else
				full_array(step->program, step->step, o);
			++arrays;
		}
	}
	htif_puts("CHAIN_END v1");
	field(CHAIN_STEPS);
	field(arrays);
	field(mismatches);
	htif_putc('\n');
	htif_puts("DONE\n");
	return runtime_errors ? 2 : (mismatches ? 1 : 0);
}
#ifdef MERLIN_CHAIN_HOST_FIXTURE
int main(void) {
	console_init();
	return run_chain();
}
#else
int main(int hart) {
	if (hart != 0)
		for (;;) {
		}
	console_init();
	int status = run_chain();
	htif_exit(status);
	return status;
}
#endif
