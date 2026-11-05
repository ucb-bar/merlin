#ifndef SMOL_FULL_DRIVER_H
#define SMOL_FULL_DRIVER_H
#include "smol_policy_raw.h"
#include "smol_full_phase.h"
#ifndef SMOL_FULL_LAUNCH_COUNT
#define SMOL_FULL_LAUNCH_COUNT 2915
#endif
#ifndef SMOL_FULL_ATTENTION
#define SMOL_FULL_ATTENTION 1
#endif
#if (SMOL_FULL_ATTENTION != 0 && SMOL_FULL_ATTENTION != 1) || SMOL_FULL_LAUNCH_COUNT != 2715 + 200 * SMOL_FULL_ATTENTION
#error "inconsistent Smol execution profile"
#endif
typedef struct {
    uint32_t kind, ordinal, layer, launch_start, launch_end, call_start, call_end, reserved;
} smol_full_phase_descriptor;
typedef struct { unsigned char *actual; size_t bytes; const char *name; } smol_full_phase_region;
typedef struct {
    const float *images, *state, *noise;
    const uint8_t *image_masks, *language_mask;
    const int64_t *language_tokens;
    size_t vocabulary;
} smol_full_driver_inputs;
extern const smol_full_phase_descriptor smol_full_phase_plan[200];
extern const smol_full_phase_region smol_full_phase_regions[56];
extern const unsigned char *const smol_full_phase_expected[201][56];
extern const smol_full_driver_inputs smol_full_driver_observation;
extern smol_full_phase_cursor smol_full_phase_state;
extern volatile uint64_t smol_full_segment_request;
extern uint32_t smol_full_segment_observations[SMOL_FULL_LAUNCH_COUNT];
extern const uint64_t smol_full_compiled_phase_layout[12];
extern const char *smol_full_segment_error_region;
static inline int smol_full_driver_differ(const void *a,const void *b,size_t size) {
    const unsigned char *x=a,*y=b;
    for(size_t i=0;i<size;++i) if(x[i]!=y[i]) return 1;
    return 0;
}
int smol_full_driver_phase(unsigned index, int segmented);
int smol_full_driver_input(unsigned index);
int smol_full_driver_drained(void);
void smol_full_segment_ready(void);
void smol_full_segment_done(void);
void smol_full_segment_failed(void);
/* Native reference observers are null during guest execution. They never
 * supply projection values or arithmetic live state to the production driver. */
typedef int (*smol_full_driver_observer)(unsigned);
void smol_full_driver_observe(smol_full_driver_observer);
void smol_full_driver_observe_ready(void (*)(void));
int smol_full_driver_vision_launch(void *,unsigned,unsigned,const uint16_t *,size_t,size_t,uint16_t *,size_t);
int smol_full_driver_policy_launch(void *,unsigned,unsigned,unsigned,unsigned,const void *,size_t,size_t,
                                  unsigned,void *,size_t);
_Static_assert(sizeof(smol_full_phase_descriptor)==32,"phase plan ABI");
_Static_assert(sizeof(smol_full_phase_region)==24,"phase region ABI");
_Static_assert(sizeof(smol_full_driver_inputs)==56,"observation pointer ABI");
#endif
