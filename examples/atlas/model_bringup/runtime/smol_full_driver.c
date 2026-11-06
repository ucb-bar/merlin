#include "smol_full_driver.h"
#include "smol_host_bits.h"
#include <stddef.h>
#include <string.h>
extern atlas_host smol_full_host;
extern smol_policy_raw_context smol_full_raw;
extern smol_policy_state smol_full_graph_state;
extern smol_vision_state smol_full_vision_states[2];
extern const smol_vision_parameters smol_full_vision_parameters;
extern const smol_vision_frame smol_full_vision_frame;
extern const smol_policy_parameters smol_full_policy_parameters;
extern const smol_policy_frame smol_full_policy_frame;
extern uint16_t smol_full_image[786432],smol_full_image_output[2][61440];
extern uint32_t smol_full_cycles[SMOL_FULL_LAUNCH_COUNT];
extern uint8_t smol_full_actual_prefix_pad_mask[177],smol_full_actual_prefix_attention_mask[177];
extern float *const smol_step_input_state[10],*const smol_step_output_state[10];
extern float *const smol_step_timestep[10],*const smol_step_velocity[10];
extern float smol_full_actual_final_full_state[1600],smol_full_actual_final_actions[300];
smol_full_phase_cursor smol_full_phase_state={1,0,0,0,0,0,0,0,0,0};
volatile uint64_t smol_full_segment_request=UINT64_MAX;
uint32_t smol_full_segment_observations[SMOL_FULL_LAUNCH_COUNT];
const char *smol_full_segment_error_region;
const uint64_t smol_full_compiled_phase_layout[12]={1,sizeof(smol_full_phase_cursor),
    offsetof(smol_full_phase_cursor,sequence),sizeof(smol_policy_state),sizeof(smol_vision_state),
    sizeof(smol_policy_raw_context),sizeof(smol_policy_panel),sizeof(smol_policy_call),sizeof(atlas_host),
    SMOL_FULL_LAUNCH_COUNT,1421,200};
static smol_full_driver_observer observer;
#ifndef __riscv
static void (*ready_observer)(void);
#endif
void smol_full_driver_observe(smol_full_driver_observer callback) { observer=callback; }
void smol_full_driver_observe_ready(void (*callback)(void)) {
#ifndef __riscv
    ready_observer=callback;
#else
    (void)callback;
#endif
}
__attribute__((noinline)) void smol_full_segment_ready(void) {
#ifndef __riscv
    if(ready_observer) ready_observer();
    __asm__ volatile("":::"memory");
#else
    __asm__ volatile(".option push\n.option norvc\nnop\n.option pop":::"memory");
#endif
}
__attribute__((noinline)) void smol_full_segment_done(void) {
#if defined(__riscv) && !defined(ATLAS_WORKLOAD_RETURN_TO_CALLER)
    __asm__ volatile("wfi":::"memory");
#else
    __asm__ volatile("":::"memory");
#endif
}
__attribute__((noinline)) void smol_full_segment_failed(void) {
#if defined(__riscv) && !defined(ATLAS_WORKLOAD_RETURN_TO_CALLER)
    __asm__ volatile("wfi":::"memory");
#else
    __asm__ volatile("":::"memory");
#endif
}
int smol_full_driver_drained(void) {
    return !smol_full_raw.failed && !smol_full_raw.result && !smol_full_host.active
        && !smol_full_host.active_buffers && !smol_full_host.active_buffer_count
        && !smol_full_host.launch_acknowledged;
}
static int boundary(unsigned index,int preserve_cycles) {
    if(index>200) return 1;
    for(unsigned i=0;i<56;++i) {
        const smol_full_phase_region *r=&smol_full_phase_regions[i];
        if(preserve_cycles && r->actual==(unsigned char*)smol_full_cycles) continue;
        if(smol_full_driver_differ(r->actual,smol_full_phase_expected[index][i],r->bytes)) {
            smol_full_segment_error_region=r->name;
            return 1;
        }
    }
    return 0;
}
int smol_full_driver_input(unsigned index) { return boundary(index,0); }
int smol_full_driver_phase(unsigned index,int segmented) {
    smol_full_segment_error_region="phase-drain";
    if(index>=200 || !smol_full_driver_drained()) return 1;
    const smol_full_phase_descriptor *p=&smol_full_phase_plan[index];
    smol_full_segment_error_region="phase-cursors";
    if(smol_full_phase_state.version!=1 || smol_full_phase_state.next_segment!=index
        || smol_full_raw.panel_cursor+smol_full_raw.attention_cursor!=p->launch_start
        || smol_full_raw.attention_cursor!=(SMOL_FULL_ATTENTION?index:0) || smol_full_raw.call_cursor!=p->call_start
        || smol_full_host.invocation_sequence!=p->launch_start) return 2;
    const smol_full_driver_inputs *o=&smol_full_driver_observation;
    int code;
    smol_full_raw.image=p->kind==0?p->ordinal:1;
    if(p->kind==0) {
        if(p->layer==0)
            for(size_t i=0;i<786432;++i) smol_full_image[i]=float_to_bf16(o->images[p->ordinal*786432+i]);
        code=smol_vision_image_layer(smol_full_image,&smol_full_vision_parameters,&smol_full_vision_frame,
            &smol_full_vision_states[p->ordinal],smol_full_driver_vision_launch,&smol_full_raw,p->layer);
        if(!code && p->layer==11)
            memcpy(smol_full_image_output[p->ordinal],smol_full_vision_frame.output,61440*sizeof(uint16_t));
    } else if(p->kind==1) {
        if(p->layer==0) {
            for(size_t i=0;i<128;++i) smol_full_actual_prefix_pad_mask[i]=o->image_masks[i/64];
            for(size_t i=0;i<48;++i) smol_full_actual_prefix_pad_mask[128+i]=o->language_mask[i];
            smol_full_actual_prefix_pad_mask[176]=1;
            for(size_t i=0;i<177;++i) smol_full_actual_prefix_attention_mask[i]=(i==176);
        }
        code=smol_policy_prefix_layer(smol_full_image_output[0],smol_full_image_output[1],o->state,
            smol_full_actual_prefix_pad_mask,smol_full_actual_prefix_attention_mask,&smol_full_policy_parameters,
            &smol_full_policy_frame,&smol_full_graph_state,smol_full_driver_policy_launch,&smol_full_raw,p->layer);
    } else if(p->kind==2) {
        const float *current=p->ordinal?smol_step_output_state[p->ordinal-1]:o->noise;
        float time=(float)(1.0+(double)p->ordinal*(-1.0/10.0));
        if(p->layer==0) {
            smol_step_timestep[p->ordinal][0]=time;
            memcpy(smol_step_input_state[p->ordinal],current,1600*sizeof(float));
        }
        code=smol_policy_step_layer(current,time,smol_step_output_state[p->ordinal],
            smol_full_actual_prefix_pad_mask,&smol_full_policy_parameters,&smol_full_policy_frame,
            &smol_full_graph_state,smol_full_driver_policy_launch,&smol_full_raw,p->layer);
        if(!code && p->layer==15) {
            float *next=smol_step_output_state[p->ordinal];
            memcpy(smol_step_velocity[p->ordinal],smol_full_policy_frame.velocity,1600*sizeof(float));
            if(p->ordinal==9) {
                memcpy(smol_full_actual_final_full_state,next,1600*sizeof(float));
                for(size_t row=0;row<50;++row) for(size_t column=0;column<6;++column)
                    smol_full_actual_final_actions[row*6+column]=next[row*32+column];
            }
        }
    } else return 3;
    smol_full_segment_error_region="phase-math";
    if(code) return 4;
    smol_full_segment_error_region="phase-drain";
    if(!smol_full_driver_drained() || smol_full_raw.panel_cursor+smol_full_raw.attention_cursor!=p->launch_end
        || smol_full_raw.attention_cursor!=(SMOL_FULL_ATTENTION?index+1:0)
        || smol_full_raw.call_cursor!=p->call_end || smol_full_host.invocation_sequence!=p->launch_end) return 5;
    smol_full_segment_error_region="phase-observations";
    if(segmented) {
        for(unsigned i=0;i<SMOL_FULL_LAUNCH_COUNT;++i) {
            if(i>=p->launch_start && i<p->launch_end) {
                if(!smol_full_cycles[i]) return 6;
                smol_full_segment_observations[i]=smol_full_cycles[i];
            } else if(smol_full_cycles[i]) return 6;
            smol_full_cycles[i]=0;
        }
    }
    smol_full_phase_state=(smol_full_phase_cursor){1,index+1,p->kind,p->ordinal,p->layer,
#if SMOL_FULL_ATTENTION
        p->launch_end-index-1,
#else
        p->launch_end,
#endif
        p->call_end,p->launch_end,0,0};
    if(boundary(index+1,!segmented)) return 7;
    smol_full_segment_error_region=0;
    if(observer && observer(index)) return 8;
    return 0;
}
