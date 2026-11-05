#include "smol_policy_graph.h"
#include "smol_vision_graph.h"
#include "smol_policy_raw.h"
#include "smol_full_phase.h"
#include <string.h>
smol_full_phase_cursor smol_full_phase_state;
typedef int (*phase_callback)(unsigned,unsigned,unsigned);
static phase_callback phase_observe;
static unsigned phase_enabled;
void smol_phase_configure(unsigned enabled,phase_callback callback) {
    phase_enabled=enabled;phase_observe=callback;
}
typedef int (*predecessor_callback)(unsigned,unsigned,const void*,size_t,size_t,unsigned,unsigned,unsigned);
static predecessor_callback predecessor_observe;
typedef int (*output_callback)(unsigned,unsigned,const void*,size_t,size_t);
static output_callback output_observe;
void smol_output_configure(output_callback callback){output_observe=callback;}
void smol_predecessor_configure(predecessor_callback callback) { predecessor_observe=callback; }
extern int __real_smol_policy_raw_vision(void*,unsigned,unsigned,const uint16_t*,size_t,size_t,uint16_t*,size_t);
extern int __real_smol_policy_raw_graph(void*,unsigned,unsigned,unsigned,unsigned,const void*,size_t,size_t,
    unsigned,void*,size_t);
int __wrap_smol_policy_raw_vision(void *context,unsigned layer,unsigned role,const uint16_t *input,
    size_t rows,size_t k,uint16_t *output,size_t n) {
    smol_policy_raw_context *raw=context;
    if(!predecessor_observe || predecessor_observe((unsigned)raw->call_cursor,0,input,rows,k,7,layer,role))
        return SMOL_RAW_INPUT;
    int code=__real_smol_policy_raw_vision(context,layer,role,input,rows,k,output,n);
if(!code && output_observe) code=output_observe((unsigned)raw->call_cursor-1,0,output,rows,n);
return code;
}
int __wrap_smol_policy_raw_graph(void *context,unsigned site,unsigned layer,unsigned role,unsigned dtype,
    const void *input,size_t rows,size_t k,unsigned output_dtype,void *output,size_t n) {
    smol_policy_raw_context *raw=context;
    if(!predecessor_observe || predecessor_observe((unsigned)raw->call_cursor,dtype,input,rows,k,site,layer,role))
        return SMOL_RAW_INPUT;
    int code=__real_smol_policy_raw_graph(context,site,layer,role,dtype,input,rows,k,output_dtype,output,n);
if(!code && output_observe) code=output_observe((unsigned)raw->call_cursor-1,output_dtype,output,rows,n);
return code;
}
extern uint16_t smol_full_image_output[2][61440];
extern float *const smol_step_velocity[10];
extern float smol_full_actual_final_full_state[1600],smol_full_actual_final_actions[300];
static int phased_vision(const uint16_t *image,const smol_vision_parameters *p,
    const smol_vision_frame *w,smol_vision_state *s,smol_vision_project callback,void *context) {
    if (!phase_enabled) return smol_vision_image(image,p,w,s,callback,context);
    unsigned ordinal=((smol_policy_raw_context*)context)->image;
    for(unsigned layer=0;layer<12;++layer) {
        int code=smol_vision_image_layer(image,p,w,s,callback,context,layer);
        if(code) return code;
        if(layer==11) memcpy(smol_full_image_output[ordinal],w->output,61440*sizeof(uint16_t));
        if(phase_observe(0,ordinal,layer)) return 72;
    }
    return 0;
}
static int phased_prefix(const uint16_t *a,const uint16_t *b,const float *state,
    const uint8_t *pad,const uint8_t *attention,const smol_policy_parameters *p,
    const smol_policy_frame *w,smol_policy_state *s,smol_policy_graph_project callback,void *context) {
    if(!phase_enabled) return smol_policy_prefix(a,b,state,pad,attention,p,w,s,callback,context);
    for(unsigned layer=0;layer<16;++layer) {
        int code=smol_policy_prefix_layer(a,b,state,pad,attention,p,w,s,callback,context,layer);
        if(code) return code;
        if(phase_observe(1,0,layer)) return 72;
    }
    return 0;
}
static int phased_step(const float *input,float time,float *output,const uint8_t *pad,
    const smol_policy_parameters *p,const smol_policy_frame *w,smol_policy_state *s,
    smol_policy_graph_project callback,void *context) {
    if(!phase_enabled) return smol_policy_step(input,time,output,pad,p,w,s,callback,context);
    unsigned ordinal=s->steps;
    for(unsigned layer=0;layer<16;++layer) {
        int code=smol_policy_step_layer(input,time,output,pad,p,w,s,callback,context,layer);
        if(code) return code;
        if(layer==15) {
            memcpy(smol_step_velocity[ordinal],w->velocity,1600*sizeof(float));
            if(ordinal==9) {
                memcpy(smol_full_actual_final_full_state,output,1600*sizeof(float));
                for(unsigned row=0;row<50;++row) for(unsigned col=0;col<6;++col)
                    smol_full_actual_final_actions[row*6+col]=output[row*32+col];
            }
        }
        if(phase_observe(2,ordinal,layer)) return 72;
    }
    return 0;
}

#include "host_matrix_fp8pack.h"
#include "smol_host_bits.h"
extern smol_policy_raw_context smol_full_raw;
extern const smol_policy_panel smol_full_panels[2715];
extern const smol_policy_call smol_full_calls[1421];
smol_policy_panel smol_capture_panels[2715];
smol_policy_call smol_capture_calls[1421];
smol_attention_panel smol_capture_attention[200];
void smol_capture_initialize(void){
    memcpy(smol_capture_panels,smol_full_panels,sizeof(smol_capture_panels));
    memcpy(smol_capture_calls,smol_full_calls,sizeof(smol_capture_calls));
    smol_full_raw.panels=smol_capture_panels;smol_full_raw.calls=smol_capture_calls;
    smol_full_raw.attention_panels=smol_capture_attention;smol_full_raw.attention_count=200;
}
int smol_capture_scale(unsigned dtype,const void *input,size_t rows,size_t k,int *scale){
    if(rows*k>smol_full_raw.input_capacity)return 1;
    for(size_t i=0;i<rows*k;++i)smol_full_raw.input_copy[i]=dtype==0?((const uint16_t*)input)[i]:float_to_bf16(((const float*)input)[i]);
    return host_matrix_scale_bf16(smol_full_raw.input_copy,rows,k,scale);
}
size_t smol_capture_raw_size(void){return sizeof(smol_policy_raw_context);}
size_t smol_capture_attention_size(void){return sizeof(smol_attention_panel);}
