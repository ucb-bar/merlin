#include <assert.h>
#include <fenv.h>
#include <stdio.h>
#include <string.h>
#include "provider.c"
static unsigned long long checks;
static void check(const float*sa,const float*sb,size_t rows,size_t columns,size_t length,int false_a,int false_b){
 double ar[17*192],br[19*192],centers[17*19];uint8_t af[17],bf[19];uint32_t metadata[2*19];
 for(size_t r=0;r<rows;r++){
  af[r]=(uint8_t)(!(false_a&&r==0));
  for(size_t z=0;z<length;z++)ar[r*length+z]=(double)sa[r*length+z];
 }
 for(size_t j=0;j<columns;j++){
  bf[j]=(uint8_t)(!(false_b&&j==0));metadata[2*j]=0;metadata[2*j+1]=UINT32_C(0x7f800000);
  for(size_t z=0;z<length;z++){
   uint32_t raw;memcpy(&raw,sb+j*length+z,4);uint32_t mag=raw&UINT32_C(0x7fffffff);
   assert(!(raw&UINT32_C(0xffff)) && mag<UINT32_C(0x7f800000));
   if(mag>metadata[2*j])metadata[2*j]=mag;if(mag<metadata[2*j+1])metadata[2*j+1]=mag;
   br[j*length+z]=(double)sb[j*length+z];
  }
 }
 for(size_t r=0;r<rows;r++)for(size_t j=0;j<columns;j++){
  double center=0;for(size_t z=0;z<length;z++)center+=(double)sa[r*length+z]*(double)sb[j*length+z];centers[r*columns+j]=center;
 }
 merlin_encoded_row_equality a={sa,0,0,ar,af,rows,length,1},b={sb,0,0,br,bf,columns,length,1};
 merlin_fma_bound environment=merlin_fma_bound_begin();
 float lo0[17*19],hi0[17*19],lo1[17*19],hi1[17*19];double max0[19],max1[19];
 memset(lo0,0xa5,sizeof(lo0));memset(lo1,0xa5,sizeof(lo1));memset(hi0,0xa5,sizeof(hi0));memset(hi1,0xa5,sizeof(hi1));memset(max0,0x5a,sizeof(max0));memset(max1,0x5a,sizeof(max1));
 int x=merlin_source_rms4_point_product_estimates(&environment,&a,&b,sa,0,0,sb,ar,br,centers,lo0,hi0,rows,columns,length,max0,19);
 int y=merlin_source_rms4_point_product_estimates_produced_max(&environment,&a,&b,sa,0,0,sb,ar,br,centers,lo1,hi1,rows,columns,length,max1,19,metadata);
 assert(x==y && !memcmp(max0,max1,sizeof(max0)));
 if(x){assert(!memcmp(lo0,lo1,sizeof(lo0)) && !memcmp(hi0,hi1,sizeof(hi0)));}
 if(false_a||false_b){assert(!x && !memcmp(lo0,lo1,sizeof(lo0)) && !memcmp(hi0,hi1,sizeof(hi0)));}
 assert(!merlin_source_rms4_point_product_estimates_produced_max(&environment,&a,&b,sa,0,0,sb,ar,br,centers,lo1,hi1,rows,columns,length,max1,19,0));
 checks++;
}
int main(void){
 float a[17*192],b[19*192];a[0]=1;
 for(uint32_t word=0;word<65536;word++){
  uint32_t raw=word<<16;if((raw&UINT32_C(0x7fffffff))>=UINT32_C(0x7f800000))continue;
  memcpy(b,&raw,4);check(a,b,1,1,1,0,0);
 }
 const size_t shapes[][3]={{1,1,1},{3,5,7},{17,19,64},{2,11,192},{5,3,128}};
 uint32_t state=UINT32_C(0xc5192afb);
 for(unsigned sample=0;sample<32;sample++)for(unsigned shape=0;shape<5;shape++){
  size_t rows=shapes[shape][0],columns=shapes[shape][1],length=shapes[shape][2];
  for(size_t z=0;z<rows*length;z++){
   state=state*1664525u+1013904223u;uint32_t raw=((state>>16)&UINT32_C(0x807f))<<16;raw|=(uint32_t)(120+(state%12))<<23;
   if(z%17==0)raw=UINT32_C(0x80000000);if(z%19==0)raw=UINT32_C(0x00010000);memcpy(a+z,&raw,4);
  }
  for(size_t z=0;z<columns*length;z++){
   state=state*1664525u+1013904223u;uint32_t raw=((state>>16)&UINT32_C(0x807f))<<16;raw|=(uint32_t)(120+(state%12))<<23;
   if(z%13==0)raw=0;if(z%11==0)raw=UINT32_C(0x80010000);memcpy(b+z,&raw,4);
  }
  check(a,b,rows,columns,length,0,0);check(a,b,rows,columns,length,1,0);check(a,b,rows,columns,length,0,1);
 }
 const int modes[]={FE_TOWARDZERO,FE_DOWNWARD,FE_UPWARD};a[0]=b[0]=1;
 for(unsigned mode=0;mode<3;mode++){assert(!fesetround(modes[mode]));check(a,b,1,1,1,0,0);}
 assert(!fesetround(FE_TONEAREST));printf("PRODUCED_MAX_CERTIFICATE_EQUIVALENCE PASS checks=%llu\n",checks);return 0;
}
