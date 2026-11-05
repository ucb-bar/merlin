#include "mm_fased.h"
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>

int main(int argc,char** argv) {
  try {
    if(argc!=5) return 2;
    uint64_t base=std::stoull(argv[1]),size=std::stoull(argv[2]);
    void* allocation=mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
    if(allocation==MAP_FAILED) return 3;
    backing_data_t backing{static_cast<uint8_t*>(allocation),static_cast<size_t>(size)};
    mm_fased_t memory(base,size,8,64,backing,std::stoul(argv[3]),std::stoul(argv[4]));
    std::string line;
    while(std::getline(std::cin,line)) {
      uint64_t reset,arv,ara,ari,ars,arl,awv,awa,awi,aws,awl,wv,strb,data,last,rr,br;
      std::istringstream row(line);
      if(!(row>>reset>>arv>>ara>>ari>>ars>>arl>>awv>>awa>>awi>>aws>>awl>>wv>>strb>>data>>last>>rr>>br))return 2;
      uint64_t extra;if(row>>extra)return 2;
      if(reset>1||arv>1||awv>1||wv>1||last>1||rr>1||br>1)return 2;
      std::cout<<memory.ar_ready()<<' '<<memory.aw_ready()<<' '<<memory.w_ready()<<' '
          <<memory.r_valid()<<' '<<memory.r_id()<<' '<<*static_cast<uint64_t*>(memory.r_data())<<' '
          <<memory.r_resp()<<' '<<memory.r_last()<<' '<<memory.b_valid()<<' '
          <<memory.b_id()<<' '<<memory.b_resp()<<std::endl;
      memory.tick(reset,arv,ara,ari,ars,arl,awv,awa,awi,aws,awl,wv,strb,&data,last,rr,br);
    }
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 2;}
}
