#include <merlin_timing/cache_response_timeline.hpp>
#include <iostream>
using namespace merlin_timing;
int main(){std::uint64_t id,accepted;while(std::cin>>id>>accepted){auto p=plan_cache_response({4,6,1,1,2},{accepted,{0,0,0,0},0,false});if(!p.complete)return 2;std::cout<<id;for(auto x:p.accepted_beats)std::cout<<' '<<x;std::cout<<' '<<*p.line_visible<<' '<<*p.lookup_response<<'\n';}}
