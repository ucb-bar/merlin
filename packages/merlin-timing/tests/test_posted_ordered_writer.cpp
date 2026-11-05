#include <merlin_timing/posted_timeline.hpp>
#include <cassert>
using namespace merlin_timing;
int main(){
 PostedTimeline q({128,128,512});ValueId value=1;auto admitted=*q.token(),first=*q.token(),captured=*q.token(),second=*q.token(),accepted=*q.token(),reader=*q.token();
 Event write{0,{},{&value,1},{},Cycle{0}};std::vector<PostedTimeline::Token>d;
 assert(q.bind_deferred_values(write,{&value,1},admitted,first,d)&&d.empty());
 PostedTimeline::OrderedWriterHandoff h{value,first,captured};
 Event update{0,{&value,1},{&value,1},{},Cycle{0}};
 assert(q.bind_ordered_deferred_values(update,{&value,1},accepted,second,d,{&h,1}));assert(d.size()==1&&d[0].id==captured.id);
 q.post(0,d,[accepted](auto&q,auto){q.complete(accepted);});
 Event read{0,{&value,1},{},{},Cycle{0}};d.clear();assert(q.bind_deferred_values(read,{},reader,reader,d));assert(d.size()==1&&d[0].id==second.id);
 q.post(0,d,[reader](auto&q,auto){q.complete(reader);});
 q.post(5,{},[captured](auto&q,auto){q.complete(captured);});q.post(10,{},[first](auto&q,auto){q.complete(first);});q.post(12,{},[second](auto&q,auto){q.complete(second);});assert(q.advance(20,true));assert(q.completed(accepted)==5&&q.completed(first)==10&&q.completed(reader)==12);
 // A stale predecessor identity cannot waive a later writer, nor any reader.
 PostedTimeline p({128,128,512});auto a=*p.token(),b=*p.token(),c=*p.token(),r=*p.token(),z=*p.token();d.clear();assert(p.bind_deferred_values(write,{&value,1},a,b,d));d.clear();assert(p.bind_deferred_values(read,{},r,r,d));h={value,c,a};d.clear();assert(p.bind_ordered_deferred_values(write,{&value,1},a,z,d,{&h,1}));bool writer=false,war=false;for(auto x:d){writer|=x.id==b.id;war|=x.id==r.id;}assert(writer&&war);
}
