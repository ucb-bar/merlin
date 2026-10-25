#include <merlin_timing/posted_engine_stage.hpp>
#include <cassert>
using namespace merlin_timing;
int main(){
 Engine e(16,{{0,1,0}});Demand hold{0,1,Cycle{20}};Event occupied{0,{},{},{&hold,1},Cycle{1}};e.submit(occupied);
 PostedTimeline q({32,32,128});auto fill=q.token(),hit=q.token(),other=q.token(),replacement=q.token();assert(fill&&hit&&other&&replacement);
 OwnedStage miss;miss.earliest=2;miss.writes={0};miss.deferred_writes={0};miss.demands={{0,1,Cycle{4}}};miss.latency=17;
 OwnedStage next;next.earliest=3;next.reads={0};next.latency=1;
 OwnedStage unrelated;unrelated.earliest=3;unrelated.writes={1};unrelated.latency=1;
 assert(PostedEngineStage::post(q,e,miss,{},*fill));assert(PostedEngineStage::post(q,e,next,{},*hit));assert(PostedEngineStage::post(q,e,unrelated,{},*other));
 // A later writer preserves the earlier pending read (actual admission, not
 // its later scalar result), plus prior fill WAW. It never publishes early.
 OwnedStage update;update.earliest=4;update.writes={0};update.deferred_writes={0};update.latency=2;
 assert(PostedEngineStage::post(q,e,update,{},*replacement));assert(q.advance(50,true));
 assert(q.completed(*fill)==37&&q.completed(*hit)==38&&q.completed(*other)==4&&q.completed(*replacement)==39);
 assert(q.progress().complete);
 // A declaration must be an actual writer. No synthetic generation/token.
 PostedTimeline invalid({4,4,8});auto end=invalid.token();assert(end);next.deferred_writes={3};
 assert(!PostedEngineStage::post(invalid,e,next,{},*end)&&invalid.progress().failed);
}
