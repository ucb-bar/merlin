#include <merlin_timing/posted_engine_stage.hpp>
#include <cassert>
using namespace merlin_timing;
int main(){
 Engine e(16,{{0,1,0}});PostedTimeline q({16,16,64});auto dma=q.token(),cpu=q.token(),compute=q.token();assert(dma&&cpu&&compute);
 OwnedStage load;load.kind=PostedTimeline::Kind::local_publish;load.earliest=20;load.writes={0};load.demands={{0,1,Cycle{1}}};load.latency=1;
 const auto before=e.report();auto view=load.event(20);assert(e.earliest_stage_start(view)==20);auto after=e.report();assert(before.events==after.events&&before.modeled_makespan==after.modeled_makespan&&e.ready(0)==0);
 assert(PostedEngineStage::post(q,e,load,{},*dma));assert(q.advance(4,true));assert(e.report().events==0&&e.ready(0)==0); // future LOAD hasn't booked bank or published row
 OwnedStage host;host.kind=PostedTimeline::Kind::source_arrival;host.earliest=4;host.writes={1};host.demands={{0,1,Cycle{8}}};host.latency=1;
 assert(PostedEngineStage::post(q,e,host,{},*cpu));
 OwnedStage consumer;consumer.kind=PostedTimeline::Kind::operand_consume;consumer.earliest=5;consumer.reads={0};consumer.writes={2};consumer.demands={{0,1,Cycle{2}}};consumer.latency=2;
 assert(PostedEngineStage::post(q,e,consumer,{*dma},*compute));assert(q.advance(13,true));assert(q.completed(*cpu)==5&&!q.completed(*dma));assert(e.ready(1)==5&&e.ready(0)==0&&e.ready(2)==0);
 assert(q.advance(24,true));assert(q.completed(*dma)==21&&q.completed(*compute)==23);assert(q.progress().complete);const auto metrics=q.summary();assert(metrics.classes[std::size_t(PostedTimeline::Kind::local_publish)].dispatched==1);assert(metrics.classes[std::size_t(PostedTimeline::Kind::source_arrival)].dispatched==1);assert(metrics.classes[std::size_t(PostedTimeline::Kind::operand_consume)].dependency_delay_cycles==16);assert(metrics.unresolved_tokens==0);assert(e.ready(0)==21&&e.ready(2)==23);
 // Real occupied port causes a retry; it is never booked at a future time by
 // a readiness query, and another already posted source may arrive meanwhile.
 Engine waiting(8,{{0,1,0}});Demand hold{0,1,Cycle{10}};Event current{0,{},{},{&hold,1},Cycle{1}};assert(waiting.submit(current).start==0);
 PostedTimeline retries({8,8,32});auto result=retries.token();assert(result);OwnedStage work;work.earliest=2;work.writes={0};work.demands={{0,1,Cycle{1}}};work.latency=1;
 assert(PostedEngineStage::post(retries,waiting,work,{},*result));assert(retries.advance(9,true));assert(waiting.report().events==1&&!retries.completed(*result));assert(retries.advance(12,true));assert(retries.completed(*result)==11);
 Engine unknown(8,{{0,1,0}});PostedTimeline refuses({8,8,32});auto token=refuses.token();assert(token);work.latency.reset();assert(PostedEngineStage::post(refuses,unknown,work,{},*token));assert(!refuses.advance(20,true));assert(unknown.report().events==0&&!refuses.completed(*token));
 // Actual target-stage admission never drags the separate CPU frontier
 // into the target's future. Accepted callback is distinct from completion.
 Engine independent(8,{{0,1,0}});PostedTimeline source({8,8,32});auto delivered=source.token();assert(delivered);
 OwnedStage future;future.earliest=20;future.writes={0};future.latency=8;
 future.origin=StageOrigin{0x80001000,0x123,std::uint64_t{19},"original_future_load"};
 bool admitted=false;assert(PostedEngineStage::post(source,independent,future,{},*delivered,[&](const Result&r){assert(r.start==20&&r.ready==28);admitted=true;}));
 auto snapshot=future.event(20);assert(source.step(true)&&admitted&&!source.completed(*delivered));assert(independent.issue_frontier()==0);
 assert(source.step(true)&&source.completed(*delivered)==28);
 assert(snapshot.origin&&snapshot.origin->pc==0x80001000&&snapshot.origin->bits==0x123&&snapshot.origin->committed_instruction==19&&snapshot.origin->actor=="original_future_load");
 // Event custody is an owned snapshot; later producer edits cannot retag it.
 future.origin->pc=0x80002000;assert(snapshot.origin->pc==0x80001000);
}
