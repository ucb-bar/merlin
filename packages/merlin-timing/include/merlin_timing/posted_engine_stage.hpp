#ifndef MERLIN_TIMING_POSTED_ENGINE_STAGE_HPP
#define MERLIN_TIMING_POSTED_ENGINE_STAGE_HPP
#include "posted_timeline.hpp"
#include <memory>
namespace merlin_timing {
// Owned views survive asynchronous dispatch. A stage has source-derived fixed
// accepted work; unknown/backpressured downstream work must be another stage.
// No memory callback is run at post time; callers post it as an arrival actor.
struct OwnedStage {
 Cycle earliest=0;
 std::vector<ValueId> reads,writes,retained_reads;
 std::vector<Demand> demands;
 std::optional<Cycle> latency;
 bool hold_reads_until_ready=false;
 PostedTimeline::Kind kind=PostedTimeline::Kind::other;
 std::optional<StageOrigin> origin{};
 std::vector<ValueId> deferred_writes;
 Event event(Cycle at)const {
   return {std::max(earliest,at),{reads.data(),reads.size()},{writes.data(),writes.size()},
     {demands.data(),demands.size()},latency,hold_reads_until_ready,{retained_reads.data(),retained_reads.size()},origin};
 }
};
class PostedEngineStage {
 struct State {Engine *engine;OwnedStage stage;PostedTimeline::Token completion;std::optional<PostedTimeline::Token> admitted;std::function<void(const Result&)> accepted;};
 static void dispatch(const std::shared_ptr<State>&state,PostedTimeline&q,Cycle at){
   auto event=state->stage.event(at);const auto start=state->engine->earliest_stage_start(event);
   if(!start){q.fail("source stage has unresolved dependency, port or latency");return;}
   if(*start>at){q.post(*start,{},[state](auto&q,auto at){dispatch(state,q,at);},state->stage.kind);return;}
   // Only accepted work is committed. A prior query is never a reservation.
   const auto result=state->engine->submit_detached(event);
   if(result.start!=std::optional<Cycle>{at}||!result.ready||*result.ready<at){q.fail("source stage admission changed or is unknown");return;}
   if(state->admitted)q.complete(*state->admitted);
   if(state->accepted)state->accepted(result);
   q.post(*result.ready,{},[state](auto&q,auto){q.complete(state->completion);},PostedTimeline::Kind::completion);
 }
 public:
 static bool post(PostedTimeline&q,Engine&engine,OwnedStage stage,
                  std::vector<PostedTimeline::Token>dependencies,PostedTimeline::Token completion,std::function<void(const Result&)> accepted={}){
   const auto at=stage.earliest;
   std::optional<PostedTimeline::Token> admitted;
   if(q.has_deferred_values()||!stage.deferred_writes.empty()){admitted=q.token();if(!admitted)return false;}
   if(!q.bind_deferred_values(stage.event(at),{stage.deferred_writes.data(),stage.deferred_writes.size()},
       admitted.value_or(completion),completion,dependencies))return false;
   auto state=std::make_shared<State>(State{&engine,std::move(stage),completion,admitted,std::move(accepted)});
   return q.post(at,std::move(dependencies),[state](auto&q,auto at){dispatch(state,q,at);},state->stage.kind);
 }
};
}
#endif
