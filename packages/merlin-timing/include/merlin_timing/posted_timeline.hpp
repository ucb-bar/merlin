#ifndef MERLIN_TIMING_POSTED_TIMELINE_HPP
#define MERLIN_TIMING_POSTED_TIMELINE_HPP
#include "engine.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <optional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
namespace merlin_timing {
// Causal execution kernel for source-owned actors. Posting an operation does
// NOT execute its callback, publish a future row or reserve a future port.
// The caller must close ALL external arrivals before each exclusive frontier.
// Completion is published only by an actual dispatched callback at that time.
// This kernel supplies no device arbitration, stage count or memory timing.
class PostedTimeline {
 public:
  enum class Kind : std::size_t {source_arrival,memory_accept,response_beat,local_publish,operand_consume,command_retire,completion,other,count};
  struct Token {std::uint64_t id;};
  struct OrderedWriterHandoff {ValueId value;Token previous_complete,previous_consumed;};
  struct ClassSummary {std::uint64_t posted=0,dispatched=0,dependency_delay_cycles=0;std::size_t pending=0,blocked_on_tokens=0;};
  struct Summary {std::array<ClassSummary,std::size_t(Kind::count)> classes{};std::size_t peak_pending=0,unresolved_tokens=0;Cycle closed=0;bool failed=false;};
  struct Spec {std::size_t pending, tokens;std::uint64_t dispatched;};
  struct Progress {Cycle closed;std::size_t pending;std::uint64_t dispatched;bool complete,failed;};
  using Action=std::function<void(PostedTimeline &,Cycle)>;
  explicit PostedTimeline(Spec spec):spec_(spec) {
    if(!spec.pending||!spec.tokens||!spec.dispatched||spec.pending>1048576||spec.tokens>67108864)
      throw std::invalid_argument("invalid bounded posted timeline");
  }
  std::optional<Token> token() {
    if(failed_||tokens_.size()>=spec_.tokens){fail("completion token capacity");return {};}
    tokens_.push_back({});return Token{tokens_.size()};
  }
  bool post(Cycle at,std::vector<Token> dependencies,Action action,Kind kind=Kind::other) {
    return post_phase(at,std::move(dependencies),std::move(action),kind,false);
  }
  // Hardware arbitration may run only after all causally ready ordinary
  // callbacks on this edge have offered their requests. No cycle is added.
  // Accepted notifications may post ordinary callbacks on the same edge;
  // consumers must reject a new late hardware request after their settlement
  // (or derive its next-edge visibility from an actual registered source).
  // The external-arrival closure requirement of advance/step is unchanged.
  bool post_settled(Cycle at,std::vector<Token> dependencies,Action action,Kind kind=Kind::other) {
    return post_phase(at,std::move(dependencies),std::move(action),kind,true);
  }
 private:
  bool post_phase(Cycle at,std::vector<Token> dependencies,Action action,Kind kind,bool settled) {
    if(failed_)return false;
    if(std::size_t(kind)>=std::size_t(Kind::count)||!action||at==UINT64_MAX||events_.size()>=spec_.pending||at<closed_||(dispatching_&&at<clock_))
      return fail("unbounded, retroactive or empty posted event");
    for(auto t:dependencies)if(!valid(t))return fail("unknown completion dependency");
    if(next_==UINT64_MAX)return fail("posted event identity overflow");
    events_.push_back({next_++,at,std::move(dependencies),std::move(action),kind,settled});
    ++classes_[std::size_t(kind)].posted;peak_pending_=std::max(peak_pending_,events_.size());return true;
  }
 public:
  bool complete(Token token) {
    if(failed_)return false;
    if(!dispatching_||!valid(token)||tokens_[token.id-1])return fail("completion outside dispatch or repeated identity");
    tokens_[token.id-1]=clock_;return true;
  }
  std::optional<Cycle> completed(Token t)const {
    return !failed_&&valid(t)?tokens_[t.id-1]:std::optional<Cycle>{};
  }
  bool has_deferred_values()const{return !visibility_.empty();}
  // Accepted classification may expose a new logical tag before its lazy
  // Engine writer can dispatch. Keep that generation pending, not ready(0).
  // Explicit declarations describe value visibility, never future port booking.
  bool bind_deferred_values(const merlin_timing::Event &event,View<ValueId> declared,
      Token admitted,Token complete,std::vector<Token>&dependencies) {
    return bind_ordered_deferred_values(event,declared,admitted,complete,dependencies,{});
  }
  // A source owner may prove that a predecessor has entered an ordered
  // internal pipeline. Such a handoff changes only same-value writer ordering:
  // external readers still wait for final publication, and independent WAR
  // obligations remain intact. The caller owns FIFO publication order and
  // internal forwarding; this kernel does not infer them from latency.
  bool bind_ordered_deferred_values(const merlin_timing::Event &event,View<ValueId> declared,
      Token admitted,Token complete,std::vector<Token>&dependencies,View<OrderedWriterHandoff>handoffs) {
    if(!valid(admitted)||!valid(complete))return fail("unbound deferred value owner");
    for(std::size_t i=0;i<handoffs.size;++i){const auto&h=handoffs[i];
      if(!valid(h.previous_complete)||!valid(h.previous_consumed)||
         std::find(declared.data,declared.data+declared.size,h.value)==declared.data+declared.size)
        return fail("ordered handoff is not an original declared writer");
    }
    const auto predecessor=[&](ValueId value,Token writer){
      for(std::size_t i=0;i<handoffs.size;++i)if(handoffs[i].value==value&&handoffs[i].previous_complete.id==writer.id)
        return handoffs[i].previous_consumed;
      return writer;
    };
    for(std::size_t i=0;i<declared.size;++i){
      if(std::find(event.writes.data,event.writes.data+event.writes.size,declared[i])==event.writes.data+event.writes.size)
        return fail("deferred declaration is not an actual writer");
      if(visibility_.size()>=1048576&&!visibility_.count(declared[i]))return fail("deferred value capacity");
      visibility_.try_emplace(declared[i]);
    }
    const auto depend=[&](Token t){if(!completed(t))dependencies.push_back(t);};
    const auto read=[&](ValueId id,bool retained){
      const auto found=visibility_.find(id);if(found==visibility_.end())return;
      auto &state=found->second;if(state.writer)depend(predecessor(id,*state.writer));
      state.readers.erase(std::remove_if(state.readers.begin(),state.readers.end(),
        [&](auto t){return completed(t).has_value();}),state.readers.end());
      state.readers.push_back(retained?complete:admitted);
    };
    for(std::size_t i=0;i<event.reads.size;++i)read(event.reads[i],event.hold_reads_until_ready);
    for(std::size_t i=0;i<event.retained_reads.size;++i)read(event.retained_reads[i],true);
    for(std::size_t i=0;i<event.writes.size;++i){
      const auto found=visibility_.find(event.writes[i]);if(found==visibility_.end())continue;
      auto &state=found->second;if(state.writer&&state.writer->id!=complete.id)depend(predecessor(event.writes[i],*state.writer));
      for(auto reader:state.readers)if(reader.id!=admitted.id&&reader.id!=complete.id)depend(reader);
      state.readers.clear();state.writer=complete;
    }
    std::sort(dependencies.begin(),dependencies.end(),[](auto a,auto b){return a.id<b.id;});
    dependencies.erase(std::unique(dependencies.begin(),dependencies.end(),[](auto a,auto b){return a.id==b.id;}),dependencies.end());
    return true;
  }
  // No implicit drain to an unknown future. In particular a LOAD actor cannot
  // call this to completion while later CPU shared arrivals are still open.
  bool advance(Cycle exclusive,bool all_external_arrivals_closed) {
    if(failed_)return false;
    if(dispatching_||!all_external_arrivals_closed||exclusive<closed_)
      return fail("missing closed source frontier or reentrant advance");
    while(true) {
      auto chosen=events_.size();Cycle earliest=UINT64_MAX;std::uint64_t identity=UINT64_MAX;bool settled=true;
      for(std::size_t i=0;i<events_.size();++i){
        auto at=events_[i].at;bool ready=true;
        for(auto t:events_[i].dependencies){const auto &done=tokens_[t.id-1];if(!done){ready=false;break;}at=std::max(at,*done);}
        if(ready&&at<exclusive&&(at<earliest||(at==earliest&&(events_[i].settled<settled||(events_[i].settled==settled&&events_[i].id<identity))))){chosen=i;earliest=at;identity=events_[i].id;settled=events_[i].settled;}
      }
      if(chosen==events_.size())break;
      if(dispatched_>=spec_.dispatched)return fail("posted dispatch budget");
      auto event=std::move(events_[chosen]);events_.erase(events_.begin()+chosen);
      if(earliest<clock_)return fail("retroactive dependency release");
      clock_=earliest;++dispatched_;dispatching_=true;
      auto &metrics=classes_[std::size_t(event.kind)];++metrics.dispatched;
      const auto delay=earliest-event.at;
      if(delay>UINT64_MAX-metrics.dependency_delay_cycles)return fail("posted diagnostic accounting overflow");
      metrics.dependency_delay_cycles+=delay;
      try{event.action(*this,clock_);}catch(...){dispatching_=false;fail("source actor exception");throw;}
      dispatching_=false;if(failed_)return false;
    }
    closed_=exclusive;return true;
  }
  // Dispatch exactly one causally ready event. An in-order source may call
  // this while its next admission actor is pending: no later instruction can
  // arrive before that actor accepts. Keep the frontier at the dispatched
  // instant, allowing another source to arrive on the SAME edge. This does not
  // reopen an earlier advance() frontier or infer a missing external arrival.
  bool step(bool all_external_arrivals_closed) {
    if(failed_)return false;
    if(dispatching_ || !all_external_arrivals_closed)return fail("open source frontier during step");
    auto chosen=events_.size();Cycle earliest=UINT64_MAX;std::uint64_t identity=UINT64_MAX;bool settled=true;
    for(std::size_t i=0;i<events_.size();++i){
      auto at=events_[i].at;bool ready=true;
      for(auto t:events_[i].dependencies){const auto &done=tokens_[t.id-1];if(!done){ready=false;break;}at=std::max(at,*done);}
      if(ready&&(at<earliest||(at==earliest&&(events_[i].settled<settled||(events_[i].settled==settled&&events_[i].id<identity))))){chosen=i;earliest=at;identity=events_[i].id;settled=events_[i].settled;}
    }
    if(chosen==events_.size())return false;
    if(dispatched_>=spec_.dispatched)return fail("posted dispatch budget");
    auto event=std::move(events_[chosen]);events_.erase(events_.begin()+chosen);
    if(earliest<clock_ || earliest<closed_)return fail("retroactive dependency release");
    clock_=closed_=earliest;++dispatched_;dispatching_=true;
    auto &metrics=classes_[std::size_t(event.kind)];++metrics.dispatched;
    const auto delay=earliest-event.at;
    if(delay>UINT64_MAX-metrics.dependency_delay_cycles)return fail("posted diagnostic accounting overflow");
    metrics.dependency_delay_cycles+=delay;
    try{event.action(*this,clock_);}catch(...){dispatching_=false;fail("source actor exception");throw;}
    dispatching_=false;return !failed_;
  }
  const Spec& specification()const{return spec_;}
  Cycle now()const{return std::max(clock_,closed_);}
  bool fail(const std::string &reason){if(!failed_)reason_=reason;failed_=true;return false;}
  const std::string &reason()const{return reason_;}
  Summary summary()const {
    Summary out;out.classes=classes_;out.peak_pending=peak_pending_;out.closed=closed_;out.failed=failed_;
    for(const auto&t:tokens_)out.unresolved_tokens+=!t;
    for(const auto&e:events_){auto &c=out.classes[std::size_t(e.kind)];++c.pending;
      for(auto t:e.dependencies)if(!tokens_[t.id-1]){++c.blocked_on_tokens;break;}}
    return out;
  }
  Progress progress()const{return {closed_,events_.size(),dispatched_,!failed_&&events_.empty()&&std::all_of(tokens_.begin(),tokens_.end(),[](const auto&t){return bool(t);}),failed_};}
 private:
  struct Visibility {std::optional<Token>writer;std::vector<Token>readers;};
  std::map<ValueId,Visibility>visibility_;
  struct Event {std::uint64_t id;Cycle at;std::vector<Token> dependencies;Action action;Kind kind;bool settled;};
  bool valid(Token t)const{return t.id&&t.id<=tokens_.size();}
  std::array<ClassSummary,std::size_t(Kind::count)>classes_{};std::size_t peak_pending_=0;
  Spec spec_;Cycle closed_=0,clock_=0;std::uint64_t next_=1,dispatched_=0;
  bool failed_=false,dispatching_=false;std::string reason_;
  std::vector<std::optional<Cycle>>tokens_;std::vector<Event>events_;
};
}
#endif
