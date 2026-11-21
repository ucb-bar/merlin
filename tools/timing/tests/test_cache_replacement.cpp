#include <merlin_timing/engine.hpp>
#include <cassert>
#include <stdexcept>
using namespace merlin_timing;
int main(){
  ReadSteppedLfsr source({16,10,0xb400,1});
  assert(source.directory_read(8)==0 && source.state()==2);
  assert(source.directory_read(8)==0 && source.state()==4);
  for(unsigned i=0;i<5;++i)source.directory_read(8);
  assert(source.state()==128);assert(source.directory_read(8)==2 && source.state()==256);
  // Explicit source law may replace a valid way even when invalid ways exist.
  SetAssociativeCache cache({1,4,64},ReadSteppedLfsrSpec{4,2,12,15});
  const auto preview=cache.preview_access(0);const auto state=cache.replacement_state();
  assert(!preview.hit&&!preview.evicted_line&&preview.slot==2);
  for(unsigned i=0;i<4;++i){const auto again=cache.preview_access(0);assert(again.slot==preview.slot);}
  assert(cache.replacement_state()==state&&cache.report().accesses==0);
  const auto a=cache.access(0,true);assert(!a.hit && a.slot==2);
  const auto b=cache.access(64);assert(!b.hit && b.slot==0);
  const auto c=cache.access(128);assert(!c.hit && c.slot==0 && c.evicted_line==1);
  assert(cache.report().entries==2 && cache.probe(0).hit && !cache.probe(64).hit);
  // Hits advance the directory generator too; no 'miss-only random' shortcut.
  assert(cache.access(0).hit);
  const auto before=cache.report();const auto dirty=cache.preview_access(192);
  assert(!dirty.hit&&dirty.dirty_eviction&&dirty.evicted_line==0&&dirty.slot==2);
  assert(cache.report().accesses==before.accesses&&cache.probe(0).hit);
  const auto d=cache.access(192);assert(!d.hit && d.slot==2 && d.evicted_line==0 && d.dirty_eviction);
  cache.clear();assert(cache.access(0).slot==0); // invalidation does not reset source RNG
  SetAssociativeCache restarted({1,4,64},ReadSteppedLfsrSpec{4,2,12,15});
  assert(restarted.access(0).slot==2);
  // Default cache continues invalid-way-first/LRU behavior, byte for byte.
  SetAssociativeCache old({1,4,64});
  for(unsigned i=0;i<4;++i)assert(!old.access(i*64).evicted_line);
  assert(old.access(256).evicted_line==0);
  bool refused=false;try{SetAssociativeCache bad({1,3,64},ReadSteppedLfsrSpec{16,10,0xb400,1});}catch(const std::invalid_argument&){refused=true;}assert(refused);
  refused=false;try{ReadSteppedLfsr bad({16,10,0xb400,0});}catch(const std::invalid_argument&){refused=true;}assert(refused);
}
