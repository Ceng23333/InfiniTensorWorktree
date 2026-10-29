#include "context/allocators/pinnable_block_allocator.hpp"
#include <infini/rt.h>
#include <cstdio>
extern "C" int test_capture_allocator() {
 infini::rt::set_runtime_device_type(infini::rt::Device::Type::kCpu);
 int failures=0;
 for (size_t size : {size_t(1024),size_t(257)*1024*1024}) {
  infinicore::PinnableBlockAllocator a({infinicore::Device::Type::kCpu,0});
  auto *p=a.allocate(size);p[0]=std::byte{42};a.begin_pin_mode();a.retain_for_capture(p);a.deallocate(p);auto lease=a.commit_pin_mode();
  auto *q=a.allocate(size);q[0]=std::byte{7};bool protected_ok=q!=p && p[0]==std::byte{42};
  printf("size=%zu retained_graph_storage_protected=%d\n",size,protected_ok);failures+=!protected_ok;
  a.deallocate(q);lease.reset();a.trim();
 }
 // Retained external inputs must not be recycled even within a capture.
 {
  infinicore::PinnableBlockAllocator a({infinicore::Device::Type::kCpu,0});auto*p=a.allocate(1024);p[0]=std::byte{42};a.begin_pin_mode();a.retain_for_capture(p);a.deallocate(p);auto*q=a.allocate(1024);q[0]=std::byte{7};bool isolated=p!=q && p[0]==std::byte{42};printf("external_capture_input_protected=%d\n",isolated);failures+=!isolated;a.deallocate(q);a.cancel_pin_mode();a.trim();
 }
 // Reuse within the same capture remains valid and bounds graph memory use.
 {
  infinicore::PinnableBlockAllocator a({infinicore::Device::Type::kCpu,0});a.begin_pin_mode();auto*p=a.allocate(1024);a.deallocate(p);auto*q=a.allocate(1024);
  bool reuse=p==q;printf("same_capture_reuse=%d\n",reuse);failures+=!reuse;a.deallocate(q);auto lease=a.commit_pin_mode();
  a.begin_pin_mode();auto*r=a.allocate(1024);bool separate=r!=p;printf("second_capture_isolated=%d\n",separate);failures+=!separate;a.deallocate(r);a.cancel_pin_mode();lease.reset();a.trim();
 }
 fflush(stdout);return failures;
}
