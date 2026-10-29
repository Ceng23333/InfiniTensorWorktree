#include <mccl.h>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <string>
#define MC(x) do { auto e=(x); if(e != mcSuccess){fprintf(stderr,"%s failed: %d\n",#x,int(e));std::abort();}}while(0)
#define CC(x) do {auto e=(x);if(e!=mcclSuccess){fprintf(stderr,"%s: %s\n",#x,mcclGetErrorString(e));std::abort();}}while(0)
int main(int argc,char**argv){
 size_t n=argc>1?std::stoull(argv[1]):5120; int iters=argc>2?atoi(argv[2]):200;
 mcclUniqueId id; CC(mcclGetUniqueId(&id)); std::atomic<int> ready{0};
 std::vector<std::thread> workers;
 for(int rank=0;rank<4;rank++)workers.emplace_back([&,rank]{
  MC(mcSetDevice(rank));mcStream_t stream;MC(mcStreamCreate(&stream));mcclComm_t comm;CC(mcclCommInitRank(&comm,4,id,rank));
  void *in,*out;MC(mcMalloc(&in,n*2));MC(mcMalloc(&out,n*2));std::vector<uint16_t> h(n,0x3f80);MC(mcMemcpy(in,h.data(),n*2,mcMemcpyHostToDevice));
  for(int i=0;i<20;i++)CC(mcclAllReduce(in,out,n,mcclBfloat16,mcclSum,comm,stream));MC(mcStreamSynchronize(stream));
  ready.fetch_add(1);while(ready.load()!=4)std::this_thread::yield();
  mcEvent_t a,b;MC(mcEventCreate(&a));MC(mcEventCreate(&b));auto start=std::chrono::steady_clock::now();MC(mcEventRecord(a,stream));
  for(int i=0;i<iters;i++)CC(mcclAllReduce(in,out,n,mcclBfloat16,mcclSum,comm,stream));MC(mcEventRecord(b,stream));MC(mcEventSynchronize(b));float ms;MC(mcEventElapsedTime(&ms,a,b));
  double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/iters;
  MC(mcMemcpy(h.data(),out,n*2,mcMemcpyDeviceToHost));for(auto x:h)if(x!=0x4080){fprintf(stderr,"incorrect reduction rank %d: %x\n",rank,x);std::abort();}
  printf("{\"rank\":%d,\"bytes\":%zu,\"iterations\":%d,\"gpu_us\":%.3f,\"wall_us\":%.3f,\"correct\":true}\n",rank,n*2,iters,ms*1000/iters,wall);
  CC(mcclCommDestroy(comm));MC(mcEventDestroy(a));MC(mcEventDestroy(b));MC(mcFree(in));MC(mcFree(out));MC(mcStreamDestroy(stream));
 });for(auto &t:workers)t.join();
}
