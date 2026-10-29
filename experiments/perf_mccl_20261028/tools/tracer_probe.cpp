#include <mcr/mc_runtime.h>
#include <thread>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <initializer_list>
int main(int argc,char**){mcSetDevice(0);void*p=nullptr;mcMalloc(&p,1024);mcMemset(p,0,1024);mcDeviceSynchronize();for (const char*n : {"TRACER_RECORD_NUM","MACA_DBG_TRACER_RECORD_NUM","HOTSPOT_BUFFER_SIZE"}) { auto p=dlsym(RTLD_DEFAULT,n); if(p)printf("%s=%llu\n",n,*(unsigned long long*)p); } puts("probe ready");fflush(stdout);std::this_thread::sleep_for(std::chrono::seconds(argc>1?0:20));mcMemset(p,0,1024);mcDeviceSynchronize();mcFree(p);puts("probe done");}
