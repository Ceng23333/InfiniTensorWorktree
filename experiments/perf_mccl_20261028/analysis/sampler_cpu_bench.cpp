#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
static inline float bf16_to_float(uint16_t h) { uint32_t u=uint32_t(h)<<16;float f;std::memcpy(&f,&u,4);return f; }
static int sample(const std::vector<uint16_t>& raw) {
 std::vector<float> out(raw.size());
 for(size_t i=0;i<raw.size();++i)out[i]=bf16_to_float(raw[i]);
 return int(std::distance(out.begin(),std::max_element(out.begin(),out.end())));
}
int main(){std::vector<uint16_t> raw(248320);for(size_t i=0;i<raw.size();++i)raw[i]=0x3f00+(i%128);std::vector<double> timings;volatile int sink=0;
 for(int i=0;i<60;++i){auto t=std::chrono::steady_clock::now();sink=sample(raw);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();if(i>=10)timings.push_back(ms);}
 std::sort(timings.begin(),timings.end());printf("{\"vocab\":248320,\"median_ms\":%.6f,\"p95_ms\":%.6f,\"argmax\":%d,\"excludes_device_copy\":true}\n",timings[25],timings[47],int(sink));}
