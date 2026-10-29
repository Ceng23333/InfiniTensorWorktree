#include <infini/ops.h>
#include "ops/infiniops_impl.hpp"
#include <stdexcept>
#include <cstdio>
#include <vector>
template<class Op> void probe(const char* name) {
 using Device=infini::ops::Device;
 auto direct=Op::active_implementation_indices(Device::Type::kMetax);
 using Fn=std::vector<std::size_t>(*)(Device::Type);
 Fn volatile external=&Op::active_implementation_indices;
 auto indirect=external(Device::Type::kMetax);
 auto bridge=infinicore::op::infiniops::activeImplementationIndices<Op>(Device::Type::kMetax);
 if (indirect.empty() || bridge!=indirect) throw std::runtime_error("provider bridge mismatch");
 (void)infinicore::op::infiniops::defaultConfigForDevice<Op>(Device::Type::kMetax);
 printf("%s direct=%zu external=%zu bridge=%zu PASS\n",name,direct.size(),indirect.size(),bridge.size());
}
extern "C" int run_probe(){probe<infini::ops::Gemm>("Gemm");probe<infini::ops::Copy>("Copy");probe<infini::ops::FusedAddRmsNorm>("FusedAddRmsNorm");fflush(stdout);return 0;}
