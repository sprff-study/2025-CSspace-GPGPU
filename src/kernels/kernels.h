#pragma once

#include "../wrappers.h"
#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void prefixsum_pre(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 b, gpuptr::u32 c, unsigned int n);
void prefixsum_main(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 c, unsigned int n);
void prefixsum_post(const gpu::WorkSize& workSize, gpuptr::u32 b, gpuptr::u32 c, unsigned int n);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}
