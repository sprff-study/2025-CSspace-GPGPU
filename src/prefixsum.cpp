#include "prefixsum.h"

#include "kernels/defines.h"
#include "wrappers.h"

void calcPrefixSum(
    gpuptr::u32 a,
    gpuptr::u32 b,
    gpuptr::u32 c,
    unsigned int n,
    unsigned int depth)
{
    int bsz = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    if (n <= GROUP_SIZE) { // trivial
        cuda::prefixsum_main(gpu::WorkSize(GROUP_SIZE, n), a, c, n);
        return;
    }
    cuda::prefixsum_pre(gpu::WorkSize(GROUP_SIZE, n), a, b, c, n);
    // need to calculate pref for b[bbase;bbase+bsz)
    calcPrefixSum(
        b, // start is b[bbase]
        b + bsz, // buffer starts at b[bbase+bsz]
        b, // calulate prefsum inplace
        bsz, depth + 1);
    cuda::prefixsum_post(gpu::WorkSize(GROUP_SIZE, n), b, c, n);
}

void prefixSum(gpu::gpu_mem_32u& in, gpu::gpu_mem_32u& out, gpu::gpu_mem_32u& buffer)
{
    calcPrefixSum(gpuptr::u32(in, 0), gpuptr::u32(buffer, 0), gpuptr::u32(out, 0), in.number(), 0);
}