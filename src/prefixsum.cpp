#include "prefixsum.h"

#include "kernels/defines.h"

void calcPrefixSum(
    gpu::gpu_mem_32u& a, unsigned int abase,
    gpu::gpu_mem_32u& b, unsigned int bbase,
    gpu::gpu_mem_32u& c, unsigned int cbase,
    unsigned int n,
    unsigned int depth)
{
    int bsz = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    if (n <= GROUP_SIZE) { // trivial
        cuda::prefixsum_main(gpu::WorkSize(GROUP_SIZE, n), a, abase, c, cbase, n);
        return;
    }
    cuda::prefixsum_pre(gpu::WorkSize(GROUP_SIZE, n), a, abase, b, bbase, c, cbase, n);
    // need to calculate pref for b[bbase;bbase+bsz)
    calcPrefixSum(
        b, bbase, // start is b[bbase]
        b, bbase + bsz, // buffer starts at b[bbase+bsz]
        b, bbase, // calulate prefsum inplace
        bsz, depth + 1);
    cuda::prefixsum_post(gpu::WorkSize(GROUP_SIZE, n), b, bbase, c, cbase, n);
}

void prefixSum(gpu::gpu_mem_32u& in, gpu::gpu_mem_32u& out, gpu::gpu_mem_32u& buffer)
{
    calcPrefixSum(in, 0, buffer, 0, out, 0, in.number(), 0);
}