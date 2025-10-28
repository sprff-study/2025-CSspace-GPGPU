#pragma once

#include "kernels/kernels.h"

void prefixSum(gpu::gpu_mem_32u& in, gpu::gpu_mem_32u& out, gpu::gpu_mem_32u& buffer);