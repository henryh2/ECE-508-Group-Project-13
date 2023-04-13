#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define CPU_MODE 0
#define GPU_MODE 1

#define BLOCK_SIZE 512

void local_clustering_coefficent(const pangolin::COOView<uint32_t> view, uint64_t* coefficents)
{
  // Compute local clustering coefficent by finding outward degree and triangles which include node
}
