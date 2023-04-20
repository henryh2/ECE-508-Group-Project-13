#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "lcc.hu"

#define CPU_MODE 0
#define GPU_MODE 1

#define BLOCK_SIZE 512

/**
 * find number of intersections using binary search
 * uPtr is linear pointer
 * vPtr is binary pointer
*/
__device__ static uint32_t binary_search(const uint32_t *const edgeDst, uint32_t uPtr, uint32_t uEnd, uint32_t vPtr, uint32_t vEnd) {
  uint32_t tc = 0;
  while(uPtr < uEnd) {
    uint32_t w1 = edgeDst[uPtr];
    int left = vPtr;
    int right = vEnd;

    while(left < right) {
      int mid = left + (right - left) / 2;
      // int mid = (right - left) / 2;
      uint32_t w2 = edgeDst[mid];
      if(w1 > w2) {
        left = mid + 1;
      }
      else if(w1 < w2) {
        right = mid;
      }
      else {
        tc++;
        break;
      }
    }
    uPtr ++;
  }

  return tc;
}

/**
 * find number of intersections using lienar search
*/
__device__ static uint32_t linear_search(const uint32_t *const edgeDst, uint32_t uPtr, uint32_t uEnd, uint32_t vPtr, uint32_t vEnd) {
  uint32_t tc = 0;
  while(uPtr < uEnd && vPtr < vEnd) {
    uint32_t w1 = edgeDst[uPtr];
    uint32_t w2 = edgeDst[vPtr];
    if(w1 < w2) {
      uPtr++;
    }
    else if(w1 > w2) {
      vPtr++;
    }
    else {
      uPtr++;
      vPtr++;
      tc ++;
    }
  }
  return tc;
}

__global__ static void triangle_count_kernel(uint32_t *__restrict__ triangleCounts, //!< per-node triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  // The source node number
  int nodeNum = edgeSrc[tx];
  if(tx < numEdges) {
    uint32_t uPtr = rowPtr[nodeNum];
    uint32_t uEnd = rowPtr[nodeNum + 1];
    uint32_t vPtr = rowPtr[edgeDst[tx]];
    uint32_t vEnd = rowPtr[edgeDst[tx] + 1];

    uint32_t uDiff = uEnd - uPtr;
    uint32_t vDiff = vEnd - vPtr;

    // From triangle counting lab
    // using binary search when V was as least 64 and V/U was at least 6 (V is the longer list length, and U the shorter one).
    if (uDiff > vDiff && uDiff >= 64 && uDiff / vDiff >= 6) {
      // One node may have many edges, use atomic add
      uint32_t x = binary_search(edgeDst, vPtr, vEnd, uPtr, uEnd);
      atomicAdd(&triangleCounts[nodeNum], x);
    }
    else if(vDiff > uDiff && vDiff >= 64 && vDiff / uDiff >= 6) {
      uint32_t x = binary_search(edgeDst, vPtr, vEnd, uPtr, uEnd);
      atomicAdd(&triangleCounts[nodeNum], x);
    }
    else{
      uint32_t x = linear_search(edgeDst, vPtr, vEnd, uPtr, uEnd);
      atomicAdd(&triangleCounts[nodeNum], x);
    }
  }
  __syncthreads();
    
}


__global__ static void coefficients_calculate_kernel(uint32_t *__restrict__ triangleCounts,     // per-node triangle counts
                                                    const uint32_t *const rowPtr,               // source node offsets in edgeDst
                                                    float *coefficients,                        // coefficients
                                                    int numNodes                                // number of nodes
) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  if(tx < numNodes) {
    int outEdge = rowPtr[tx + 1] - rowPtr[tx];
    // If the node has at least two neighbors, calculate the coefficient. Otherwise the coefficient is 0
    if(outEdge > 1) {
      coefficients[tx] = 2 * triangleCounts[tx] / (outEdge * (outEdge - 1));
    }
  }
}

std::vector<uint32_t> LCC(const pangolin::COOView<uint32_t> view, int numNodes) 
{
  dim3 dimBlock(BLOCK_SIZE);
  // calculate the number of blocks needed
  dim3 dimGridCount(ceil(view.nnz() * 1.0 / BLOCK_SIZE));

  // Store triangle counts for each node 
  uint32_t *triangleCounts;
  uint32_t* coefficients = (uint32_t*)malloc(numNodes * sizeof(uint32_t));
  cudaMalloc((void**) (&triangleCounts), numNodes * sizeof(uint32_t));
  // Kernel coefficients array
  float *kernel_coefficents;
  cudaMalloc((void**) (&kernel_coefficents), numNodes * sizeof(float));

  triangle_count_kernel<<<dimGridCount, dimBlock>>>(triangleCounts, view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());

  // One thread calculate the coefficient for one node
  dim3 dimGridCoefficient(numNodes * 1.0 / BLOCK_SIZE);
  // launch another kernal to compute llc
  coefficients_calculate_kernel<<<dimGridCoefficient, dimBlock>>>(triangleCounts, view.row_ptr(), kernel_coefficents, numNodes);

  cudaMemcpy(coefficients, kernel_coefficents, sizeof(float) * numNodes, cudaMemcpyDeviceToHost);

  cudaFree(kernel_coefficents);
  cudaFree(triangleCounts);

  std::vector<uint32_t> coef;
  for(int i = 0; i < numNodes; ++i)
    coef.push_back((uint32_t)coefficients[i]);
  return coef;
}