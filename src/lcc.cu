#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "lcc.hu"

#define CPU_MODE 0
#define GPU_MODE 1

#define BLOCK_SIZE 512

typedef uint32_t u32;

/**
 * find number of intersections using binary search
 * uPtr is linear pointer
 * vPtr is binary pointer
*/
__global__ static int binary_search_kernel(u32 start, u32 end, u32 ts, u32 tend, u32 src, u32 dst, const uint32_t *const edgeDst, uint32_t *__restrict__ triangleCounts) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx + start >= end) {
    return;
  }
  int val = edgeDst[tx];
  while (ts <= tend) {
    int mid = (tend + ts) / 2;
    if (edgeDst[mid] == val) {
      atomicAdd(triangleCounts + val, 1);
      atomicAdd(triangleCounts + src, 1);
      atomicAdd(triangleCounts + dst, 1);
    } else if (edgeDst[mid] > val) {
      tend = mid - 1;
    } else {
      ts = mid + 1;
    }
  }
  return 0;
}

__device__ static uint32_t binary_search_and_add(const uint32_t *const edgeDst, uint32_t *triangleCount, uint32_t uPtr, uint32_t uEnd, uint32_t vPtr, uint32_t vEnd) {
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
        atomicAdd(&triangleCount[w1], 1);
        break;
      }
    }
    uPtr ++;
  }

  return tc;
}

__device__ static uint32_t binary_search_one_value(const uint32_t *const edgeDst, uint32_t uPtr, uint32_t uEnd, uint32_t vPtr, uint32_t vEnd) {
  return 0;
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

__device__ static uint32_t linear_search_and_add(const uint32_t *const edgeDst, uint32_t *triangleCount, uint32_t uPtr, uint32_t uEnd, uint32_t vPtr, uint32_t vEnd) {
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
      atomicAdd(&triangleCount[w1], 1);
    }
  }
  return tc;
}


__global__ static void triangle_count_kernel(uint32_t *__restrict__ triangleCounts, //!< per-node triangle counts
                                 uint32_t *__restrict__ nodeCounts,
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  // The source node number map to array index
  if(tx < numEdges) {
    int nodeNum = edgeSrc[tx];
    int dstNode = edgeDst[tx];
    uint32_t uptr = rowPtr[nodeNum];
    uint32_t uend = rowPtr[nodeNum + 1];
    uint32_t vptr = rowPtr[edgeDst[tx]];
    uint32_t vend = rowPtr[edgeDst[tx] + 1];

    uint32_t start, end, ts, tend;

    if (uend - uptr < vend - vptr) {
      start = uptr; end = uend; ts = vptr; tend = vend;
    } else {
      start = vptr; end = vend; ts = uptr; tend = uend;
    }

    // From triangle counting lab
    // using binary search when V was as least 64 and V/U was at least 6 (V is the longer list length, and U the shorter one).
    // if (tend - ts >= 64 && (tend - ts) / (end - start) >= 6) {
     dim3 dimBlock(BLOCK_SIZE);
     dim3 dimGridCount(ceil((tend - ts) * 1.0 / BLOCK_SIZE));
     binary_search_kernel<<<dimGridCount, dimBlock>>>(start, end, ts, tend, nodeNum, dstNode, edgeDst, triangleCounts); 
    // } else {
    //   x = linear_intersect(uptr, uend, vptr, vend, w1, w2, edgeDst);
    //   atomicAdd(&triangleCounts[nodeNum], x);
    //   atomicAdd(&triangleCounts[dstNode], x);
    // }
    atomicAdd(&nodeCounts[nodeNum], 1);
    atomicAdd(&nodeCounts[dstNode], 1);
    
    // x = binary_search_and_add(edgeDst, triangleCounts, vPtr, vEnd, uPtr, uEnd);
    // x = linear_search_and_add(edgeDst, triangleCounts, vPtr, vEnd, uPtr, uEnd);


    // printf("Thread: %u, count: %u\n", tx, x);
    // printf("Src node: %d\n", nodeNum);
    // printf("Dst node: %d\n", dstNode);

  }
  
  __syncthreads();
}


__global__ static void coefficients_calculate_kernel(uint32_t *__restrict__ triangleCounts,     // per-node triangle counts
                                                    uint32_t *__restrict__ nodeCounts,
                                                    const uint32_t *const rowPtr,               // source node offsets in edgeDst
                                                    float *coefficients,                        // coefficients
                                                    int numNodes                                // number of nodes
) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  // if(blockIdx.x == 0 && tx == 0) {
  //   for(int i = 0; i <= numNodes; i++) {
  //     printf("rowPtr %d: %d\n", i, rowPtr[i]);
  //   }
  // }
  if(tx < numNodes) {
    // int outEdge = rowPtr[tx + 1] - rowPtr[tx];
    int outEdge = nodeCounts[tx];

    // If the node has at least two neighbors, calculate the coefficient. Otherwise the coefficient is 0
    if(outEdge > 1) {
      // printf("coefficient: %f\n", 2.0 * triangleCounts[tx] / (outEdge * (outEdge - 1)));
      coefficients[tx] = 2.0 * triangleCounts[tx] / (outEdge * (outEdge - 1));
    }
  }
}

std::vector<float> LCC(const pangolin::COOView<uint32_t> view, uint32_t numNodes) 
{
  dim3 dimBlock(BLOCK_SIZE);
  // calculate the number of blocks needed
  dim3 dimGridCount(ceil(view.nnz() * 1.0 / BLOCK_SIZE));
  // Store triangle counts for each node 
  uint32_t *triangleCounts;
  uint32_t *nodeCounts;
  cudaMalloc((void**) (&triangleCounts), numNodes * sizeof(uint32_t));
  cudaMalloc((void**) (&nodeCounts), numNodes * sizeof(uint32_t));
  float* coefficients = (float*)malloc(numNodes * sizeof(float));
  
  // Kernel coefficients array
  float *kernel_coefficents;
  cudaMalloc((void**) (&kernel_coefficents), numNodes * sizeof(float));
  // triangle_count_kernel<<<dimGridCount, dimBlock>>>(triangleCounts, view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());
  triangle_count_kernel<<<dimGridCount, dimBlock>>>(triangleCounts, nodeCounts, view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());

  // One thread calculate the coefficient for one node
  dim3 dimGridCoefficient(ceil(numNodes * 1.0 / BLOCK_SIZE));
  // launch another kernal to compute lcc
  // coefficients_calculate_kernel<<<dimGridCoefficient, dimBlock>>>(triangleCounts, view.row_ptr(), kernel_coefficents, numNodes);
  coefficients_calculate_kernel<<<dimGridCoefficient, dimBlock>>>(triangleCounts, nodeCounts, view.row_ptr(), kernel_coefficents, numNodes);

  cudaMemcpy(coefficients, kernel_coefficents, sizeof(float) * numNodes, cudaMemcpyDeviceToHost); 

  cudaFree(kernel_coefficents);
  cudaFree(triangleCounts);

  std::vector<float> coef;
  for(int i = 1; i < numNodes; ++i) {
    // printf("%d: %f\n", i, coefficients[i]);
    coef.push_back(coefficients[i]);
  }
  free(coefficients);
  return coef;
}


//====================================Sequential Version=========================================

static void linear_intersect(u32 uptr, u32 uend, u32 vptr, u32 vend, u32 w1, u32 w2, 
                                 const uint32_t *const edgeDst,
                                 double *lccs,
                                 u32 src,
                                 u32 dst) {
  while (uptr < uend && vptr < vend) {
    if (w1 < w2) {
      w1 = edgeDst[++uptr];
    } else if (w1 > w2) {
      w2 = edgeDst[++vptr];
    } else {
      lccs[w1] += 1;
      lccs[src] += 1;
      lccs[dst] += 1;
      w1 = edgeDst[++uptr];
      w2 = edgeDst[++vptr];
    }
  }
}

static int binary_search(u32 val, u32 ts, u32 tend, const uint32_t *const edgeDst) {
  while (ts <= tend) {
    int mid = (tend + ts) / 2;
    if (edgeDst[mid] == val) {
      return 1;
    } else if (edgeDst[mid] > val) {
      tend = mid - 1;
    } else {
      ts = mid + 1;
    }
  }
  return 0;
}

static void binary_intersect(u32 uptr, u32 uend, u32 vptr, u32 vend, 
                                 const uint32_t *const edgeDst,
                                 uint64_t *lccs,
                                 u32 src,
                                 u32 dst) {
  uint64_t count = 0;
  for(int i = uptr; i < uend; i++) {
    int triangle = binary_search(edgeDst[i], vptr, vend - 1, edgeDst);
    count += triangle;
    lccs[edgeDst[i]] += triangle;
    lccs[src] += triangle;
    lccs[dst] += triangle;
  }
}

pangolin::Vector<double> count_triangles(const pangolin::COOView<uint32_t> view, const int directed) {

  //COOView must contain both edges (u, v) and (v, u). Only for undirected graphs

  const size_t num_edges = view.nnz();
  const size_t num_nodes = view.num_rows();

  pangolin::Vector<uint64_t> edge_counts(num_nodes);
  pangolin::Vector<double> lccs(num_nodes);
  const uint32_t *const edgeSrc = view.row_ind();
  const uint32_t *const edgeDst = view.col_ind();
  const uint32_t *const rowPtr = view.row_ptr();

  for (int i = 0; i < num_edges; i++) {
    u32 start, end, ts, tend;

    u32 uptr = rowPtr[edgeSrc[i]];
    u32 uend = rowPtr[edgeSrc[i] + 1];
    u32 vptr = rowPtr[edgeDst[i]];
    u32 vend = rowPtr[edgeDst[i] + 1];
    u32 w1 = edgeDst[uptr];
    u32 w2 = edgeDst[vptr];

    u32 node_src = edgeSrc[i];
    u32 node_dst = edgeDst[i];

    linear_intersect(uptr, uend, vptr, vend, w1, w2, edgeDst, lccs.data(), node_src, node_dst);
    edge_counts[node_src] += 1;
    edge_counts[node_dst] += 1;

  }
  for (int i = 0; i < num_nodes; i++) {
    if (edge_counts[i] <= 1 || lccs[i] == 0) {
        lccs[i] = 0;
    } else {
      // edge_counts[i] = edge_counts[i] / 2;
      // lccs[i] /= 4;
      lccs[i] = (lccs[i] * 2) / (edge_counts[i] * (edge_counts[i] - 1));
    }
  }
  return lccs;
}