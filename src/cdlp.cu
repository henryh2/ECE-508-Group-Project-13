#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <cmath>
#include <thrust/sort.h>

#include "cdlp.hu"

#define BLOCK_SIZE 512

typedef uint32_t u32;

template <typename datatype>
__device__ static void selection_sort(datatype *data, size_t left, size_t right)
{
    for( size_t i = left ; i < right ; ++i ){
        datatype min_val = data[i];
        size_t min_idx = i;

        for(size_t j = i+1 ; j < right ; ++j ){
            datatype val_j = data[j];
            if( val_j < min_val ){
                min_idx = j;
                min_val = val_j;
            }
        }

        if( i != min_idx ){
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

template<typename datatype>
__device__ static datatype find_most_frequent_item(datatype * data,size_t size){
    /// This assumes a minumum array size of 1
    datatype max_item = data[0];
    size_t max_count = 0;
    datatype current =data[0];
    size_t current_count = 1;
    for(size_t i=1;i<size;i++){
        datatype next = data[i];
        if(next!=current){
            if(current_count>max_count){
                max_count = current_count;
                max_item = current;
            }
            current_count = 1;
            current = next;
        }else{
            current_count++;
            current = next;
        }
    }
    if(current_count>max_count){
        max_item = current;
    }
    return max_item;
}

template <typename datatype>
__global__ static void CDLP_gpu( const datatype * labels, //!< per-edge triangle counts
                                 datatype * labels_after,
                                 datatype * neighbor_space, //space in global memory used to hold labels of all neighbors per node
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numNodes                  //!< how many edges to count triangles for
) {
    uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x<numNodes){
        int neighbor_count = rowPtr[x+1]-rowPtr[x];
        if(neighbor_count==0){
            labels_after[x] = labels[x];
            return;
        }
        ///TODO used a lots of global memory, could optimize here
        for(int i=rowPtr[x];i<rowPtr[x+1];i++){
            neighbor_space[i] = labels[edgeDst[i]];
        }
        //TODO using selection sort right now cause it's a small list, could optimize
        //O NlogN
        if(neighbor_count>1000){
            thrust::sort(thrust::device,neighbor_space+rowPtr[x],neighbor_space+rowPtr[x]+neighbor_count);
        }else{
            selection_sort(neighbor_space+rowPtr[x],0,neighbor_count);
        }
        //O N
        datatype most_frequent = find_most_frequent_item(neighbor_space+rowPtr[x],neighbor_count);
        labels_after[x]=most_frequent;
    }
}

std::vector<uint32_t> CDLP(const pangolin::COOView<uint32_t> view,int iterations) {
  //@@ create a pangolin::Vector (uint32_t) to hold per-edge triangle counts
  // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
  // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
  cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL*1024);

  uint32_t total = view.num_rows();
  std::vector<uint32_t> labels(total);

//  std::cout<<"num of nodes "<<total<<"\n";
//  uint32_t maximum = 0;
//    for(int i=0;i<100;i++){
//        std::cout<<view.row_ptr()[i]<<",";
//    }
//    std::cout<<"num of nodes "<<total<<"\n";
//    for(int i=0;i<100;i++){
//      std::cout<<view.col_ind()[i]<<",";
//  }
//
//    uint32_t last_node = view.row_ptr()[total];
//    std::cout<<"last nodes "<<last_node<<"\n";
//    std::cout<<"none zeros"<<view.nnz()<<"\n";

    uint32_t * cuda_mem;
  uint32_t * cuda_mem2;
  uint32_t * neighbor_space;

  ///TODO we can do this in GPU
  for(uint32_t i=0;i<total;i++){
      labels[i]=i;
  }
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid (std::ceil(total/float(BLOCK_SIZE)));
  CUDA_RUNTIME(cudaMalloc((void **)&cuda_mem,
                          total * sizeof(uint32_t)));
  CUDA_RUNTIME(cudaMalloc((void **)&cuda_mem2,
                          total * sizeof(uint32_t)));
  CUDA_RUNTIME(cudaMalloc((void **)&neighbor_space,
                          view.nnz() * sizeof(uint32_t)));
  CUDA_RUNTIME(cudaMemcpy(cuda_mem, labels.data(), total * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  ///TODO I can't think of good ways to sync all threads for every iteration, so we do it on CPU side
  for(int i=0;i<iterations;i++){
      CDLP_gpu<<<dimGrid, dimBlock>>>(cuda_mem,cuda_mem2,neighbor_space,view.row_ind(),view.col_ind(),view.row_ptr(),total);
      CUDA_RUNTIME(cudaDeviceSynchronize());
      uint32_t * temp = cuda_mem;
      cuda_mem = cuda_mem2;
      cuda_mem2 = temp;
  }
  CUDA_RUNTIME(cudaMemcpy(labels.data(), cuda_mem,
                          total*sizeof(uint32_t), cudaMemcpyDeviceToHost));

  return labels;
}

//====================================Sequential Version=========================================

int label(int *labels, u32 self_label, u32 start, u32 end, const uint32_t *const edge_dst) {
  int max_label = self_label;
  int oc = 1;
  for (u32 i = start; i < end; i++) {
      if (labels[edge_dst[i]] == self_label) {
          oc += 1;
      }
  }

  for (u32 i = start; i < end; i++) {
    int curr_label = labels[edge_dst[i]];
    int curr_oc = 0;
    for (int j = start; j < end; j++) {
      if (labels[edge_dst[j]] == curr_label) {
        curr_oc += 1;
      }
    }
    if (curr_oc > oc) {
      max_label = curr_label;
      oc = curr_oc;
    } else if (curr_oc == oc && curr_label < max_label) {
      max_label = curr_label;
    } else {
      continue;
    }
  }
  return max_label;
}

//graph: graph adjacency matrix
int* cldp(const pangolin::COOView<uint32_t> view, int num_iterations) {
  size_t num_nodes = view.num_rows();
  int labels[num_nodes];
  auto edge_dst = view.col_ind();
  auto row_ptr = view.row_ptr();

  for (int i = 0; i < num_nodes; i++) {
    labels[i] = i + 1; //assume nodes are labeled 1-N
  }
  for (int c = 0; c < num_iterations; c++) {
    for (int i = 0; i < num_nodes; i++) {
      u32 start = row_ptr[i];
      u32 end = row_ptr[i + 1];
      int new_label = label(labels, labels[i], start, end, edge_dst);
      labels[i] = new_label;
    }
  }
  return labels;
}
