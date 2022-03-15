
#include "nvcomp/cascaded.h"

#include "../src/common.h"
#include <assert.h>
#include <vector>

#define REQUIRE(a) assert(a)

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)


template <typename T>
size_t test_cascaded(
    const std::vector<T>& data,
    int numRLEs,
    int numDeltas,
    int bitPacking)
{
  const nvcompType_t type = nvcomp::TypeOf<T>();

  // these two items will be the only forms of communication between
  // compression and decompression
  void* d_comp_out;
  size_t comp_out_bytes;

  // create GPU only input buffer
  void* d_in_data;
  const size_t in_bytes = sizeof(T) * data.size();
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, data.data(), in_bytes, cudaMemcpyHostToDevice));

  nvcompCascadedFormatOpts comp_opts;
  comp_opts.num_RLEs = numRLEs;
  comp_opts.num_deltas = numDeltas;
  comp_opts.use_bp = bitPacking;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompStatus_t status;

  // Compress on the GPU
  size_t comp_temp_bytes;
  size_t metadata_bytes;

  status = nvcompCascadedCompressConfigure(
      &comp_opts,
      nvcomp::TypeOf<T>(),
      in_bytes,
      &metadata_bytes,
      &comp_temp_bytes,
      &comp_out_bytes);

  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  status = nvcompCascadedCompressAsync(
      &comp_opts,
      nvcomp::TypeOf<T>(),
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      &comp_out_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  cudaFree(d_in_data);
  cudaStreamDestroy(stream);

  return comp_out_bytes;
}

int main(){

  size_t size;
  {
    std::vector<int8_t> input;
    for(int i=0; i<32; i++)
      input.push_back(i);
    input[3] = 1;
    size = test_cascaded<int8_t>(input, 0, 1, 1);
    printf("int8_t compress size: %zu\n", size);
  }

  {
    std::vector<uint8_t> input;
    for(int i=0; i<32; i++)
      input.push_back(i);
    input[3] = 1;
    size = test_cascaded<uint8_t>(input, 0, 1, 1);
    printf("uint8_t compress size: %zu\n", size);
  }
}