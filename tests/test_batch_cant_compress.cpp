#include "../src/common.h"
#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

using run_type = uint16_t;
using nvcomp::roundUpToAlignment;

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      return 0;                                                                \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

size_t max_compressed_size(size_t uncompressed_size)
{
  return (uncompressed_size + 3) / 4 * 4 + 4;
}

template <typename data_type>
size_t test_predefined_cases(int rle, int delta, int bp)
{
  std::vector<data_type> input0_host;

  for(int i=-120; i<120; i++)
    input0_host.push_back(i);

  void* input0_device;
  CUDA_CHECK(
      cudaMalloc(&input0_device, input0_host.size() * sizeof(data_type)));
  CUDA_CHECK(cudaMemcpy(
      input0_device,
      input0_host.data(),
      input0_host.size() * sizeof(data_type),
      cudaMemcpyHostToDevice));

  printf("input data(size:%zu) : ", input0_host.size());
  for(auto el: input0_host)
    printf("%d:", el);

  printf("\n");

  // Copy uncompressed pointers and sizes to device memory

  std::vector<void*> uncompressed_ptrs_host
      = {input0_device};
  std::vector<size_t> uncompressed_bytes_host
      = {input0_host.size() * sizeof(data_type)
      };
  const size_t batch_size = uncompressed_ptrs_host.size();

  void** uncompressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&uncompressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_ptrs_device,
      uncompressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* uncompressed_bytes_device;
  CUDA_CHECK(
      cudaMalloc(&uncompressed_bytes_device, sizeof(size_t) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_bytes_device,
      uncompressed_bytes_host.data(),
      sizeof(size_t) * batch_size,
      cudaMemcpyHostToDevice));

  // Allocate compressed buffers and sizes

  std::vector<void*> compressed_ptrs_host;
  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    void* compressed_ptr;
    CUDA_CHECK(cudaMalloc(
        &compressed_ptr,
        max_compressed_size(uncompressed_bytes_host[partition_idx])*4));
    compressed_ptrs_host.push_back(compressed_ptr);
  }

  void** compressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&compressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      compressed_ptrs_device,
      compressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* compressed_bytes_device;
  CUDA_CHECK(cudaMalloc(&compressed_bytes_device, sizeof(size_t) * batch_size));

  // Launch batched compression

  nvcompBatchedCascadedOpts_t comp_opts
      = {batch_size, nvcomp::TypeOf<data_type>(), rle, delta, bp};

  auto status = nvcompBatchedCascadedCompressAsync(
      uncompressed_ptrs_device,
      uncompressed_bytes_device,
      0, // not used
      batch_size,
      nullptr, // not used
      0,       // not used
      compressed_ptrs_device,
      compressed_bytes_device,
      comp_opts,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  // Verify compressed bytes alignment

  std::vector<size_t> compressed_bytes_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
      compressed_bytes_host.data(),
      compressed_bytes_device,
      sizeof(size_t) * batch_size,
      cudaMemcpyDeviceToHost));

  for (auto const& compressed_bytes_partition : compressed_bytes_host) {
    REQUIRE(compressed_bytes_partition % 4 == 0);
    REQUIRE(compressed_bytes_partition % sizeof(data_type) == 0);
  }

  // Check the test case is small enough to fit inside one batch
  constexpr size_t chunk_size = 4096;
  for (auto const& uncompressed_byte : uncompressed_bytes_host) {
    REQUIRE(uncompressed_byte <= chunk_size);
  }

  for(int i=0; i < batch_size; i++) {
    size_t _size = compressed_bytes_host[i];
    printf("output compressed data(size:%zu): ", _size);

    std::vector<data_type> compressed_data_host(_size);
    CUDA_CHECK(cudaMemcpy(
        compressed_data_host.data(),
        compressed_ptrs_host[i],
        _size - 0,
        cudaMemcpyDeviceToHost));

    for (auto el : compressed_data_host) {
      printf("%d:", el);
    }
    printf("\n");
  }
  // Cleanup

  CUDA_CHECK(cudaFree(input0_device));
  CUDA_CHECK(cudaFree(uncompressed_ptrs_device));
  CUDA_CHECK(cudaFree(uncompressed_bytes_device));
  for (void* const& ptr : compressed_ptrs_host)
    CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaFree(compressed_ptrs_device));
  CUDA_CHECK(cudaFree(compressed_bytes_device));


  return compressed_bytes_host[0];
}

int main()
{
  size_t size;
  printf("Delta option - no modify:\n");
  int rle = 0; int delta = 1; int bp = 0;
  size = test_predefined_cases<int8_t>(rle, delta, bp);
  printf("result compressed size: %zu\n", size);
  printf("\n----------------------------------------------------------\n");

  printf("RLE option - no modify:\n");
  int rle = 1; int delta = 0; int bp = 0;
  size = test_predefined_cases<int8_t>(rle, delta, bp);
  printf("result compressed size: %zu\n", size);
  printf("\n----------------------------------------------------------\n");

  printf("RLE + BP option - no modify:\n");
  int rle = 1; int delta = 0; int bp = 1;
  size = test_predefined_cases<int8_t>(rle, delta, bp);
  printf("result compressed size: %zu\n", size);
  printf("\n----------------------------------------------------------\n");

  printf("BP option - no modify:\n");
  int rle = 0; int delta = 0; int bp = 1;
  size = test_predefined_cases<int8_t>(rle, delta, bp);
  printf("result compressed size: %zu\n", size);
  printf("\n----------------------------------------------------------\n");

  printf("Delta + BP option - compress! :\n");
  int rle = 0; int delta = 1; int bp = 1;
  size = test_predefined_cases<int8_t>(rle, delta, bp);
  printf("result compressed size: %zu\n", size);
  printf("\n----------------------------------------------------------\n");

}
