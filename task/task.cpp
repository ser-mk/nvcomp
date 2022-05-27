#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

#include <assert.h>
#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <thread>


namespace nvcomp
{
cudaError_t nv_check_error_last_call_and_clear();
}

using namespace nvcomp;

using T = uint8_t;
using INPUT_VECTOR_TYPE = const std::vector<T>;

const nvcompType_t _DATA_TYPE = NVCOMP_TYPE_UCHAR;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = (cond);                                                  \
    if(err != cudaSuccess){ printf("Check " #cond " at %d failed.\n", __LINE__);  return cudaErrorUnknown; }   \
  } while (false)

static void print_options(const nvcompBatchedCascadedOpts_t & options){
  printf("chunk_size %zu, rle %d, delta %d, M2Mode %d, bp %d\n",
         options.chunk_size, options.num_RLEs, options.num_deltas, options.is_m2_deltas_mode, options.use_bp);
}


struct GPUbuffer {
  uint8_t* ptr;
  size_t size = 0;
};


static cudaError_t check_then_relloc(GPUbuffer& buffer, const size_t relloc_size){
  if(buffer.size >= relloc_size) // only up
    return cudaSuccess;
  uint8_t* tmp;
  if (cudaSuccess != cudaMalloc(&tmp, relloc_size)) {
    printf("can't cudaMalloc for GPU buffer %zu\n", relloc_size);
    return cudaErrorUnknown;
  }
  if(buffer.size > 0)
    if (cudaSuccess != cudaFree(buffer.ptr))
      printf("cudaFree call failed: (size) %zu\n", buffer.size);
  buffer.ptr = tmp;
  buffer.size = relloc_size;
  return cudaSuccess;
}

static void free_gpu_buffer(GPUbuffer& buf){
  if(buf.size == 0)  // avoid free invalid ref
    return;
  const cudaError_t err = cudaFree(buf.ptr);
  if (cudaSuccess != err)
    printf("cudaFree return code %d ( %zu bytes size)\n", err, buf.size);
  buf.size = 0;
}

static cudaError_t nv_compress(cudaStream_t & stream, const nvcompBatchedCascadedOpts_t & options, GPUbuffer & input_data, GPUbuffer & compress_data, size_t & comp_size){
  CascadedManager manager{options, stream};
  CUDA_CHECK(nv_check_error_last_call_and_clear());
  auto comp_config = manager.configure_compression(input_data.size);
  CUDA_CHECK(nv_check_error_last_call_and_clear());
  const size_t size_out = comp_config.max_compressed_buffer_size;
  CUDA_CHECK(check_then_relloc(compress_data, size_out));

  manager.compress(input_data.ptr, compress_data.ptr, comp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(nv_check_error_last_call_and_clear());
  const nvcompStatus_t status = *comp_config.get_status();
  if(status != nvcompSuccess){
    printf("cascade compress status: %d\n", status);
  }
  comp_size = manager.get_compressed_output_size(compress_data.ptr);
  CUDA_CHECK(nv_check_error_last_call_and_clear());

  return cudaSuccess;
}

cudaError_t max_compress(cudaStream_t & stream, INPUT_VECTOR_TYPE & input, GPUbuffer& compress_data, GPUbuffer& tmp){
  const size_t input_size = sizeof(uint8_t) * input.size();
  CUDA_CHECK((check_then_relloc(tmp, input_size)));

  CUDA_CHECK(cudaMemcpy(tmp.ptr, input.data(), tmp.size, cudaMemcpyHostToDevice));

  size_t min_size = SIZE_MAX;
  size_t comp_size = 0;
  nvcompBatchedCascadedOpts_t min_options = {4096, NVCOMP_TYPE_UCHAR, 0, 0, false, 1};

  // find max compressing scheme
  for(size_t chunk_size = 16384; chunk_size >= 512; chunk_size -= 512)
    for(int rle = 0; rle <= 3; rle++)
      for(int bp = 0; bp <= 1; bp++) {
        // No delta without BitPack
        const int max_delta_num = bp == 0 ? 0 : 4;
        for (int delta = 0; delta <= max_delta_num; delta++) {
          // No delta mode without delta nums
          const int max_delta_mode = delta == 0 ? 0 : 1; // Description of mode: https://github.com/NVIDIA/nvcomp/issues/61
          for (int delta_mode = 0; delta_mode <= max_delta_mode; delta_mode++) {
            if((rle + bp + delta) == 0)
              continue;
            const nvcompBatchedCascadedOpts_t options = {chunk_size, _DATA_TYPE, rle, delta, static_cast<bool>(delta_mode), bp};
            printf("\n");
            print_options(options);
            CUDA_CHECK(nv_compress(stream, options, tmp, compress_data, comp_size));
            printf("compress size: %zu", comp_size);
            if(min_size <= comp_size)
              continue;

            min_size = comp_size;
            min_options = options;
          }
        }
      }
  printf("\n");
  printf("min compress size: %zu, ", min_size);
  printf("min_options: ");
  print_options(min_options);
  CUDA_CHECK(nv_compress(stream, min_options, tmp, compress_data, comp_size));

  return cudaSuccess;
}

cudaError_t nv_decompress(cudaStream_t & stream, GPUbuffer & compress_data, GPUbuffer & decompress_data, size_t & output_size){
  nvcompBatchedCascadedOpts_t options = nvcompBatchedCascadedDefaultOpts;
  /*
   * the decompression manager doesn't check the data type in the input stream
   * https://github.com/NVIDIA/nvcomp/issues/63
   */
  options.type = _DATA_TYPE;

  CascadedManager manager{options, stream};
  CUDA_CHECK(nv_check_error_last_call_and_clear());
  auto decomp_config = manager.configure_decompression(compress_data.ptr);
  CUDA_CHECK(nv_check_error_last_call_and_clear());
  CUDA_CHECK(check_then_relloc(decompress_data, decomp_config.decomp_data_size));

  manager.decompress(
      decompress_data.ptr,
      compress_data.ptr,
      decomp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const nvcompStatus_t status = *decomp_config.get_status();
  if(status != nvcompSuccess){
    printf("cascade decompress status: %d\n", status);
  }

  CUDA_CHECK(nv_check_error_last_call_and_clear());

  output_size = decomp_config.decomp_data_size;

  return cudaSuccess;
}


static INPUT_VECTOR_TYPE input = {
#include "data.h"
};

static std::vector<T> results(input.size());

int main()
{
  cudaStream_t stream;
  if(cudaSuccess != cudaStreamCreate(&stream)){
    stream = 0;
  }

  GPUbuffer compress_data{};
  GPUbuffer tmp{};

  std::thread dummy([]{
    std::cout << "The task requires multithreading... Okey\n";
  });

  do {
    if(max_compress(stream, input, compress_data, tmp) != cudaSuccess)
      break;

    cudaMemset(tmp.ptr, 0, tmp.size);
    /*
     * Should there be a copy of the compressed data to host's memory in this step?
     *  The task states nothing about it... so let's decompress the output data
     *  which are in GPU memory.
     */

    size_t output_size;
    if(nv_decompress(stream, compress_data, tmp, output_size) != cudaSuccess)
      break;

    if(output_size != input.size()) {
      printf("output size not equal input size\n");
      break;
    }

    const cudaError_t err = cudaMemcpy(
        &results[0], tmp.ptr, output_size * sizeof(T), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
      printf("error cudaMemcpy: %d\n", err);

    assert(input == results);

    printf("\n\n successfully compression and decompression!\n");
  } while(false);

  free_gpu_buffer(compress_data);
  free_gpu_buffer(tmp);

  const cudaError_t err = cudaStreamDestroy(stream);
  if( err != cudaSuccess){
    printf("cudaStreamDestroy return code: %d\n", err);
  }

  dummy.join();
}
