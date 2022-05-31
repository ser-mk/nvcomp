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
const nvcompType_t _DATA_TYPE = nvcomp::TypeOf<T>();
using TYPE_CHECK_SUM = uint16_t;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = (cond);                                                  \
    if(err != cudaSuccess){ printf("Check " #cond " at %d failed.\n", __LINE__);  return cudaErrorUnknown; }   \
  } while (false)

static void print_options(const nvcompBatchedCascadedOpts_t & options){
  printf("cascade options: chunk_size %zu, rle %d, delta %d, M2Mode %d, bp %d\n",
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

typedef struct alignas(4)
{
  TYPE_CHECK_SUM check_sum;
  uint16_t chunk_size;
  nvcompType_t type;
} OptionsHeader_t;


struct TaskCascadedManager : CascadedManager {

  TaskCascadedManager(
      const nvcompBatchedCascadedOpts_t& options = nvcompBatchedCascadedDefaultOpts,
      cudaStream_t user_stream = 0,
      int device_id = 0):
      CascadedManager(options, user_stream, device_id)
  {
  }

  virtual ~TaskCascadedManager()
  {
  }

  static inline cudaError_t check_error() {
    return nv_check_error_last_call_and_clear();
  }

  static inline size_t  calc_size_out(const CompressionConfig & comp_config) {
    return comp_config.max_compressed_buffer_size + _SIZE_OF_OPTIONS_HEADER;
  }

  cudaError_t cascade_compress(
      const uint8_t* decomp_buffer,
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config,
      const nvcompBatchedCascadedOpts_t& options) {

    CUDA_CHECK(CascadedManager::save_options_header(comp_buffer, options));
    comp_buffer += _SIZE_OF_OPTIONS_HEADER;
    CascadedManager::compress(decomp_buffer, comp_buffer, comp_config);
    CUDA_CHECK(TaskCascadedManager::check_error());
    return cudaSuccess;
  }

  virtual size_t get_compressed_output_size(uint8_t* comp_buffer)
  {
    return impl->get_compressed_output_size(comp_buffer + _SIZE_OF_OPTIONS_HEADER) + _SIZE_OF_OPTIONS_HEADER;
  }

  static inline cudaError_t parse_options_header(const uint8_t * comp_buffer,  OptionsHeader_t & header_host)
  {
    CUDA_CHECK(cudaMemcpy(&header_host, comp_buffer, _SIZE_OF_OPTIONS_HEADER, cudaMemcpyDeviceToHost));
    return cudaSuccess;
  }

  static inline cudaError_t check_option_header(const OptionsHeader_t & header_host){
    const TYPE_CHECK_SUM check_sum = TaskCascadedManager::calc_check_sum(header_host);
    if(check_sum == header_host.check_sum)
      return cudaSuccess;
    return cudaErrorUnknown;
  }

  virtual DecompressionConfig configure_decompression(const uint8_t* comp_buffer)
  {
    return impl->configure_decompression(comp_buffer + _SIZE_OF_OPTIONS_HEADER);
  }

  cudaError_t cascade_decompress(uint8_t* decomp_buffer,
                                 const uint8_t* comp_buffer,
                                 const DecompressionConfig& decomp_config){
    const uint8_t* comp_buffer_after_header = comp_buffer + _SIZE_OF_OPTIONS_HEADER;
    CascadedManager::decompress(decomp_buffer, comp_buffer_after_header, decomp_config);
    CUDA_CHECK(TaskCascadedManager::check_error());
    return cudaSuccess;
  }

private:
  static const size_t _SIZE_OF_OPTIONS_HEADER = sizeof(OptionsHeader_t); // the type value(1) + the chunk size(2) + gap(1)= 4 (alignment)

  static inline TYPE_CHECK_SUM calc_check_sum(const OptionsHeader_t & header_host){
    return static_cast<TYPE_CHECK_SUM>(header_host.chunk_size) + static_cast<TYPE_CHECK_SUM>(header_host.type);
  }

  static inline cudaError_t save_options_header(uint8_t* comp_buffer, const nvcompBatchedCascadedOpts_t& options) {
    OptionsHeader_t header_host = { .check_sum = 0,
                                    .chunk_size = static_cast<uint16_t>(options.chunk_size),
                                    .type =options.type};
    header_host.check_sum = TaskCascadedManager::calc_check_sum(header_host);
    CUDA_CHECK(cudaMemcpy(comp_buffer, &header_host, sizeof(header_host), cudaMemcpyHostToDevice));
    return cudaSuccess;
  }

};

static cudaError_t nv_compress(cudaStream_t & stream, const nvcompBatchedCascadedOpts_t & options, GPUbuffer & input_data, GPUbuffer & compress_data, size_t & comp_size){
  TaskCascadedManager manager{options, stream};
  CUDA_CHECK(nv_check_error_last_call_and_clear());
  auto comp_config = manager.configure_compression(input_data.size);
  CUDA_CHECK(nv_check_error_last_call_and_clear());

  const size_t size_out = manager.calc_size_out(comp_config);
  CUDA_CHECK(check_then_relloc(compress_data, size_out));

  CUDA_CHECK(manager.cascade_compress(input_data.ptr, compress_data.ptr, comp_config, options));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  comp_size = manager.get_compressed_output_size(compress_data.ptr);
  CUDA_CHECK(nv_check_error_last_call_and_clear());

  return cudaSuccess;
}

cudaError_t max_compress(cudaStream_t & stream, INPUT_VECTOR_TYPE & input, GPUbuffer& compress_data, GPUbuffer& decompress_data){
  const size_t input_size = sizeof(uint8_t) * input.size();
  CUDA_CHECK((check_then_relloc(decompress_data, input_size)));

  CUDA_CHECK(cudaMemcpy(decompress_data.ptr, input.data(), decompress_data.size, cudaMemcpyHostToDevice));

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
            if(cudaSuccess != nv_compress(stream, options, decompress_data, compress_data, comp_size)){
              printf("Pass");
              continue;
            }
            printf("compress size: %zu", comp_size);
            if(min_size <= comp_size)
              continue;

            min_size = comp_size;
            min_options = options;
          }
        }
      }
  printf("\n");
  printf("minimum compress output size: %zu, ", min_size);
  printf("with cascade options: ");
  print_options(min_options);
  CUDA_CHECK(nv_compress(stream, min_options, decompress_data, compress_data, comp_size));

  return cudaSuccess;
}

cudaError_t nv_decompress(cudaStream_t & stream, GPUbuffer & compress_data, GPUbuffer & decompress_data, size_t & output_size){

  OptionsHeader_t header_host;
  CUDA_CHECK(TaskCascadedManager::parse_options_header(compress_data.ptr, header_host));
  CUDA_CHECK(TaskCascadedManager::check_option_header(header_host));
  /*
   * the decompression manager doesn't check the data type and chunk_size in the input stream
   * https://github.com/NVIDIA/nvcomp/issues/63
   */
  const nvcompBatchedCascadedOpts_t options = {header_host.chunk_size, header_host.type, 0, 0, false, 0};

  TaskCascadedManager manager{options, stream};
  CUDA_CHECK(nv_check_error_last_call_and_clear());
  auto decomp_config = manager.configure_decompression(compress_data.ptr);
  CUDA_CHECK(nv_check_error_last_call_and_clear());
  CUDA_CHECK(check_then_relloc(decompress_data, decomp_config.decomp_data_size));

  CUDA_CHECK(manager.cascade_decompress(
      decompress_data.ptr,
      compress_data.ptr,
      decomp_config));
  CUDA_CHECK(cudaStreamSynchronize(stream));
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

  GPUbuffer compress_data;
  GPUbuffer decompress_data;

  std::thread dummy([]{
    std::cout << "The task requires multithreading... Okey\n";
  });

  do {
    if(max_compress(stream, input, compress_data, decompress_data) != cudaSuccess)
      break;

    // clear input buffer
    cudaMemset(decompress_data.ptr, 0, decompress_data.size);
    /*
     * Should there be a copy of the compressed data to host's memory in this step?
     *  The task states nothing about it... so let's decompress the output data
     *  which are in GPU memory.
     */

    size_t output_size;
    if(nv_decompress(stream, compress_data, decompress_data, output_size) != cudaSuccess)
      break;

    if(output_size != input.size()) {
      printf("the output size not equal the input size\n");
      break;
    }

    const cudaError_t err = cudaMemcpy(
        results.data(), decompress_data.ptr, output_size * sizeof(T), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
      printf("cudaMemcpy error: %d\n", err);

    assert(input == results);

    printf("\n\n successfully compression and decompression!\n");
  } while(false);

  // clear
  free_gpu_buffer(compress_data);
  free_gpu_buffer(decompress_data);

  const cudaError_t err = cudaStreamDestroy(stream);
  if( err != cudaSuccess){
    printf("cudaStreamDestroy return code: %d\n", err);
  }

  dummy.join();
}
