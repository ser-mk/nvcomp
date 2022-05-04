/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define CATCH_CONFIG_MAIN

#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

#include "catch.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <limits>

// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace nvcomp;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = (cond);                                                  \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
std::vector<T> buildRuns(const size_t numRuns, const size_t runSize)
{
  std::vector<T> input;
  for (size_t i = 0; i < numRuns; i++) {
    for (size_t j = 0; j < runSize; j++) {
      input.push_back(static_cast<T>(i));
    }
  }

  return input;
}

template <typename T>
void test_cascaded(const std::vector<T>& input, nvcompType_t data_type)
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompBatchedCascadedOpts_t options = nvcompBatchedCascadedDefaultOpts;
  options.type = data_type;
  CascadedManager manager{options, stream};
  auto comp_config = manager.configure_compression(in_bytes);

  // Allocate output buffer
  uint8_t* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

  manager.compress(
      reinterpret_cast<const uint8_t*>(d_in_data),
      d_comp_out,
      comp_config);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t comp_out_bytes = manager.get_compressed_output_size(d_comp_out);

  cudaFree(d_in_data);

  // Test to make sure copying the compressed file is ok
  uint8_t* copied = 0;
  CUDA_CHECK(cudaMalloc(&copied, comp_out_bytes));
  CUDA_CHECK(
      cudaMemcpy(copied, d_comp_out, comp_out_bytes, cudaMemcpyDeviceToDevice));
  cudaFree(d_comp_out);
  d_comp_out = copied;

  auto decomp_config = manager.configure_decompression(d_comp_out);

  T* out_ptr;
  cudaMalloc(&out_ptr, decomp_config.decomp_data_size);

  // make sure the data won't match input if not written to, so we can verify
  // correctness
  cudaMemset(out_ptr, 0, decomp_config.decomp_data_size);

  manager.decompress(
      reinterpret_cast<uint8_t*>(out_ptr),
      d_comp_out,
      decomp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy result back to host
  std::vector<T> res(input.size());
  cudaMemcpy(
      &res[0], out_ptr, input.size() * sizeof(T), cudaMemcpyDeviceToHost);

  // Verify correctness
  REQUIRE(res == input);

  cudaFree(d_comp_out);
  cudaFree(out_ptr);
}

} // namespace

/******************************************************************************
 * UNIT TESTS *****************************************************************
 *****************************************************************************/

TEST_CASE("comp/decomp cascaded-small", "[nvcomp]")
{
  using T = int;

  std::vector<T> input = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 2, 3, 3};

  test_cascaded(input, NVCOMP_TYPE_INT);
}

TEST_CASE("comp/decomp cascaded-1", "[nvcomp]")
{
  using T = int;

  const int num_elems = 500;
  std::vector<T> input;
  for (int i = 0; i < num_elems; ++i) {
    input.push_back(i >> 2);
  }

  test_cascaded(input, NVCOMP_TYPE_INT);
}

TEST_CASE("comp/decomp cascaded-all-small-sizes", "[nvcomp][small]")
{
  using T = uint8_t;

  for (int total = 1; total < 4096; ++total) {
    std::vector<T> input = buildRuns<T>(total, 1);
    test_cascaded(input, NVCOMP_TYPE_UCHAR);
  }
}

TEST_CASE("comp/decomp cascaded-multichunk", "[nvcomp][large]")
{
  using T = int;

  for (int total = 10; total < (1 << 24); total = total * 2 + 7) {
    std::vector<T> input = buildRuns<T>(total, 10);
    test_cascaded(input, NVCOMP_TYPE_INT);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint8", "[nvcomp][small]")
{
  using T = uint8_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, NVCOMP_TYPE_UCHAR);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint16", "[nvcomp][small]")
{
  using T = uint16_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, NVCOMP_TYPE_USHORT);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint32", "[nvcomp][small]")
{
  using T = uint32_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, NVCOMP_TYPE_UINT);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint64", "[nvcomp][small]")
{
  using T = uint64_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, NVCOMP_TYPE_ULONGLONG);
  }
}

TEST_CASE("comp/decomp cascaded-none-aligned-sizes", "[nvcomp][small]")
{
  std::vector<size_t> input_sizes = { 1, 33, 1021 };

  std::vector<nvcompType_t> data_types = {
    NVCOMP_TYPE_CHAR,
    NVCOMP_TYPE_SHORT,
    NVCOMP_TYPE_INT,
    NVCOMP_TYPE_LONGLONG,
  };
  for (auto size : input_sizes) {
    std::vector<uint8_t> input = buildRuns<uint8_t>(1, size);
    for (auto type : data_types ) {
      test_cascaded(input, type);
    }
  }
}


template <typename T>
size_t test_cascaded(const std::vector<T>& input,
                     const nvcompBatchedCascadedOpts_t opt)
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompBatchedCascadedOpts_t options = opt;
  options.type = nvcomp::TypeOf<T>();
  CascadedManager manager{options, stream};
  auto comp_config = manager.configure_compression(in_bytes);

  // Allocate output buffer
  uint8_t* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

  manager.compress(
      reinterpret_cast<const uint8_t*>(d_in_data),
      d_comp_out,
      comp_config);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t comp_out_bytes = manager.get_compressed_output_size(d_comp_out);

  cudaFree(d_in_data);

  // Test to make sure copying the compressed file is ok
  uint8_t* copied = 0;
  CUDA_CHECK(cudaMalloc(&copied, comp_out_bytes));
  CUDA_CHECK(
      cudaMemcpy(copied, d_comp_out, comp_out_bytes, cudaMemcpyDeviceToDevice));
  cudaFree(d_comp_out);
  d_comp_out = copied;

  auto decomp_config = manager.configure_decompression(d_comp_out);

  T* out_ptr;
  cudaMalloc(&out_ptr, decomp_config.decomp_data_size);

  // make sure the data won't match input if not written to, so we can verify
  // correctness
  cudaMemset(out_ptr, 0, decomp_config.decomp_data_size);

  manager.decompress(
      reinterpret_cast<uint8_t*>(out_ptr),
      d_comp_out,
      decomp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy result back to host
  std::vector<T> res(input.size());
  cudaMemcpy(
      &res[0], out_ptr, input.size() * sizeof(T), cudaMemcpyDeviceToHost);

  // Verify correctness
  REQUIRE(res == input);

  cudaFree(d_comp_out);
  cudaFree(out_ptr);

  return comp_out_bytes;
}

template <typename T>
std::vector<T> data_stair( const T start, const int64_t step,
                          const T  base, const size_t min_count ){

  std::vector<T> input;
  for (int i = 0; i < min_count; i++)
    input.push_back(start + (i*step) % base);

  return input;
}

template <typename T>
void stair_delta_bp_test(const T start, const int64_t step,
                         const T  base, const size_t min_count,
                         const size_t expect_size_common_delta,
                         const size_t expect_size_m2_delta){
  size_t size_common_delta = 0;
  size_t size_m2_delta = 0;
  nvcompBatchedCascadedOpts_t opt = nvcompBatchedCascadedDefaultOpts;
  opt.num_RLEs = 0;
  opt.use_bp = 1;
  opt.num_deltas = 1;

  std::cout << start << " | " << step << " | " << base << " | " << min_count << " | "
            << expect_size_common_delta << " | " <<  expect_size_m2_delta  << std::endl;

  using data_type = T;

  auto input = data_stair<data_type>(start, step, base, min_count);
  opt.is_m2_deltas_mode = false;
  size_common_delta = test_cascaded<data_type>(input, opt);
  printf("size_common_delta: %zu\n", size_common_delta);
  REQUIRE(expect_size_common_delta == size_common_delta);

  opt.is_m2_deltas_mode = true;
  size_m2_delta = test_cascaded<data_type>(input, opt);
  printf("size_m2_delta: %zu\n", size_m2_delta);
  REQUIRE(expect_size_m2_delta == size_m2_delta);

  REQUIRE(size_m2_delta < size_common_delta);
}
#include <limits>

template <typename T>
void test_unsigned(const char * name,
                   const size_t expect_size_common_delta,
                   const size_t expect_size_m2_delta){
  using unsignedT = std::make_unsigned_t<T>;
  using signedT = std::make_signed_t<T>;
  std::cout << "test type " << name << " / " << typeid(T).name() << std::endl;
  const T _maxU = std::numeric_limits<unsignedT>::max();
  const T _minU = std::numeric_limits<unsignedT>::min();
  const T _maxS = std::numeric_limits<signedT>::max();
  const T _minS = std::numeric_limits<signedT>::min();
  const size_t count = 5000;
  const T base = 32;

  stair_delta_bp_test<T>(_minU, 1, base, count, expect_size_common_delta, expect_size_m2_delta);
  stair_delta_bp_test<T>(_maxS - base/2, 1, base, count, expect_size_common_delta, expect_size_m2_delta);
  stair_delta_bp_test<T>(_maxS - base/2, 1, base, count, expect_size_common_delta, expect_size_m2_delta);

  stair_delta_bp_test<T>(_minU + base/2, -1, base, count, expect_size_common_delta, expect_size_m2_delta);
  stair_delta_bp_test<T>(base, -1, base, count, expect_size_common_delta, expect_size_m2_delta);
  stair_delta_bp_test<T>(_minS + base/2, -1, base, count, expect_size_common_delta, expect_size_m2_delta);
}


TEST_CASE("comp/decomp cascaded-m2delta-delta-size", "[nvcomp][small]")
{
  // 5000 count
  test_unsigned<uint8_t>("uint8_t", 3952, 200);
  test_unsigned<uint16_t>("uint16_t", 4004, 264);
  test_unsigned<uint32_t>("uint32_t", 4108, 396);
  test_unsigned<uint64_t>("uint64_t", 4488, 896);

}
