/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVCOMP_BITCOMP_H
#define NVCOMP_BITCOMP_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef ENABLE_BITCOMP

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure for configuring Bitcomp compression.
 */
typedef struct
{
  /**
   * @brief Bitcomp algorithm options.
   *  algorithm_type: The type of Bitcomp algorithm used.
   *    0 : Default algorithm, usually gives the best compression ratios
   *    1 : "Sparse" algorithm, works well on sparse data (with lots of zeroes).
   *        and is usually a faster than the default algorithm.
   */
  int algorithm_type;
} nvcompBitcompFormatOpts;


/**
 * @brief Extracts the metadata from the input in_ptr on the device and copies
 * it to the host. This function synchronizes on the stream.
 *
 * @param in_ptr The compressed memory on the device.
 * @param in_bytes The size of the compressed memory on the device.
 * @param metadata_ptr The metadata on the host to create from the compresesd
 * data.
 * @param stream The stream to use for copying from the device to the host.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompDecompressGetMetadata(
    const void* in_ptr,
    size_t in_bytes,
    void** metadata_ptr,
    cudaStream_t stream);

/**
 * @brief Destroys the metadata object and frees the associated memory.
 *
 * @param metadata_ptr The pointer to destroy.
 */
void nvcompBitcompDecompressDestroyMetadata(void* metadata_ptr);

/**
 * @brief Computes the temporary storage size needed to decompress.
 *
 * NOTE: bitcomp does not require any temporary storage for decompression
 * 
 * @param metadata_ptr The metadata.
 * @param temp_bytes The size of temporary workspace required to perform
 * decomrpession, in bytes (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompDecompressGetTempSize(
    const void* metadata_ptr, size_t* temp_bytes);

/**
 * @brief Computes the decompressed size of the data.
 *
 * @param metadata_ptr The metadata.
 * @param output_bytes The size of the decompressed data in bytes (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompDecompressGetOutputSize(
    const void* metadata_ptr, size_t* output_bytes);

/**
 * @brief Perform the asynchronous decompression.
 *
 * @param in_ptr The compressed data on the device to decompress.
 * @param in_bytes The size of the compressed data.
 * @param temp_ptr The temporary workspace on the device. Not used, can pass NULL.
 * @param temp_bytes The size of the temporary workspace. Not used.
 * @param metadata_ptr The metadata.
 * @param out_ptr The output location on the device.
 * @param out_bytes The size of the output location.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompDecompressAsync(
    const void* in_ptr,
    size_t in_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    void* metadata_ptr,
    void* out_ptr,
    size_t out_bytes,
    cudaStream_t stream);

/**
 * @brief Get the temporary workspace size required to perform compression.
 *
 * @param in_ptr The uncompressed data on the device.
 * @param in_bytes The size of the uncompressed data in bytes.
 * @param in_type The type of the uncompressed data.
 * @param format_opts The bitcomp format options (can pass NULL for default options).
 * @param temp_bytes The size of the required temporary workspace in bytes
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompCompressGetTempSize(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    size_t* temp_bytes);
/**
 * @brief Get the required output size to perform compression.
 *
 * NOTE: Computing exact output size is currently not supported.
 *
 * @param in_ptr The uncompressed data on the device.
 * @param in_bytes The size of the uncompressed data in bytes.
 * @param in_type The type of the uncompressed data.
 * @param format_opts The bitcomp format options (can pass NULL for default options).
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param out_bytes The required size of the output location in bytes (output).
 * @param exact_out_bytes Whether or not to compute the exact number of bytes
 * needed, or quickly compute a loose upper bound.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompCompressGetOutputSize(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    const nvcompBitcompFormatOpts* format_opts,
    void* temp_ptr,
    size_t temp_bytes,
    size_t* out_bytes,
    int exact_out_bytes);

/**
 * @brief Perform asynchronous compression. The pointer `out_bytes` must be to
 * pinned memory for this to be asynchronous.
 *
 * @param in_ptr The uncompressed data on the device.
 * @param in_bytes The size of the uncompressed data in bytes.
 * @param in_type The data type of the uncompressed data.
 * @param format_opts The bitcomp format options (can pass NULL for default options).
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param out_ptr The location to write compresesd data to on the device.
 * @param out_bytes The size of the output location on input, and the size of
 * the compressed data on output. If pinned memory, the stream must be
 * synchronized with, before reading.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompCompressAsync(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    const nvcompBitcompFormatOpts* format_opts,
    void* temp_ptr,
    size_t temp_bytes,
    void* out_ptr,
    size_t* out_bytes,
    cudaStream_t stream);

/**
 * @brief Checks if the compressed data was compressed with bitcomp.
 *
 * @param in_ptr The compressed data.
 * @param in_bytes The size of the compressed buffer.
 *
 * @return 1 if the data was compressed with bitcomp, 0 otherwise
 */
int nvcompIsBitcompData(const void* const in_ptr, size_t in_bytes);

#ifdef __cplusplus
}
#endif

#endif  // ENABLE_BITCOMP

#endif