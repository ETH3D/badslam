// Copyright 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <libvis/cuda/cuda_buffer.cuh>

#include "badslam/cuda_util.cuh"
#include "badslam/kernels.cuh"

namespace vis {

struct ReverseSumIterator {
  typedef ReverseSumIterator difference_type;
  typedef u32 value_type;
  typedef u32& reference;
  typedef u32* pointer;
  typedef std::random_access_iterator_tag iterator_category;
  
  __forceinline__ __host__ __device__ ReverseSumIterator(u32* start)
      : ptr(start) {}
  
  __forceinline__ __host__ __device__ ReverseSumIterator& operator++() {
    --ptr;
    return *this;
  }
  
  __forceinline__ __host__ __device__ ReverseSumIterator operator+(int count) {
    return ReverseSumIterator(ptr - count);
  }
  
  __forceinline__ __host__ __device__ u32 operator*() const {
    // NOTE: The value is inverted.
    return 1 - *ptr;
  }
  
  __forceinline__ __host__ __device__ u32 operator[](int index) const {
    // NOTE: The value is inverted.
    return 1 - *(ptr - index);
  }
  
  u32* ptr;
};

struct ReverseOutIterator {
  typedef ReverseOutIterator difference_type;
  typedef u32 value_type;
  typedef u32& reference;
  typedef u32* pointer;
  typedef std::random_access_iterator_tag iterator_category;
  
  __forceinline__ __host__ __device__ ReverseOutIterator(u32* start)
      : ptr(start) {}
  
  __forceinline__ __host__ __device__ ReverseOutIterator& operator++() {
    --ptr;
    return *this;
  }
  
  __forceinline__ __host__ __device__ ReverseOutIterator operator+(int count) {
    return ReverseOutIterator(ptr - count);
  }
  
  __forceinline__ __host__ __device__ u32& operator*() const {
    return *ptr;
  }
  
  __forceinline__ __host__ __device__ u32& operator[](int index) const {
    return *(ptr - index);
  }
  
  u32* ptr;
};

__global__ void FlagInvalidSurfelsInAccum2CUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index < surfels_size) {
    *reinterpret_cast<u32*>(&surfels(kSurfelAccum2, surfel_index)) =
        ((__float_as_int(surfels(kSurfelX, surfel_index)) == 0x7fffffff) ? 1 : 0);
  }
}

__global__ void CreateSurfelsForKeyframeCUDAWriteFreeSpotListKernel(
    u32 new_surfel_count,
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    u32* out_index_list) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index < surfels_size) {
    if (__float_as_int(surfels(kSurfelX, surfel_index)) == 0x7fffffff) {
      // No subtraction of 1 needs to be done here since an exclusive scan is used in this case.
      u32 free_spot_index = *reinterpret_cast<u32*>(&surfels(kSurfelAccum0, surfel_index));
      if (free_spot_index < new_surfel_count) {
        out_index_list[free_spot_index] = surfel_index;
      }
    }
  }
}

template <bool adapt_active_surfels>
__global__ void CompactSurfelsCUDAKernel(
    u32 surfels_size,
    u32 free_spot_count,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index < surfels_size &&
      __float_as_int(surfels(kSurfelX, surfel_index)) != 0x7fffffff) {
    // Reverse index of surfel is in kSurfelAccum0.
    // List of free spots is in kSurfelAccum3.
    
    u32 surfel_reverse_index = *reinterpret_cast<u32*>(&surfels(kSurfelAccum0, surfel_index));
    if (surfel_reverse_index < free_spot_count) {
      u32 free_spot_index = *reinterpret_cast<u32*>(&surfels(kSurfelAccum3, surfel_reverse_index));
      
      if (free_spot_index < surfel_index) {
        // Copy the surfel to free_spot_index.
        #pragma unroll
        for (int row = 0; row < kSurfelDataAttributeCount; ++ row) {
          surfels(row, free_spot_index) = surfels(row, surfel_index);
        }
        
        if (adapt_active_surfels) {
          active_surfels(0, free_spot_index) = active_surfels(0, surfel_index);
        }
      }
    }
  }
}

void CompactSurfelsCUDA(
    cudaStream_t stream,
    void** free_spots_temp_storage,
    usize* free_spots_temp_storage_bytes,
    u32 surfel_count,
    u32* surfels_size,
    CUDABuffer_<float>* surfels,
    CUDABuffer_<u8>* active_surfels) {
  CUDA_CHECK();
  if (*surfels_size == surfel_count) {
    return;
  }
  
  // Algorithm:
  // 1. Count free spots and save their indices in the free_spot_index_list.
  // 2. Count occupied spots in reverse order (the last occupied surfel gets
  //    index 0, the second last gets 1, etc.)
  // 3. For all surfels for which free_spot_index_list[reverse_index] is smaller
  //    than the surfel index, copy the surfel to this free spot.
  // 4. The surfel with largest surfel_index which is not copied, or the largest
  //    target index, whichever is larger, determines the new surfels_size.
  
  // 1. Count free spots (put list into kSurfelAccum3)
  
  // Flag in kSurfelAccum2 whether the surfel is invalid
  CUDA_AUTO_TUNE_1D(
      FlagInvalidSurfelsInAccum2CUDAKernel,
      1024,
      *surfels_size,
      0, stream,
      /* kernel parameters */
      *surfels_size,
      *surfels);
  CUDA_CHECK();
  
  // Do prefix scan
  u32* surfels_accum2_u32 = reinterpret_cast<u32*>(reinterpret_cast<u8*>(surfels->address()) + kSurfelAccum2 * surfels->pitch());
  void* surfels_accum1_void = reinterpret_cast<void*>(reinterpret_cast<u8*>(surfels->address()) + kSurfelAccum1 * surfels->pitch());
  u32* surfels_accum0_u32 = reinterpret_cast<u32*>(reinterpret_cast<u8*>(surfels->address()) + kSurfelAccum0 * surfels->pitch());
  
  if (*free_spots_temp_storage_bytes == 0) {
    cub::DeviceScan::ExclusiveSum(
        *free_spots_temp_storage,
        *free_spots_temp_storage_bytes,
        surfels_accum2_u32,
        surfels_accum0_u32,
        surfels->width(),  // Make sure that the temp storage is sufficient for the whole surfels buffer
        stream);
    
    // Do not allocate a separate memory region for this, just verify that
    // one of the surfel accum buffers is large enough.
    CHECK_GE(surfels->width() * sizeof(float), *free_spots_temp_storage_bytes);
    // cudaMalloc(free_spots_temp_storage, *free_spots_temp_storage_bytes);
  }
  
  cub::DeviceScan::ExclusiveSum(
      surfels_accum1_void,  // used as temp storage
      *free_spots_temp_storage_bytes,
      surfels_accum2_u32,
      surfels_accum0_u32,
      *surfels_size,
      stream);
  CUDA_CHECK();
  
  // Write a compact list of free spot indices that can be used to find the
  // free spot for a new surfel.
  u32* free_spot_index_list = reinterpret_cast<u32*>(reinterpret_cast<u8*>(surfels->address()) + kSurfelAccum3 * surfels->pitch());
  CUDA_AUTO_TUNE_1D(
      CreateSurfelsForKeyframeCUDAWriteFreeSpotListKernel,
      1024,
      *surfels_size,
      0, stream,
      /* kernel parameters */
      *surfels_size - surfel_count,
      *surfels_size,
      *surfels,
      free_spot_index_list);
  CUDA_CHECK();
  
  // 2. Reverse count occupied spots (put index into kSurfelAccum0)
  
  cub::DeviceScan::ExclusiveSum(
      surfels_accum1_void,  // used as temp storage
      *free_spots_temp_storage_bytes,
      ReverseSumIterator(surfels_accum2_u32 + (*surfels_size - 1)),
      ReverseOutIterator(surfels_accum0_u32 + (*surfels_size - 1)),
      *surfels_size,
      stream);
  CUDA_CHECK();
  
  // 3. / 4. Copy surfels which move to a smaller index, get new surfels_size
  
  if (active_surfels != nullptr) {
    CUDA_AUTO_TUNE_1D_TEMPLATED(
        CompactSurfelsCUDAKernel,
        1024,
        *surfels_size,
        0, stream,
        TEMPLATE_ARGUMENTS(true),
        /* kernel parameters */
        *surfels_size,
        *surfels_size - surfel_count,
        *surfels,
        *active_surfels);
  } else {
    CUDA_AUTO_TUNE_1D_TEMPLATED(
        CompactSurfelsCUDAKernel,
        1024,
        *surfels_size,
        0, stream,
        TEMPLATE_ARGUMENTS(false),
        /* kernel parameters */
        *surfels_size,
        *surfels_size - surfel_count,
        *surfels,
        CUDABuffer_<u8>());
  }
  CUDA_CHECK();
  
  *surfels_size = surfel_count;
}

}
