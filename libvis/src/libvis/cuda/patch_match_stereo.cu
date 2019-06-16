// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
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


// Avoid warnings in includes with CUDA compiler
#pragma GCC diagnostic ignored "-Wattributes"
#pragma diag_suppress code_is_unreachable

#include "libvis/cuda/patch_match_stereo.cuh"

#include <math_constants.h>

#include "libvis/cuda/cuda_auto_tuner.h"
#include "libvis/cuda/cuda_unprojection_lookup.cuh"
#include "libvis/cuda/cuda_util.h"

namespace vis {

constexpr float kMinInvDepth = 1e-5f;  // TODO: Make parameter


__forceinline__ __device__ float SampleAtProjectedPosition(
    const float x, const float y, const float z,
    const PixelCornerProjector_& projector,
    const CUDAMatrix3x4& stereo_tr_reference,
    cudaTextureObject_t stereo_texture) {
  float3 pnxy = stereo_tr_reference * make_float3(x, y, z);
  if (pnxy.z <= 0.f) {
    return CUDART_NAN_F;
  }
  
  const float2 pxy = projector.Project(pnxy);
  
  if (pxy.x < 0.5f ||
      pxy.y < 0.5f ||
      pxy.x >= projector.width - 0.5f ||
      pxy.y >= projector.height - 0.5f) {
    return CUDART_NAN_F;
  } else {
    return 255.0f * tex2D<float>(stereo_texture, pxy.x, pxy.y);
  }
}


__forceinline__ __device__ float CalculatePlaneDepth2(
    float d, const float2& normal_xy, float normal_z,
    float query_x, float query_y) {
  return d / (query_x * normal_xy.x + query_y * normal_xy.y + normal_z);
}

__forceinline__ __device__ float CalculatePlaneInvDepth2(
    float d, const float2& normal_xy, float normal_z,
    float query_x, float query_y) {
  return (query_x * normal_xy.x + query_y * normal_xy.y + normal_z) / d;
}


template <int kContextRadius>
__forceinline__ __device__ float ComputeCostsSSD(
    int x, int y,
    const float2& normal_xy,
    const float inv_depth,
    const CUDAUnprojectionLookup2D_& unprojector,
    const CUDABuffer_<u8>& reference_image,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& projector,
    cudaTextureObject_t stereo_image) {
  if (inv_depth < kMinInvDepth) {
    return CUDART_NAN_F;
  }
  
  const float normal_z =
      -sqrtf(1.f - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y);
  const float depth = 1.f / inv_depth;
  const float2 center_nxy =
      unprojector.UnprojectPoint(x, y);
  const float plane_d =
      (center_nxy.x * depth) * normal_xy.x +
      (center_nxy.y * depth) * normal_xy.y + depth * normal_z;
  
  float cost = 0;
  
  #pragma unroll
  for (int dy = -kContextRadius; dy <= kContextRadius; ++ dy) {
    #pragma unroll
    for (int dx = -kContextRadius; dx <= kContextRadius; ++ dx) {
      float2 nxy = unprojector.UnprojectPoint(x + dx, y + dy);
      float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x, nxy.y);
      nxy.x *= plane_depth;
      nxy.y *= plane_depth;
      
      float sample =
            SampleAtProjectedPosition(nxy.x, nxy.y, plane_depth,
                                      projector,
                                      stereo_tr_reference,
                                      stereo_image);
      
      const float diff = sample - reference_image(y + dy, x + dx);
      cost += diff * diff;
    }
  }
  
  return cost;
}

// Computes 0.5f * (1 - ZNCC), so that the result can be used
// as a cost value with range [0; 1].
__forceinline__ __device__ float ComputeZNCCBasedCost(
    const int num_samples,
    const float sum_a,
    const float squared_sum_a,
    const float sum_b,
    const float squared_sum_b,
    const float product_sum) {
  const float normalizer = 1.0f / num_samples;

  const float numerator =
      product_sum - normalizer * (sum_a * sum_b);
  const float denominator_reference =
      squared_sum_a - normalizer * sum_a * sum_a;
  const float denominator_other =
      squared_sum_b - normalizer * sum_b * sum_b;
  
  // NOTE: Using a threshold on homogeneous patches is required here since
  //       otherwise the optimum might be a noisy value in a homogeneous area.
  constexpr float kHomogeneousThreshold = 0.1f;
  if (denominator_reference < kHomogeneousThreshold ||
      denominator_other < kHomogeneousThreshold) {
    return 1.0f;
  } else {
    return 0.5f * (1.0f - numerator *
        rsqrtf(denominator_reference * denominator_other));
  }
}

constexpr int kNumSamples = 81;

__constant__ float kSamplesCUDA[kNumSamples][2];

// // Sphere:
// constexpr float kSamples[kNumSamples][2] = {
//     {-0.352334470334, -0.698301652151},
//     {0.30186894608, -0.855127426665},
//     {0.0717640086134, -0.268622166175},
//     {-0.884002150451, 0.0148714663788},
//     {-0.925008683116, -0.132708632675},
//     {-0.150961621715, 0.653704249344},
//     {-0.752396077701, -0.553522070786},
//     {0.254866444811, 0.895417884914},
//     {0.154205897235, -0.206639050698},
//     {0.716936918097, -0.420781427337},
//     {-0.383036351796, 0.63225271824},
//     {-0.638547240152, 0.163200327325},
//     {0.277826937852, -0.255204914549},
//     {0.0954889314191, -0.874422050053},
//     {0.360799946364, -0.144815388661},
//     {-0.371705659246, 0.171123727015},
//     {-0.0936312472584, -0.400466006273},
//     {0.588758963045, 0.397988867459},
//     {-0.511806978556, 0.148847420517},
//     {0.0503930076229, 0.750274991147},
//     {0.458890578878, -0.42412447022},
//     {-0.16375435643, 0.51428185913},
//     {-0.696030930679, -0.0220737990484},
//     {-0.921585485905, 0.336431713069},
//     {0.529141732426, 0.146051880555},
//     {0.750955623662, -0.372504974304},
//     {0.390590732547, 0.18873975421},
//     {0.159790408565, -0.0875893373972},
//     {-0.0518033251607, 0.328304410949},
//     {-0.878661144806, 0.402984042609},
//     {0.643849573219, -0.430808935812},
//     {-0.228417115107, 0.337305431768},
//     {-0.954874143889, -0.0766094274005},
//     {-0.741319555963, -0.504770332606},
//     {-0.218100593734, 0.742843948253},
//     {-0.8388373976, -0.101625198101},
//     {0.0988798182881, 0.766767652883},
//     {0.638559675671, 0.727968939397},
//     {-0.443157870972, -0.169406965577},
//     {-0.282457669337, 0.768385654396},
//     {-0.647564543019, -0.536086266361},
//     {-0.533327832638, -0.0300745393173},
//     {0.178247007465, -0.474506761403},
//     {-0.261492854211, 0.132682447413},
//     {0.90619585105, 0.380987314272},
//     {0.0309828661416, 0.235185498818},
//     {0.352400164899, -0.892014213552},
//     {0.799066020116, 0.559938981412},
//     {0.749026368269, 0.595746242393},
//     {-0.215242186217, -0.202042335359},
//     {-0.792925812579, 0.268579131371},
//     {-0.582473629108, -0.675393624456},
//     {-0.319892695535, -0.894848792219},
//     {-0.797071263955, -0.272780155931},
//     {0.228137975577, -0.702899029338},
//     {-0.495484486886, -0.305220907893},
//     {-0.271673120943, -0.754315538476},
//     {-0.0680210816801, -0.0323306871675},
//     {-0.314728323514, -0.470486216566},
//     {0.657710756243, -0.677122778947},
//     {0.0565147900842, -0.706794922202},
//     {0.0863448517642, -0.945915017156},
//     {0.0562188818766, 0.957002485438},
//     {0.726650060579, 0.392393571816},
//     {-0.477769605541, -0.266600416478},
//     {-0.665915930931, 0.543875816804},
//     {0.0651847949858, 0.558109782676},
//     {-0.340670009904, -0.553916653794},
//     {0.705257597493, 0.612157169571},
//     {0.636665886651, 0.479746040751},
//     {-0.546521019937, 0.035277448487},
//     {-0.28887491329, -0.942039698517},
//     {-0.481651273464, 0.3850438834},
//     {0.913030152683, -0.105544644467},
//     {0.910001262643, -0.270728229276},
//     {-0.559075354008, -0.546308346539},
//     {-0.606587673161, -0.591253273448},
//     {0.248132794876, 0.800616675768},
//     {0.680871054559, -0.0410531474769},
//     {0.305956085682, 0.599287489699},
//     {-0.830443027099, 0.32117130041}
// };

// Square:
constexpr float kSamples[kNumSamples][2] = {
    {0.912068543778f, 0.895654974119f},
    {-0.886897264546f, -0.830256009682f},
    {0.670997756259f, 0.471939978137f},
    {0.33946080288f, -0.383727084822f},
    {0.211888331357f, 0.213603467282f},
    {0.162408034224f, -0.68323425949f},
    {-0.138660719417f, -0.212936359589f},
    {0.446024162475f, 0.989639125899f},
    {0.898790946186f, 0.0883540948586f},
    {-0.110291622548f, -0.463518516701f},
    {-0.928151341214f, -0.945110285818f},
    {-0.0702122758054f, -0.363069744293f},
    {-0.239970156199f, 0.783578915657f},
    {0.0515055382921f, 0.121020722053f},
    {-0.52775318577f, -0.952283841718f},
    {-0.349714142478f, -0.726605214027f},
    {0.0204476916744f, 0.997367136385f},
    {0.348959394692f, -0.636313006354f},
    {0.787143073166f, 0.593519842843f},
    {0.468803383788f, 0.813187299795f},
    {0.525770967666f, 0.579495274924f},
    {-0.292426044317f, 0.961953146144f},
    {0.923801875796f, -0.677630693392f},
    {0.508008143304f, 0.430301796475f},
    {-0.077186604516f, 0.0607114322469f},
    {-0.0199721562996f, 0.849664144189f},
    {0.00168212526131f, 0.663048979584f},
    {-0.292151590263f, 0.765701837163f},
    {0.799401177513f, -0.0779756702367f},
    {0.13541014084f, 0.840660878384f},
    {0.447545907744f, -0.0267828902768f},
    {-0.556377978018f, -0.350665512462f},
    {0.39914327614f, -0.667860629012f},
    {0.815880993252f, -0.4637249742f},
    {0.822755671736f, -0.380873750101f},
    {0.914723423112f, 0.412411612735f},
    {0.00849763396664f, 0.0354955122971f},
    {0.302828797934f, 0.175889423589f},
    {-0.376311350898f, -0.584363050924f},
    {0.0237833167106f, 0.868308718268f},
    {0.246530173452f, -0.849249261852f},
    {0.640799989424f, 0.451898574955f},
    {0.815307241903f, -0.617194533392f},
    {0.489565448555f, -0.882482207203f},
    {0.305819854869f, -0.453800535326f},
    {-0.54676694151f, 0.750982342896f},
    {-0.787468034709f, 0.0447253307179f},
    {0.70788601437f, -0.510336044062f},
    {-0.579042122609f, 0.761163518733f},
    {-0.154164703221f, 0.43392219781f},
    {-0.936253859747f, -0.275286177394f},
    {-0.656238015749f, 0.345530882827f},
    {-0.834193645191f, 0.909124330691f},
    {-0.949310570346f, 0.458847014884f},
    {-0.957710260554f, -0.488619891885f},
    {0.626708774805f, -0.685763422626f},
    {-0.632522381498f, 0.382990852027f},
    {-0.228868237295f, -0.913678008405f},
    {0.980003092406f, -0.697159782487f},
    {-0.927462011513f, -0.311597988926f},
    {0.23047896665f, 0.484919246252f},
    {-0.773770193964f, -0.325572453607f},
    {-0.938378284747f, -0.102693475015f},
    {0.531939873176f, 0.479893327441f},
    {0.8040403167f, 0.51132430735f},
    {0.724891552651f, 0.410690280102f},
    {-0.0544409758034f, -0.548944859304f},
    {0.321656997314f, -0.367388146543f},
    {-0.795901789901f, -0.104356277162f},
    {0.749526082696f, -0.744927070762f},
    {0.169911396221f, -0.21409489979f},
    {0.0296053935284f, -0.712341072046f},
    {0.919462372974f, -0.481807153496f},
    {0.212155878107f, -0.160488908689f},
    {-0.96393356175f, 0.115900247678f},
    {-0.718861242077f, -0.886438008354f},
    {-0.932887507503f, -0.677669969638f},
    {-0.808256112734f, 0.270151395077f},
    {0.0165183680713f, 0.966932188095f},
    {0.868260637394f, 0.989050466519f},
    {-0.535052318664f, -0.110605089761f}
};

template <int kContextRadius>
__forceinline__ __device__ float ComputeCostsZNCC(
    int x, int y,
    const float2& normal_xy,
    const float inv_depth,
    const CUDAUnprojectionLookup2D_& unprojector,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& projector,
    cudaTextureObject_t stereo_image) {
  if (inv_depth < kMinInvDepth) {
    return CUDART_NAN_F;
  }
  
  const float normal_z =
      -sqrtf(1.f - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y);
  const float depth = 1.f / inv_depth;
  const float2 center_nxy =
      unprojector.UnprojectPoint(x, y);
  const float plane_d =
      (center_nxy.x * depth) * normal_xy.x +
      (center_nxy.y * depth) * normal_xy.y + depth * normal_z;
  
  float sum_a = 0;
  float squared_sum_a = 0;
  float sum_b = 0;
  float squared_sum_b = 0;
  float product_sum = 0;
  
  for (int sample = 0; sample < kNumSamples; ++ sample) {
    float dx = 1.25f * kContextRadius * kSamplesCUDA[sample][0];  // TODO: magic constant factor
    float dy = 1.25f * kContextRadius * kSamplesCUDA[sample][1];  // TODO: magic constant factor
    
    float2 nxy = unprojector.UnprojectPoint(x + dx, y + dy);  // NOTE: This is only approximate (bilinear interpolation of exact values sampled at pixel centers).
    float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x, nxy.y);
    nxy.x *= plane_depth;
    nxy.y *= plane_depth;
    
    float stereo_value =
          SampleAtProjectedPosition(nxy.x, nxy.y, plane_depth,
                                    projector,
                                    stereo_tr_reference,
                                    stereo_image);
    
    sum_a += stereo_value;
    squared_sum_a += stereo_value * stereo_value;
    
    float reference_value = 255.f * tex2D<float>(reference_texture, x + dx + 0.5f, y + dy + 0.5f);
    
    sum_b += reference_value;
    squared_sum_b += reference_value * reference_value;
    
    product_sum += stereo_value * reference_value;
  }
  
//   #pragma unroll
//   for (int dy = -kContextRadius; dy <= kContextRadius; ++ dy) {
//     #pragma unroll
//     for (int dx = -kContextRadius; dx <= kContextRadius; ++ dx) {
//       float2 nxy = unprojector.UnprojectPoint(x + dx, y + dy);
//       float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x, nxy.y);
//       nxy.x *= plane_depth;
//       nxy.y *= plane_depth;
//       
//       float stereo_value =
//             SampleAtProjectedPosition(nxy.x, nxy.y, plane_depth,
//                                       projector,
//                                       stereo_tr_reference,
//                                       stereo_image);
//       
//       sum_a += stereo_value;
//       squared_sum_a += stereo_value * stereo_value;
//       
//       float reference_value = reference_image(y + dy, x + dx);
//       
//       sum_b += reference_value;
//       squared_sum_b += reference_value * reference_value;
//       
//       product_sum += stereo_value * reference_value;
//     }
//   }
  
  return ComputeZNCCBasedCost(
      kNumSamples, sum_a, squared_sum_a, sum_b, squared_sum_b, product_sum);
}

template <int kContextRadius>
__forceinline__ __device__ float ComputeCostsCensus(
    int x, int y,
    const float2& normal_xy,
    const float inv_depth,
    const CUDAUnprojectionLookup2D_& unprojector,
    const CUDABuffer_<u8>& reference_image,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& projector,
    cudaTextureObject_t stereo_image) {
  if (inv_depth < kMinInvDepth) {
    return CUDART_NAN_F;
  }
  
  const float normal_z =
      -sqrtf(1.f - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y);
  const float depth = 1.f / inv_depth;
  const float2 center_nxy =
      unprojector.UnprojectPoint(x, y);
  const float plane_d =
      (center_nxy.x * depth) * normal_xy.x +
      (center_nxy.y * depth) * normal_xy.y + depth * normal_z;
  
  float stereo_center_value =
      SampleAtProjectedPosition(center_nxy.x * depth, center_nxy.y * depth, depth,
                                projector,
                                stereo_tr_reference,
                                stereo_image);
  u8 reference_center_value = reference_image(y, x);
  
  float cost = 0;
  
  constexpr int kSpreadFactor = 2;  // TODO: Make parameter
  
  #pragma unroll
  for (int dy = -kSpreadFactor * kContextRadius; dy <= kSpreadFactor * kContextRadius; dy += kSpreadFactor) {
    #pragma unroll
    for (int dx = -kSpreadFactor * kContextRadius; dx <= kSpreadFactor * kContextRadius; dx += kSpreadFactor) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      if (x + dx < 0 ||
          y + dy < 0 ||
          x + dx >= reference_image.width() ||
          y + dy >= reference_image.height()) {
        continue;
      }
      
      float2 nxy = unprojector.UnprojectPoint(x + dx, y + dy);
      float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x, nxy.y);
      nxy.x *= plane_depth;
      nxy.y *= plane_depth;
      
      float stereo_value =
            SampleAtProjectedPosition(nxy.x, nxy.y, plane_depth,
                                      projector,
                                      stereo_tr_reference,
                                      stereo_image);
      if (::isnan(stereo_value)) {
        return CUDART_NAN_F;
      }
      int stereo_bit = stereo_value > stereo_center_value;
      
      u8 reference_value = reference_image(y + dy, x + dx);
      int reference_bit = reference_value > reference_center_value;
      
      cost += stereo_bit != reference_bit;
    }
  }
  
  return cost;
}

template <int kContextRadius>
__forceinline__ __device__ float ComputeCosts(
    int x, int y,
    const float2& normal_xy,
    const float inv_depth,
    const CUDAUnprojectionLookup2D_& unprojector,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& projector,
    cudaTextureObject_t stereo_image,
    int match_metric,
    float second_best_min_distance_factor,
    CUDABuffer_<float> best_inv_depth_map) {
  if (second_best_min_distance_factor > 0) {
    // Reject estimates which are too close to the best inv depth.
    float best_inv_depth = best_inv_depth_map(y, x);
    float factor = best_inv_depth / inv_depth;
    if (factor < 1) {
      factor = 1 / factor;
    }
    if (factor < second_best_min_distance_factor) {
      return CUDART_NAN_F;
    }
  }
  
  // TODO: Commented out for higher compile speed (and since only ZNCC is consistent with outlier filtering etc.)
//   if (match_metric == kPatchMatchStereo_MatchMetric_SSD) {
//     return ComputeCostsSSD<kContextRadius>(
//         x, y, normal_xy, inv_depth, unprojector, reference_image,
//         stereo_tr_reference, projector, stereo_image);
//   } else if (match_metric == kPatchMatchStereo_MatchMetric_ZNCC) {
    return ComputeCostsZNCC<kContextRadius>(
        x, y, normal_xy, inv_depth, unprojector, reference_image, reference_texture,
        stereo_tr_reference, projector, stereo_image);
//   } else if (match_metric == kPatchMatchStereo_MatchMetric_Census) {
//     return ComputeCostsCensus<kContextRadius>(
//         x, y, normal_xy, inv_depth, unprojector, reference_image,
//         stereo_tr_reference, projector, stereo_image);
//   }
  
  // This should never be reached since all metrics should be handled above.
  return 0;
}

template <int kContextRadius>
__global__ void InitPatchMatchCUDAKernel(
    int match_metric,
    float max_normal_2d_length,
    CUDAUnprojectionLookup2D_ unprojector,
    CUDABuffer_<u8> reference_image,
    cudaTextureObject_t reference_texture,
    CUDAMatrix3x4 stereo_tr_reference,
    PixelCornerProjector_ projector,
    cudaTextureObject_t stereo_image,
    float inv_min_depth,
    float inv_max_depth,
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<char2> normals,
    CUDABuffer_<float> costs,
    CUDABuffer_<curandState> random_states,
    CUDABuffer_<float> lambda,
    float second_best_min_distance_factor,
    CUDABuffer_<float> best_inv_depth_map) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= kContextRadius && y >= kContextRadius &&
      x < inv_depth_map.width() - kContextRadius && y < inv_depth_map.height() - kContextRadius) {
    // Initialize random states
    // TODO: Would it be better to do this only once for each PatchMatchStereo object?
    int id = x + inv_depth_map.width() * y;
    curand_init(id, 0, 0, &random_states(y, x));
    
    // Initialize random initial normals
    constexpr float kNormalRange = 1.0f;
    float2 normal_xy;
    normal_xy.x = kNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
    normal_xy.y = kNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
    float length = sqrtf(normal_xy.x * normal_xy.x + normal_xy.y * normal_xy.y);
    if (length > max_normal_2d_length) {
      normal_xy.x *= max_normal_2d_length / length;
      normal_xy.y *= max_normal_2d_length / length;
    }
    normals(y, x) = make_char2(normal_xy.x * 127.f, normal_xy.y * 127.f);
    
    // Initialize random initial depths
    const float inv_depth = inv_max_depth + (inv_min_depth - inv_max_depth) * curand_uniform(&random_states(y, x));
    inv_depth_map(y, x) = inv_depth;
    
    // Initialize lambda
    lambda(y, x) = 1.02f;  // TODO: tune
    
    // Compute initial costs
    costs(y, x) = ComputeCosts<kContextRadius>(
        x, y,
        normal_xy,
        inv_depth,
        unprojector,
        reference_image,
        reference_texture,
        stereo_tr_reference,
        projector,
        stereo_image,
        match_metric,
        second_best_min_distance_factor,
        best_inv_depth_map);
  }
}

void InitPatchMatchCUDA(
      cudaStream_t stream,
      int match_metric,
      int context_radius,
      float max_normal_2d_length,
      cudaTextureObject_t reference_unprojection_lookup,
      const CUDABuffer_<u8>& reference_image,
      cudaTextureObject_t reference_texture,
      const CUDAMatrix3x4& stereo_tr_reference,
      const PixelCornerProjector_& stereo_camera,
      const cudaTextureObject_t stereo_image,
      float inv_min_depth,
      float inv_max_depth,
      CUDABuffer_<float>* inv_depth_map,
      CUDABuffer_<char2>* normals,
      CUDABuffer_<float>* costs,
      CUDABuffer_<curandState>* random_states,
      CUDABuffer_<float>* lambda,
      float second_best_min_distance_factor,
      CUDABuffer_<float>* best_inv_depth_map) {
  // TODO: Do this separately
  static bool initialized = false;
  if (!initialized) {
    cudaMemcpyToSymbol(kSamplesCUDA, kSamples, kNumSamples * 2 * sizeof(float));
    
    initialized = true;
  }
  
  CHECK_CUDA_NO_ERROR();
  COMPILE_INT_4_OPTIONS(context_radius, 1, 2, 4, 5, CUDA_AUTO_TUNE_2D(
      InitPatchMatchCUDAKernel<_context_radius>,
      16, 16,
      inv_depth_map->width(), inv_depth_map->height(),
      0, stream,
      /* kernel parameters */
      match_metric,
      max_normal_2d_length,
      CUDAUnprojectionLookup2D_(reference_unprojection_lookup),
      reference_image,
      reference_texture,
      stereo_tr_reference,
      stereo_camera,
      stereo_image,
      inv_min_depth,
      inv_max_depth,
      *inv_depth_map,
      *normals,
      *costs,
      *random_states,
      *lambda,
      second_best_min_distance_factor,
      best_inv_depth_map ? *best_inv_depth_map : CUDABuffer_<float>()));
  CHECK_CUDA_NO_ERROR();
}


template <int kContextRadius, bool mutate_depth, bool mutate_normal>
__global__ void PatchMatchMutationStepCUDAKernel(
    int match_metric,
    float max_normal_2d_length,
    CUDAUnprojectionLookup2D_ unprojector,
    CUDABuffer_<u8> reference_image,
    cudaTextureObject_t reference_texture,
    CUDAMatrix3x4 stereo_tr_reference,
    PixelCornerProjector_ projector,
    cudaTextureObject_t stereo_image,
    float step_range,
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<char2> normals,
    CUDABuffer_<float> costs,
    CUDABuffer_<curandState> random_states,
    float second_best_min_distance_factor,
    CUDABuffer_<float> best_inv_depth_map) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= kContextRadius && y >= kContextRadius &&
      x < inv_depth_map.width() - kContextRadius && y < inv_depth_map.height() - kContextRadius) {
    float proposed_inv_depth = inv_depth_map(y, x);
    if (mutate_depth) {
      proposed_inv_depth =
          max(kMinInvDepth,
              fabsf(proposed_inv_depth + step_range *
                  (curand_uniform(&random_states(y, x)) - 0.5f)));
    }
    
    constexpr float kRandomNormalRange = 1.0f;
    const char2 proposed_normal_char = normals(y, x);
    float2 proposed_normal = make_float2(
        proposed_normal_char.x * (1 / 127.f), proposed_normal_char.y * (1 / 127.f));
    if (mutate_normal) {
      proposed_normal.x += kRandomNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
      proposed_normal.y += kRandomNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
      float length = sqrtf(proposed_normal.x * proposed_normal.x + proposed_normal.y * proposed_normal.y);
      if (length > max_normal_2d_length) {
        proposed_normal.x *= max_normal_2d_length / length;
        proposed_normal.y *= max_normal_2d_length / length;
      }
    }
    
    // Test whether to accept the proposal
    float proposal_costs = ComputeCosts<kContextRadius>(
        x, y,
        proposed_normal,
        proposed_inv_depth,
        unprojector,
        reference_image,
        reference_texture,
        stereo_tr_reference,
        projector,
        stereo_image,
        match_metric,
        second_best_min_distance_factor,
        best_inv_depth_map);
    
    if (!::isnan(proposal_costs) && !(proposal_costs >= costs(y, x))) {
      costs(y, x) = proposal_costs;
      normals(y, x) = make_char2(proposed_normal.x * 127.f, proposed_normal.y * 127.f);
      inv_depth_map(y, x) = proposed_inv_depth;
    }
  }
}

void PatchMatchMutationStepCUDA(
    cudaStream_t stream,
    int match_metric,
    int context_radius,
    float max_normal_2d_length,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& stereo_camera,
    const cudaTextureObject_t stereo_image,
    float step_range,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<char2>* normals,
    CUDABuffer_<float>* costs,
    CUDABuffer_<curandState>* random_states,
    float second_best_min_distance_factor,
    CUDABuffer_<float>* best_inv_depth_map) {
  CHECK_CUDA_NO_ERROR();
  COMPILE_INT_4_OPTIONS(context_radius, 1, 2, 4, 5, CUDA_AUTO_TUNE_2D_TEMPLATED(
      PatchMatchMutationStepCUDAKernel,
      16, 16,
      inv_depth_map->width(), inv_depth_map->height(),
      0, stream,
      TEMPLATE_ARGUMENTS(_context_radius, true, true),
      /* kernel parameters */
      match_metric,
      max_normal_2d_length,
      CUDAUnprojectionLookup2D_(reference_unprojection_lookup),
      reference_image,
      reference_texture,
      stereo_tr_reference,
      stereo_camera,
      stereo_image,
      step_range,
      *inv_depth_map,
      *normals,
      *costs,
      *random_states,
      second_best_min_distance_factor,
      best_inv_depth_map ? *best_inv_depth_map : CUDABuffer_<float>()));
  CHECK_CUDA_NO_ERROR();
  COMPILE_INT_4_OPTIONS(context_radius, 1, 2, 4, 5, CUDA_AUTO_TUNE_2D_TEMPLATED(
      PatchMatchMutationStepCUDAKernel,
      16, 16,
      inv_depth_map->width(), inv_depth_map->height(),
      0, stream,
      TEMPLATE_ARGUMENTS(_context_radius, true, false),
      /* kernel parameters */
      match_metric,
      max_normal_2d_length,
      CUDAUnprojectionLookup2D_(reference_unprojection_lookup),
      reference_image,
      reference_texture,
      stereo_tr_reference,
      stereo_camera,
      stereo_image,
      step_range,
      *inv_depth_map,
      *normals,
      *costs,
      *random_states,
      second_best_min_distance_factor,
      best_inv_depth_map ? *best_inv_depth_map : CUDABuffer_<float>()));
  CHECK_CUDA_NO_ERROR();
  COMPILE_INT_4_OPTIONS(context_radius, 1, 2, 4, 5, CUDA_AUTO_TUNE_2D_TEMPLATED(
      PatchMatchMutationStepCUDAKernel,
      16, 16,
      inv_depth_map->width(), inv_depth_map->height(),
      0, stream,
      TEMPLATE_ARGUMENTS(_context_radius, false, true),
      /* kernel parameters */
      match_metric,
      max_normal_2d_length,
      CUDAUnprojectionLookup2D_(reference_unprojection_lookup),
      reference_image,
      reference_texture,
      stereo_tr_reference,
      stereo_camera,
      stereo_image,
      step_range,
      *inv_depth_map,
      *normals,
      *costs,
      *random_states,
      second_best_min_distance_factor,
      best_inv_depth_map ? *best_inv_depth_map : CUDABuffer_<float>()));
  CHECK_CUDA_NO_ERROR();
}


// (Mostly) auto-generated function.
typedef float Scalar;

// opcount = 243
__forceinline__ __device__ void ComputeResidualAndJacobian(
    Scalar cx, Scalar cy, Scalar fx, Scalar fy,
    Scalar inv_depth, Scalar n_x, Scalar n_y,
    Scalar nx, Scalar ny,
    Scalar other_nx, Scalar other_ny,
    Scalar ref_intensity,
    Scalar str_0_0, Scalar str_0_1, Scalar str_0_2, Scalar str_0_3,
    Scalar str_1_0, Scalar str_1_1, Scalar str_1_2, Scalar str_1_3,
    Scalar str_2_0, Scalar str_2_1, Scalar str_2_2, Scalar str_2_3,
    cudaTextureObject_t stereo_texture,
    Scalar* residuals, Scalar* jacobian) {
  const Scalar term0 = sqrt(-n_x*n_x - n_y*n_y + 1);
  const Scalar term1 = n_x*other_nx + n_y*other_ny - term0;
  const Scalar term2 = 1.0f/term1;
  const Scalar term3 = str_1_2*term2;
  const Scalar term4 = 1.0f/inv_depth;
  const Scalar term5 = n_x*nx;
  const Scalar term6 = n_y*ny;
  const Scalar term7 = -term0*term4 + term4*term5 + term4*term6;
  const Scalar term8 = other_nx*str_1_0*term2;
  const Scalar term9 = other_ny*str_1_1*term2;
  const Scalar term10 = str_1_3 + term3*term7 + term7*term8 + term7*term9;
  const Scalar term11 = str_2_2*term2;
  const Scalar term12 = other_nx*str_2_0*term2;
  const Scalar term13 = other_ny*str_2_1*term2;
  const Scalar term14 = str_2_3 + term11*term7 + term12*term7 + term13*term7;
  const Scalar term15 = 1.0f/term14;
  const Scalar term16 = fy*term15;
  
  float py = cy + term10*term16;
  int iy = static_cast<int>(py);
  const Scalar term17 = py - iy;
  
  const Scalar term18 = str_0_2*term2;
  const Scalar term19 = other_nx*str_0_0*term2;
  const Scalar term20 = other_ny*str_0_1*term2;
  const Scalar term21 = str_0_3 + term18*term7 + term19*term7 + term20*term7;
  const Scalar term22 = fx*term15;
  
  float px = cx + term21*term22;
  int ix = static_cast<int>(px);
  const Scalar term23 = px - ix;
  
  Scalar top_left = 255.0f * tex2D<float>(stereo_texture, ix + 0.5f, iy + 0.5f);
  Scalar top_right = 255.0f * tex2D<float>(stereo_texture, ix + 1.5f, iy + 0.5f);
  Scalar bottom_left = 255.0f * tex2D<float>(stereo_texture, ix + 0.5f, iy + 1.5f);
  Scalar bottom_right = 255.0f * tex2D<float>(stereo_texture, ix + 1.5f, iy + 1.5f);
  
  const Scalar term24 = -term23 + 1;
  const Scalar term25 = bottom_left*term24 + bottom_right*term23;
  const Scalar term26 = -term17 + 1;
  const Scalar term27 = term23*top_right;
  const Scalar term28 = term24*top_left;
  const Scalar term29 = -term17*(bottom_left - bottom_right) - term26*(top_left - top_right);
  const Scalar term30 = term4 * term4;
  const Scalar term31 = term0 - term5 - term6;
  const Scalar term32 = term30*term31;
  const Scalar term33 = term15 * term15;
  const Scalar term34 = term30*term31*term33*(term11 + term12 + term13);
  const Scalar term35 = term25 - term27 - term28;
  const Scalar term36 = 1.0f/term0;
  const Scalar term37 = n_x*term36;
  const Scalar term38 = nx*term4 + term37*term4;
  const Scalar term39 = -other_nx - term37;
  const Scalar term40 = term2 * term2;
  
  const Scalar term40Xterm7 = term40*term7;
  
  const Scalar term41 = str_0_2*term40Xterm7;
  const Scalar term42 = other_nx*str_0_0*term40Xterm7;
  const Scalar term43 = other_ny*str_0_1*term40Xterm7;
  const Scalar term44 = fx*term21*term33;
  const Scalar term45 = str_2_2*term40Xterm7;
  const Scalar term46 = other_nx*str_2_0*term40Xterm7;
  const Scalar term47 = other_ny*str_2_1*term40Xterm7;
  const Scalar term48 = -term11*term38 - term12*term38 - term13*term38 - term39*term45 - term39*term46 - term39*term47;
  const Scalar term49 = str_1_2*term40Xterm7;
  const Scalar term50 = other_nx*str_1_0*term40Xterm7;
  const Scalar term51 = other_ny*str_1_1*term40Xterm7;
  const Scalar term52 = fy*term10*term33;
  const Scalar term53 = n_y*term36;
  const Scalar term54 = ny*term4 + term4*term53;
  const Scalar term55 = -other_ny - term53;
  const Scalar term56 = -term11*term54 - term12*term54 - term13*term54 - term45*term55 - term46*term55 - term47*term55;
  
  *residuals = -ref_intensity + term17*term25 + term26*(term27 + term28);
  jacobian[0] = term29*(-fx*term21*term34 + term22*(term18*term32 + term19*term32 + term20*term32)) + term35*(-fy*term10*term34 + term16*(term3*term32 + term32*term8 + term32*term9));
  jacobian[1] = term29*(term22*(term18*term38 + term19*term38 + term20*term38 + term39*term41 + term39*term42 + term39*term43) + term44*term48) + term35*(term16*(term3*term38 + term38*term8 + term38*term9 + term39*term49 + term39*term50 + term39*term51) + term48*term52);
  jacobian[2] = term29*(term22*(term18*term54 + term19*term54 + term20*term54 + term41*term55 + term42*term55 + term43*term55) + term44*term56) + term35*(term16*(term3*term54 + term49*term55 + term50*term55 + term51*term55 + term54*term8 + term54*term9) + term52*term56);
}

// template <int kContextRadius>
// __global__ void PatchMatchOptimizationStepCUDAKernel(
//     int match_metric,
//     float max_normal_2d_length,
//     CUDAUnprojectionLookup2D_ unprojector,
//     CUDABuffer_<u8> reference_image,
//     cudaTextureObject_t reference_texture,
//     CUDAMatrix3x4 stereo_tr_reference,
//     PixelCornerProjector projector,
//     cudaTextureObject_t stereo_image,
//     CUDABuffer_<float> inv_depth_map,
//     CUDABuffer_<char2> normals,
//     CUDABuffer_<float> costs,
//     CUDABuffer_<curandState> random_states,
//     CUDABuffer_<float> lambda) {
//   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//   
//   if (x >= kContextRadius && y >= kContextRadius &&
//       x < inv_depth_map.width() - kContextRadius && y < inv_depth_map.height() - kContextRadius) {
//     float inv_depth = inv_depth_map(y, x);
//     char2 normal_xy_char = normals(y, x);
//     float2 normal_xy = make_float2(
//         normal_xy_char.x * (1 / 127.f), normal_xy_char.y * (1 / 127.f));
//     float2 nxy = unprojector.UnprojectPoint(x, y);
//     
//     // Gauss-Newton update equation coefficients.
//     float H[3 + 2 + 1] = {0, 0, 0, 0, 0, 0};
//     float b[3] = {0, 0, 0};
//     
//     #pragma unroll
//     for (int dy = -kContextRadius; dy <= kContextRadius; ++ dy) {
//       #pragma unroll
//       for (int dx = -kContextRadius; dx <= kContextRadius; ++ dx) {
//         float raw_residual;
//         float jacobian[3];
//         
//         float2 other_nxy = unprojector.UnprojectPoint(x + dx, y + dy);
//         
//         ComputeResidualAndJacobian(
//             projector.cx - 0.5f, projector.cy - 0.5f, projector.fx, projector.fy,
//             inv_depth, normal_xy.x, normal_xy.y,
//             nxy.x, nxy.y,
//             other_nxy.x, other_nxy.y,
//             reference_image(y + dy, x + dx),
//             stereo_tr_reference.row0.x, stereo_tr_reference.row0.y, stereo_tr_reference.row0.z, stereo_tr_reference.row0.w,
//             stereo_tr_reference.row1.x, stereo_tr_reference.row1.y, stereo_tr_reference.row1.z, stereo_tr_reference.row1.w,
//             stereo_tr_reference.row2.x, stereo_tr_reference.row2.y, stereo_tr_reference.row2.z, stereo_tr_reference.row2.w,
//             stereo_image,
//             &raw_residual, jacobian);
//         
//         // Accumulate
//         b[0] += raw_residual * jacobian[0];
//         b[1] += raw_residual * jacobian[1];
//         b[2] += raw_residual * jacobian[2];
//         
//         H[0] += jacobian[0] * jacobian[0];
//         H[1] += jacobian[0] * jacobian[1];
//         H[2] += jacobian[0] * jacobian[2];
//         
//         H[3] += jacobian[1] * jacobian[1];
//         H[4] += jacobian[1] * jacobian[2];
//         
//         H[5] += jacobian[2] * jacobian[2];
//       }
//     }
//     
//     /*// TEST: Optimize inv_depth only
//     b[0] = b[0] / H[0];
//     inv_depth -= b[0];*/
//     
//     // Levenberg-Marquardt
//     const float kDiagLambda = lambda(y, x);
//     H[0] *= kDiagLambda;
//     H[3] *= kDiagLambda;
//     H[5] *= kDiagLambda;
//     
//     // Solve for the update using Cholesky decomposition
//     // (H[0]          )   (H[0] H[1] H[2])   (x[0])   (b[0])
//     // (H[1] H[3]     ) * (     H[3] H[4]) * (x[1]) = (b[1])
//     // (H[2] H[4] H[5])   (          H[5])   (x[2])   (b[2])
//     H[0] = sqrtf(H[0]);
//     
//     H[1] = 1.f / H[0] * H[1];
//     H[3] = sqrtf(H[3] - H[1] * H[1]);
//     
//     H[2] = 1.f / H[0] * H[2];
//     H[4] = 1.f / H[3] * (H[4] - H[1] * H[2]);
//     H[5] = sqrtf(H[5] - H[2] * H[2] - H[4] * H[4]);
//     
//     // Re-use b for the intermediate vector
//     b[0] = (b[0] / H[0]);
//     b[1] = (b[1] - H[1] * b[0]) / H[3];
//     b[2] = (b[2] - H[2] * b[0] - H[4] * b[1]) / H[5];
//     
//     // Re-use b for the delta vector
//     b[2] = (b[2] / H[5]);
//     b[1] = (b[1] - H[4] * b[2]) / H[3];
//     b[0] = (b[0] - H[1] * b[1] - H[2] * b[2]) / H[0];
//     
//     // Apply the update, sanitize normal if necessary
//     inv_depth -= b[0];
//     normal_xy.x -= b[1];
//     normal_xy.y -= b[2];
//     
//     float length = sqrtf(normal_xy.x * normal_xy.x + normal_xy.y * normal_xy.y);
//     if (length > max_normal_2d_length) {
//       normal_xy.x *= max_normal_2d_length / length;
//       normal_xy.y *= max_normal_2d_length / length;
//     }
//     
//     // Test whether the update lowers the cost
//     float proposal_costs = ComputeCosts<kContextRadius>(
//         x, y,
//         normal_xy,
//         inv_depth,
//         unprojector,
//         reference_image,
//         reference_texture,
//         stereo_tr_reference,
//         projector,
//         stereo_image,
//         match_metric,
//         0,  // TODO: Update if using this function again
//         CUDABuffer_<float>());  // TODO: Update if using this function again
//     
//     if (!::isnan(proposal_costs) && !(proposal_costs >= costs(y, x))) {
//       costs(y, x) = proposal_costs;
//       normals(y, x) = make_char2(normal_xy.x * 127.f, normal_xy.y * 127.f);  // TODO: in this and similar places: rounding?
//       inv_depth_map(y, x) = inv_depth;
//       
//       lambda(y, x) *= 0.5f;
//     } else {
//       lambda(y, x) *= 2.f;
//     }
//   }
// }
// 
// void PatchMatchOptimizationStepCUDA(
//     cudaStream_t stream,
//     int match_metric,
//     int context_radius,
//     float max_normal_2d_length,
//     cudaTextureObject_t reference_unprojection_lookup,
//     const CUDABuffer_<u8>& reference_image,
//     cudaTextureObject_t reference_texture,
//     const CUDAMatrix3x4& stereo_tr_reference,
//     const PixelCornerProjector_& stereo_camera,
//     const cudaTextureObject_t stereo_image,
//     CUDABuffer_<float>* inv_depth_map,
//     CUDABuffer_<char2>* normals,
//     CUDABuffer_<float>* costs,
//     CUDABuffer_<curandState>* random_states,
//     CUDABuffer_<float>* lambda) {
//   CHECK_CUDA_NO_ERROR();
//   COMPILE_INT_4_OPTIONS(context_radius, 1, 2, 4, 5, CUDA_AUTO_TUNE_2D(
//       PatchMatchOptimizationStepCUDAKernel<_context_radius>,
//       16, 16,
//       inv_depth_map->width(), inv_depth_map->height(),
//       0, stream,
//       /* kernel parameters */
//       match_metric,
//       max_normal_2d_length,
//       CUDAUnprojectionLookup2D_(reference_unprojection_lookup),
//       reference_image,
//       reference_texture,
//       stereo_tr_reference,
//       stereo_camera,
//       stereo_image,
//       stereo_camera.width(),
//       stereo_camera.height(),
//       *inv_depth_map,
//       *normals,
//       *costs,
//       *random_states,
//       *lambda));
//   cudaDeviceSynchronize();
//   CHECK_CUDA_NO_ERROR();
// }


template <int kContextRadius>
__global__ void PatchMatchPropagationStepCUDAKernel(
    int match_metric,
    CUDAUnprojectionLookup2D_ unprojector,
    CUDABuffer_<u8> reference_image,
    cudaTextureObject_t reference_texture,
    CUDAMatrix3x4 stereo_tr_reference,
    PixelCornerProjector_ projector,
    cudaTextureObject_t stereo_image,
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<char2> normals,
    CUDABuffer_<float> costs,
    CUDABuffer_<curandState> random_states,
    float second_best_min_distance_factor,
    CUDABuffer_<float> best_inv_depth_map) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= kContextRadius && y >= kContextRadius &&
      x < inv_depth_map.width() - kContextRadius && y < inv_depth_map.height() - kContextRadius) {
    // "Pulling" the values inwards.
    float2 nxy = unprojector.UnprojectPoint(x, y);
    
    #pragma unroll
    for (int dy = -1; dy <= 1; ++ dy) {
      #pragma unroll
      for (int dx = -1; dx <= 1; ++ dx) {
        if ((dx == 0 && dy == 0) ||
            (dx != 0 && dy != 0)) {
          continue;
        }
        
        // Compute inv_depth for propagating the pixel at (x + dx, y + dy) to the center pixel.
        float2 other_nxy = unprojector.UnprojectPoint(x + dx, y + dy);
        
        float other_inv_depth = inv_depth_map(y + dy, x + dx);
        float other_depth = 1.f / other_inv_depth;
        
        char2 other_normal_xy_char = normals(y + dy, x + dx);
        const float2 other_normal_xy = make_float2(
            other_normal_xy_char.x * (1 / 127.f), other_normal_xy_char.y * (1 / 127.f));
        float other_normal_z = -sqrtf(1.f - other_normal_xy.x * other_normal_xy.x - other_normal_xy.y * other_normal_xy.y);
        
        float plane_d = (other_nxy.x * other_depth) * other_normal_xy.x + (other_nxy.y * other_depth) * other_normal_xy.y + other_depth * other_normal_z;
        
        float inv_depth = CalculatePlaneInvDepth2(plane_d, other_normal_xy, other_normal_z, nxy.x, nxy.y);
        
        // Test whether to propagate
        float proposal_costs = ComputeCosts<kContextRadius>(
            x, y,
            other_normal_xy,
            inv_depth,
            unprojector,
            reference_image,
            reference_texture,
            stereo_tr_reference,
            projector,
            stereo_image,
            match_metric,
            second_best_min_distance_factor,
            best_inv_depth_map);
        
        if (!::isnan(proposal_costs) && !(proposal_costs >= costs(y, x))) {
          costs(y, x) = proposal_costs;
          
          // NOTE: Other threads could read these values while they are written,
          //       but it should not be very severe if that happens.
          //       Could use ping-pong buffers to avoid that.
          normals(y, x) = make_char2(other_normal_xy.x * 127.f, other_normal_xy.y * 127.f);
          inv_depth_map(y, x) = inv_depth;
        }
      }  // loop over dx
    }  // loop over dy
  }
}

void PatchMatchPropagationStepCUDA(
    cudaStream_t stream,
    int match_metric,
    int context_radius,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& stereo_camera,
    const cudaTextureObject_t stereo_image,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<char2>* normals,
    CUDABuffer_<float>* costs,
    CUDABuffer_<curandState>* random_states,
    float second_best_min_distance_factor,
    CUDABuffer_<float>* best_inv_depth_map) {
  CHECK_CUDA_NO_ERROR();
  COMPILE_INT_4_OPTIONS(context_radius, 1, 2, 4, 5, CUDA_AUTO_TUNE_2D(
      PatchMatchPropagationStepCUDAKernel<_context_radius>,
      16, 16,
      inv_depth_map->width(), inv_depth_map->height(),
      0, stream,
      /* kernel parameters */
      match_metric,
      CUDAUnprojectionLookup2D_(reference_unprojection_lookup),
      reference_image,
      reference_texture,
      stereo_tr_reference,
      stereo_camera,
      stereo_image,
      *inv_depth_map,
      *normals,
      *costs,
      *random_states,
      second_best_min_distance_factor,
      best_inv_depth_map ? *best_inv_depth_map : CUDABuffer_<float>()));
  CHECK_CUDA_NO_ERROR();
}

template <int kContextRadius>
__global__ void PatchMatchDiscreteRefinementStepCUDAKernel(
    int match_metric,
    CUDAUnprojectionLookup2D_ unprojector,
    CUDABuffer_<u8> reference_image,
    cudaTextureObject_t reference_texture,
    CUDAMatrix3x4 stereo_tr_reference,
    PixelCornerProjector_ projector,
    cudaTextureObject_t stereo_image,
    int num_steps,
    float range_factor,
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<char2> normals,
    CUDABuffer_<float> costs) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= kContextRadius && y >= kContextRadius &&
      x < inv_depth_map.width() - kContextRadius && y < inv_depth_map.height() - kContextRadius) {
    float original_inv_depth = inv_depth_map(y, x);
    
    const char2 normal_char = normals(y, x);
    float2 normal = make_float2(
        normal_char.x * (1 / 127.f), normal_char.y * (1 / 127.f));
    
    for (int step = 0; step < num_steps; ++ step) {
      float proposed_inv_depth = (1 + range_factor * 2 * ((step / (num_steps - 1.f)) - 0.5f)) * original_inv_depth;
      
      // Test whether to accept the proposal
      float proposal_costs = ComputeCosts<kContextRadius>(
          x, y,
          normal,
          proposed_inv_depth,
          unprojector,
          reference_image,
          reference_texture,
          stereo_tr_reference,
          projector,
          stereo_image,
          match_metric,
          0,  // TODO: Update if using this function within the second best cost step
          inv_depth_map);  // TODO: Update if using this function within the second best cost step
      
      if (!::isnan(proposal_costs) && !(proposal_costs >= costs(y, x))) {
        costs(y, x) = proposal_costs;
        inv_depth_map(y, x) = proposed_inv_depth;
      }
    }
  }
}

void PatchMatchDiscreteRefinementStepCUDA(
    cudaStream_t stream,
    int match_metric,
    int context_radius,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& stereo_camera,
    const cudaTextureObject_t stereo_image,
    int num_steps,
    float range_factor,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<char2>* normals,
    CUDABuffer_<float>* costs) {
  CHECK_CUDA_NO_ERROR();
  COMPILE_INT_4_OPTIONS(context_radius, 1, 2, 4, 5, CUDA_AUTO_TUNE_2D_TEMPLATED(
      PatchMatchDiscreteRefinementStepCUDAKernel,
      16, 16,
      inv_depth_map->width(), inv_depth_map->height(),
      0, stream,
      TEMPLATE_ARGUMENTS(_context_radius),
      /* kernel parameters */
      match_metric,
      CUDAUnprojectionLookup2D_(reference_unprojection_lookup),
      reference_image,
      reference_texture,
      stereo_tr_reference,
      stereo_camera,
      stereo_image,
      num_steps,
      range_factor,
      *inv_depth_map,
      *normals,
      *costs));
  CHECK_CUDA_NO_ERROR();
}


template <int kContextRadius>
__global__ void PatchMatchLeftRightConsistencyCheckCUDAKernel(
    float lr_consistency_factor_threshold,
    CUDAUnprojectionLookup2D_ unprojector,
    CUDAMatrix3x4 stereo_tr_reference,
    PixelCornerProjector_ projector,
    CUDABuffer_<float> lr_consistency_inv_depth,
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<float> inv_depth_map_out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const float kInvalidInvDepth = 0;
  
  if (x >= kContextRadius && y >= kContextRadius &&
      x < inv_depth_map.width() - kContextRadius && y < inv_depth_map.height() - kContextRadius) {
    float inv_depth = inv_depth_map(y, x);
    float depth = 1 / inv_depth;
    
    float2 center_nxy = unprojector.UnprojectPoint(x, y);
    float3 reference_point = make_float3(depth * center_nxy.x, depth * center_nxy.y, depth);
    
    float3 pnxy = stereo_tr_reference * reference_point;
    if (pnxy.z <= 0.f) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      return;
    }
    
    const float2 rmin_pxy = projector.Project(pnxy);
    if (rmin_pxy.x < kContextRadius ||
        rmin_pxy.y < kContextRadius ||
        rmin_pxy.x >= projector.width - 1 - kContextRadius ||
        rmin_pxy.y >= projector.height - 1 - kContextRadius) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      return;
    }
    
    float lr_check_inv_depth = lr_consistency_inv_depth(rmin_pxy.y, rmin_pxy.x);
    if (lr_check_inv_depth == kInvalidInvDepth) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      return;
    }
    
    float factor = pnxy.z * lr_check_inv_depth;
    if (factor < 1) {
      factor = 1 / factor;
    }
    
    if (factor > lr_consistency_factor_threshold) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
    } else {
      inv_depth_map_out(y, x) = inv_depth;
    }
  } else if (x < inv_depth_map.width() && y < inv_depth_map.height()) {
    inv_depth_map_out(y, x) = kInvalidInvDepth;
  }
}

void PatchMatchLeftRightConsistencyCheckCUDA(
    cudaStream_t stream,
    int context_radius,
    float lr_consistency_factor_threshold,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& stereo_camera,
    const CUDABuffer_<float>& lr_consistency_inv_depth,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out) {
  CHECK_CUDA_NO_ERROR();
  COMPILE_INT_4_OPTIONS(context_radius, 1, 2, 4, 5, CUDA_AUTO_TUNE_2D(
      PatchMatchLeftRightConsistencyCheckCUDAKernel<_context_radius>,
      16, 16,
      inv_depth_map->width(), inv_depth_map->height(),
      0, stream,
      /* kernel parameters */
      lr_consistency_factor_threshold,
      CUDAUnprojectionLookup2D_(reference_unprojection_lookup),
      stereo_tr_reference,
      stereo_camera,
      lr_consistency_inv_depth,
      *inv_depth_map,
      *inv_depth_map_out));
  CHECK_CUDA_NO_ERROR();
}


// TODO: move to better place
__forceinline__ __device__ void CrossProduct(const float3& a, const float3& b, float3* result) {
  *result = make_float3(a.y * b.z - b.y * a.z,
                        b.x * a.z - a.x * b.z,
                        a.x * b.y - b.x * a.y);
}

// TODO: move to better place
__forceinline__ __device__ float Dot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// TODO: move to better place
__forceinline__ __device__ float Norm(const float3& vec) {
  return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

// TODO: move to better place
__forceinline__ __device__ float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// TODO: move to better place
__forceinline__ __device__ float SquaredLength(const float3& vec) {
  return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}

template <int kContextRadius>
__global__ void PatchMatchFilterOutliersCUDAKernel(
    float min_inv_depth,
    float required_range_min_depth,
    float required_range_max_depth,
    CUDAUnprojectionLookup2D_ unprojector,
    CUDABuffer_<u8> reference_image,
    cudaTextureObject_t reference_texture,
    CUDAMatrix3x4 stereo_tr_reference,
    CUDAMatrix3x4 reference_tr_stereo,
    PixelCornerProjector_ projector,
    cudaTextureObject_t stereo_image,
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<float> inv_depth_map_out,
    CUDABuffer_<char2> normals,
    CUDABuffer_<float> costs,
    float cost_threshold,
    float epipolar_gradient_threshold,
    float min_cos_angle,
    CUDABuffer_<float> second_best_costs,
    float second_best_min_cost_factor) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const float kInvalidInvDepth = 0;
  
  if (x >= kContextRadius && y >= kContextRadius &&
      x < inv_depth_map.width() - kContextRadius && y < inv_depth_map.height() - kContextRadius) {
    if (!(costs(y, x) <= cost_threshold) ||  // includes NaNs
        !(inv_depth_map(y, x) >= min_inv_depth)) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
    } else {
      // If there is another depth value with similar cost, reject the depth
      // estimate as ambiguous.
      if (second_best_min_cost_factor > 1) {
        if (!(second_best_costs(y, x) >= second_best_min_cost_factor * costs(y, x))) {  // includes NaNs
          inv_depth_map_out(y, x) = kInvalidInvDepth;
          return;
        }
      }
      
      // If at the maximum or minimum depth for this pixel the stereo frame
      // would not observe that point, discard the pixel (i.e., enforce that
      // this depth range is observed by both frames).
      // This is to protect against mistakes that often happen when the frames
      // overlap in only a small depth range and the actual depth is not within
      // that range.
      float2 center_nxy = unprojector.UnprojectPoint(x, y);
      float3 range_min_point = make_float3(required_range_min_depth * center_nxy.x, required_range_min_depth * center_nxy.y, required_range_min_depth);
      float3 range_max_point = make_float3(required_range_max_depth * center_nxy.x, required_range_max_depth * center_nxy.y, required_range_max_depth);
      
      float3 rmin_pnxy = stereo_tr_reference * range_min_point;
      if (rmin_pnxy.z <= 0.f) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        return;
      }
      
      const float2 rmin_pxy = projector.Project(rmin_pnxy);
      if (rmin_pxy.x < kContextRadius ||
          rmin_pxy.y < kContextRadius ||
          rmin_pxy.x >= projector.width - 1 - kContextRadius ||
          rmin_pxy.y >= projector.height - 1 - kContextRadius) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        return;
      }
      
      float3 rmax_pnxy = stereo_tr_reference * range_max_point;
      if (rmax_pnxy.z <= 0.f) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        return;
      }
      
      const float2 rmax_pxy = projector.Project(rmax_pnxy);
      if (rmax_pxy.x < kContextRadius ||
          rmax_pxy.y < kContextRadius ||
          rmax_pxy.x >= projector.width - 1 - kContextRadius ||
          rmax_pxy.y >= projector.height - 1 - kContextRadius) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        return;
      }
      
      // Texture filtering: remove pixels with too small gradients along the epipolar line direction in the patch used for matching.
      // TODO: The code below is only valid for the current ZNCC implementation, not SSD or Census!
      float inv_depth = inv_depth_map(y, x);
      
      const char2 normal_char = normals(y, x);
      float2 normal_xy = make_float2(
          normal_char.x * (1 / 127.f), normal_char.y * (1 / 127.f));
      
      const float normal_z =
          -sqrtf(1.f - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y);
      const float depth = 1.f / inv_depth;
      const float plane_d =
          (center_nxy.x * depth) * normal_xy.x +
          (center_nxy.y * depth) * normal_xy.y + depth * normal_z;
      
      float total_gradient_magnitude = 0;
          
      for (int sample = 0; sample < kNumSamples; ++ sample) {
        float dx = 1.25f * kContextRadius * kSamplesCUDA[sample][0];  // TODO: magic constant factor
        float dy = 1.25f * kContextRadius * kSamplesCUDA[sample][1];  // TODO: magic constant factor
        
        float2 nxy = unprojector.UnprojectPoint(x + dx, y + dy);  // NOTE: This is only approximate (bilinear interpolation of exact values sampled at pixel centers).
        float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x, nxy.y);
        
        float3 original_reference_point = make_float3(nxy.x * plane_depth, nxy.y * plane_depth, plane_depth);
        float3 original_stereo_point = stereo_tr_reference * original_reference_point;
        constexpr float kShiftZ = 0.01f;
        float3 shifted_stereo_point = make_float3(original_stereo_point.x, original_stereo_point.y, original_stereo_point.z + kShiftZ);
        float3 shifted_reference_point = reference_tr_stereo * shifted_stereo_point;
        
        const float2 shifted_projection = projector.Project(shifted_reference_point);
        float2 epipolar_direction = make_float2(shifted_projection.x - 0.5f - (x + dx),
                                                shifted_projection.y - 0.5f - (y + dy));
        
        float length = sqrtf(epipolar_direction.x * epipolar_direction.x + epipolar_direction.y * epipolar_direction.y);
        epipolar_direction = make_float2(epipolar_direction.x / length, epipolar_direction.y / length);  // Normalize to length of 1 pixel
        
        float reference_value = 255.f * tex2D<float>(reference_texture, x + dx + 0.5f, y + dy + 0.5f);
        float shifted_reference_value = 255.f * tex2D<float>(reference_texture, x + dx + 0.5f + epipolar_direction.x, y + dy + 0.5f + epipolar_direction.y);
        
        total_gradient_magnitude += fabs(shifted_reference_value - reference_value);
      }
      
      if (total_gradient_magnitude < epipolar_gradient_threshold) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        return;
      }
      
      // Angle filtering.
      // Estimate the surface normal from the depth map.
      float center_depth = 1.f / inv_depth_map(y, x);
      float right_depth = 1.f / inv_depth_map(y, x + 1);
      float left_depth = 1.f / inv_depth_map(y, x - 1);
      float bottom_depth = 1.f / inv_depth_map(y + 1, x);
      float top_depth = 1.f / inv_depth_map(y - 1, x);
      
      float2 left_nxy = unprojector.UnprojectPoint(x - 1, y);
      float3 left_point = make_float3(left_depth * left_nxy.x, left_depth * left_nxy.y, left_depth);
      
      float2 right_nxy = unprojector.UnprojectPoint(x + 1, y);
      float3 right_point = make_float3(right_depth * right_nxy.x, right_depth * right_nxy.y, right_depth);
      
      float2 top_nxy = unprojector.UnprojectPoint(x, y - 1);
      float3 top_point = make_float3(top_depth * top_nxy.x, top_depth * top_nxy.y, top_depth);
      
      float2 bottom_nxy = unprojector.UnprojectPoint(x, y + 1);
      float3 bottom_point = make_float3(bottom_depth * bottom_nxy.x, bottom_depth * bottom_nxy.y, bottom_depth);
      
      float3 center_point = make_float3(center_depth * center_nxy.x, center_depth * center_nxy.y, center_depth);
      
      constexpr float kRatioThreshold = 2.f;
      constexpr float kRatioThresholdSquared = kRatioThreshold * kRatioThreshold;
      
      float left_dist_squared = SquaredLength(left_point - center_point);
      float right_dist_squared = SquaredLength(right_point - center_point);
      float left_right_ratio = left_dist_squared / right_dist_squared;
      float3 left_to_right;
      if (left_right_ratio < kRatioThresholdSquared &&
          left_right_ratio > 1.f / kRatioThresholdSquared) {
        left_to_right = right_point - left_point;
      } else if (left_dist_squared < right_dist_squared) {
        left_to_right = center_point - left_point;
      } else {  // left_dist_squared >= right_dist_squared
        left_to_right = right_point - center_point;
      }
      
      float bottom_dist_squared = SquaredLength(bottom_point - center_point);
      float top_dist_squared = SquaredLength(top_point - center_point);
      float bottom_top_ratio = bottom_dist_squared / top_dist_squared;
      float3 bottom_to_top;
      if (bottom_top_ratio < kRatioThresholdSquared &&
          bottom_top_ratio > 1.f / kRatioThresholdSquared) {
        bottom_to_top = top_point - bottom_point;
      } else if (bottom_dist_squared < top_dist_squared) {
        bottom_to_top = center_point - bottom_point;
      } else {  // bottom_dist_squared >= top_dist_squared
        bottom_to_top = top_point - center_point;
      }
      
      float3 normal;
      CrossProduct(left_to_right, bottom_to_top, &normal);
      
      // Apply angle threshold.
      const float normal_length = Norm(normal);
      const float point_distance = Norm(center_point);
      const float view_cos_angle = Dot(normal, center_point) / (normal_length * point_distance);
      
      if (view_cos_angle > min_cos_angle) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
      } else {
        inv_depth_map_out(y, x) = inv_depth_map(y, x);
      }
    }
  } else if (x < inv_depth_map.width() && y < inv_depth_map.height()) {
    inv_depth_map_out(y, x) = kInvalidInvDepth;
  }
}

void PatchMatchFilterOutliersCUDA(
    cudaStream_t stream,
    int context_radius,
    float min_inv_depth,
    float required_range_min_depth,
    float required_range_max_depth,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const CUDAMatrix3x4& reference_tr_stereo,
    const PixelCornerProjector_& stereo_camera,
    const cudaTextureObject_t stereo_image,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out,
    CUDABuffer_<char2>* normals,
    CUDABuffer_<float>* costs,
    float cost_threshold,
    float epipolar_gradient_threshold,
    float min_cos_angle,
    CUDABuffer_<float>* second_best_costs,
    float second_best_min_cost_factor) {
  CHECK_CUDA_NO_ERROR();
  COMPILE_INT_4_OPTIONS(context_radius, 1, 2, 4, 5, CUDA_AUTO_TUNE_2D(
      PatchMatchFilterOutliersCUDAKernel<_context_radius>,
      16, 16,
      inv_depth_map->width(), inv_depth_map->height(),
      0, stream,
      /* kernel parameters */
      min_inv_depth,
      required_range_min_depth,
      required_range_max_depth,
      CUDAUnprojectionLookup2D_(reference_unprojection_lookup),
      reference_image,
      reference_texture,
      stereo_tr_reference,
      reference_tr_stereo,
      stereo_camera,
      stereo_image,
      *inv_depth_map,
      *inv_depth_map_out,
      *normals,
      *costs,
      cost_threshold,
      epipolar_gradient_threshold,
      min_cos_angle,
      *second_best_costs,
      second_best_min_cost_factor));
  CHECK_CUDA_NO_ERROR();
}

__global__ void MedianFilterDepthMap3x3CUDAKernel(
    int context_radius,
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<float> inv_depth_map_out,
    CUDABuffer_<float> costs,
    CUDABuffer_<float> costs_out,
    CUDABuffer_<float> second_best_costs,
    CUDABuffer_<float> second_best_costs_out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const float kInvalidInvDepth = 0;  // TODO: De-duplicate with above
  
  if (x >= context_radius && y >= context_radius &&
      x < inv_depth_map.width() - context_radius && y < inv_depth_map.height() - context_radius) {
    // Collect valid depth values of 3x3 neighborhood
    int count = 1;
    float inv_depths[9];
    float cost[9];
    float second_best_cost[9];
    
    inv_depths[0] = inv_depth_map(y, x);
    if (inv_depths[0] == kInvalidInvDepth) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      costs_out(y, x) = CUDART_NAN_F;
      second_best_costs_out(y, x) = CUDART_NAN_F;
      return;
    }
    cost[0] = costs(y, x);
    second_best_cost[0] = second_best_costs(y, x);
    
    #pragma unroll
    for (int dy = -1; dy <= 1; ++ dy) {
      if (y + dy < context_radius || y + dy >= inv_depth_map.height() - context_radius) {
        continue;
      }
      
      #pragma unroll
      for (int dx = -1; dx <= 1; ++ dx) {
        if (dy == 0 && dx == 0) {
          continue;
        }
        
        if (x + dx < context_radius || x + dx >= inv_depth_map.width() - context_radius) {
          continue;
        }
        
        float inv_depth = inv_depth_map(y + dy, x + dx);
        if (inv_depth != kInvalidInvDepth) {
          inv_depths[count] = inv_depth;
          cost[count] = costs(y + dy, x + dx);
          second_best_cost[count] = second_best_costs(y + dy, x + dx);
          ++ count;
        }
      }
    }
    
    // Sort depth values up to the middle of the maximum count
    for (int i = 0; i <= 4; ++ i) {
      for (int k = i + 1; k < 9; ++ k) {
        if (k < count && inv_depths[i] > inv_depths[k]) {
          // Swap.
          float temp = inv_depths[i];
          inv_depths[i] = inv_depths[k];
          inv_depths[k] = temp;
          
          temp = cost[i];
          cost[i] = cost[k];
          cost[k] = temp;
          
          temp = second_best_cost[i];
          second_best_cost[i] = second_best_cost[k];
          second_best_cost[k] = temp;
        }
      }
    }
    
    // Assign the median
    if (count % 2 == 1) {
      inv_depth_map_out(y, x) = inv_depths[count / 2];
      costs_out(y, x) = cost[count / 2];
      second_best_costs_out(y, x) = second_best_cost[count / 2];
    } else {
      // For disambiguation in the even-count case, use the value which is
      // closer to the average of the two middle values.
      float average = 0.5f * (inv_depths[count / 2 - 1] + inv_depths[count / 2]);
      if (fabs(average - inv_depths[count / 2 - 1]) <
          fabs(average - inv_depths[count / 2])) {
        inv_depth_map_out(y, x) = inv_depths[count / 2 - 1];
        costs_out(y, x) = cost[count / 2 - 1];
        second_best_costs_out(y, x) = second_best_cost[count / 2 - 1];
      } else {
        inv_depth_map_out(y, x) = inv_depths[count / 2];
        costs_out(y, x) = cost[count / 2];
        second_best_costs_out(y, x) = second_best_cost[count / 2];
      }
    }
  } else if (x < inv_depth_map_out.width() && y < inv_depth_map_out.height()) {
    inv_depth_map_out(y, x) = kInvalidInvDepth;
    costs_out(y, x) = CUDART_NAN_F;
    second_best_costs_out(y, x) = CUDART_NAN_F;
  }
}

void MedianFilterDepthMap3x3CUDA(
    cudaStream_t stream,
    int context_radius,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out,
    CUDABuffer_<float>* costs,
    CUDABuffer_<float>* costs_out,
    CUDABuffer_<float>* second_best_costs,
    CUDABuffer_<float>* second_best_costs_out) {
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      MedianFilterDepthMap3x3CUDAKernel,
      32, 32,
      inv_depth_map->width(), inv_depth_map->height(),
      0, stream,
      /* kernel parameters */
      context_radius,
      *inv_depth_map,
      *inv_depth_map_out,
      *costs,
      *costs_out,
      *second_best_costs,
      *second_best_costs_out);
  CHECK_CUDA_NO_ERROR();
}


__global__ void BilateralFilterCUDAKernel(
    float denom_xy,
    float denom_value,
    int radius,
    int radius_squared,
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<float> inv_depth_map_out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const float kInvalidInvDepth = 0;  // TODO: De-duplicate with above
  
  if (x < inv_depth_map_out.width() && y < inv_depth_map_out.height()) {
    const float center_value = inv_depth_map(y, x);
    if (center_value == kInvalidInvDepth) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      return;
    }
    
    // Bilateral filtering.
    float sum = 0;
    float weight = 0;
    
    const int min_y = max(static_cast<int>(0), static_cast<int>(y - radius));
    const int max_y = min(static_cast<int>(inv_depth_map_out.height() - 1), static_cast<int>(y + radius));
    for (int sample_y = min_y; sample_y <= max_y; ++ sample_y) {
      const int dy = sample_y - y;
      
      const int min_x = max(static_cast<int>(0), static_cast<int>(x - radius));
      const int max_x = min(static_cast<int>(inv_depth_map_out.width() - 1), static_cast<int>(x + radius));
      for (int sample_x = min_x; sample_x <= max_x; ++ sample_x) {
        const int dx = sample_x - x;
        
        const int grid_distance_squared = dx * dx + dy * dy;
        if (grid_distance_squared > radius_squared) {
          continue;
        }
        
        const float sample = inv_depth_map(sample_y, sample_x);
        if (sample == kInvalidInvDepth) {
          continue;
        }
        
        float value_distance_squared = center_value - sample;
        value_distance_squared *= value_distance_squared;
        float w = exp(-grid_distance_squared / denom_xy + -value_distance_squared / denom_value);
        sum += w * sample;
        weight += w;
      }
    }
    
    inv_depth_map_out(y, x) = (weight == 0) ? kInvalidInvDepth : (sum / weight);
  }
}

void BilateralFilterCUDA(
    cudaStream_t stream,
    float sigma_xy,
    float sigma_value,
    float radius_factor,
    const CUDABuffer_<float>& inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out) {
  CHECK_CUDA_NO_ERROR();
  
  int radius = radius_factor * sigma_xy + 0.5f;
  
  CUDA_AUTO_TUNE_2D(
      BilateralFilterCUDAKernel,
      32, 32,
      inv_depth_map_out->width(), inv_depth_map_out->height(),
      0, stream,
      /* kernel parameters */
      2.0f * sigma_xy * sigma_xy,
      2.0f * sigma_value * sigma_value,
      radius,
      radius * radius,
      inv_depth_map,
      *inv_depth_map_out);
  CHECK_CUDA_NO_ERROR();
}


__global__ void FillHolesCUDAKernel(
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<float> inv_depth_map_out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const float kInvalidInvDepth = 0;  // TODO: De-duplicate with above
  
  if (x < inv_depth_map_out.width() && y < inv_depth_map_out.height()) {
    const float center_inv_depth = inv_depth_map(y, x);
    if (center_inv_depth != kInvalidInvDepth ||
        x < 1 ||
        y < 1 ||
        x >= inv_depth_map.width() - 1 ||
        y >= inv_depth_map.height() - 1) {
      inv_depth_map_out(y, x) = center_inv_depth;
      return;
    }
    
    // Get the average depth of the neighbor pixels.
    float sum = 0;
    int count = 0;
    
    #pragma unroll
    for (int dy = -1; dy <= 1; ++ dy) {
      #pragma unroll
      for (int dx = -1; dx <= 1; ++ dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        
        float inv_depth = inv_depth_map(y + dy, x + dx);
        if (inv_depth != kInvalidInvDepth) {
          sum += inv_depth;
          ++ count;
        }
      }
    }
    
    float avg_inv_depth = sum / count;
    
    // Fill in this pixel if there are at least a minimum number of valid
    // neighbor pixels nearby which have similar depth.
    constexpr float kSimilarDepthFactorThreshold = 1.01f;  // TODO: Make parameter
    constexpr int kMinSimilarPixelsForFillIn = 6;  // TODO: Make parameter
    
    sum = 0;
    count = 0;
    
    #pragma unroll
    for (int dy = -1; dy <= 1; ++ dy) {
      #pragma unroll
      for (int dx = -1; dx <= 1; ++ dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        
        float inv_depth = inv_depth_map(y + dy, x + dx);
        if (inv_depth != kInvalidInvDepth) {
          float factor = inv_depth / avg_inv_depth;
          if (factor < 1) {
            factor = 1 / factor;
          }
          
          if (factor <= kSimilarDepthFactorThreshold) {
            sum += inv_depth;
            ++ count;
          }
        }
      }
    }
    
    inv_depth_map_out(y, x) = (count >= kMinSimilarPixelsForFillIn) ? (sum / count) : kInvalidInvDepth;
  }
}

void FillHolesCUDA(
    cudaStream_t stream,
    const CUDABuffer_<float>& inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out) {
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      FillHolesCUDAKernel,
      32, 32,
      inv_depth_map_out->width(), inv_depth_map_out->height(),
      0, stream,
      /* kernel parameters */
      inv_depth_map,
      *inv_depth_map_out);
  CHECK_CUDA_NO_ERROR();
}

}
