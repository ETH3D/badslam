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

#pragma once

#include <cuda_runtime.h>
#include <libvis/libvis.h>

namespace vis {

// This bit is set in the depth value of pixels which should not be used (since
// for example their normal direction is unknown).
constexpr u16 kInvalidDepthBit = 1 << 15;

// This value is used as depth value for pixels with unknown depth.
constexpr u16 kUnknownDepth = 65535;  // highest 16-bit unsigned value

// This flag is set for active surfels.
constexpr u8 kSurfelActiveFlag = 1 << 0;

// The number of buffers to use for surfel merging. In effect, this is the
// number of different surfels that can be stored for one pixel. If there are
// more individual surfels in one pixel that should not be merged than this
// value, then we might overlook cases where surfels should be merged. However,
// the more buffers are used, the slower it gets.
constexpr int kMergeBufferCount = 3;

// Invalid index value used in the "supporting surfels" buffers.
constexpr u32 kInvalidIndex = 4294967295;  // = numeric_limits<u32>::max();

// Threshold on angle difference between surfels to consider them compatible.
// constexpr float normal_compatibility_threshold_deg = 40;  // TODO: make parameter?
constexpr float cos_normal_compatibility_threshold = 0.76604f;  // = cosf(M_PI / 180.f * normal_compatibility_threshold_deg);


// Scalar type used in the PCG-based Gauss-Newton optimization (float or double).
typedef float PCGScalar;


// The surfel structure is stored in large buffers. It is organized
// such that each row stores one attribute and each column stores the
// attribute values for one surfel.

constexpr int kSurfelX = 0;  // float
constexpr int kSurfelY = 1;  // float
constexpr int kSurfelZ = 2;  // float
constexpr int kSurfelNormal = 3;  // (2 bits unused, s10 z, s10 y, s10 x)
constexpr int kSurfelRadiusSquared = 4;  // float
constexpr int kSurfelColor = 5;  // (u8 r, u8 g, u8 b, 8 bits unused)
constexpr int kSurfelDescriptor1 = 6;  // float
constexpr int kSurfelDescriptor2 = 7;  // float

constexpr int kSurfelAccum0 = 8;  // float
constexpr int kSurfelAccum1 = 9;  // float
constexpr int kSurfelAccum2 = 10;  // float
constexpr int kSurfelAccum3 = 11;  // float
constexpr int kSurfelAccum4 = 12;  // float
constexpr int kSurfelAccum5 = 13;  // float
constexpr int kSurfelAccum6 = 14;  // float
constexpr int kSurfelAccum7 = 15;  // float
constexpr int kSurfelAccum8 = 16;  // float

// This first number of attributes will be copied if a surfel is copied to a
// different index.
constexpr int kSurfelDataAttributeCount = 8;
// Total surfel attribute count, including temporary attributes (which are not
// preserved during copies).
constexpr int kSurfelAttributeCount = 17;

}
