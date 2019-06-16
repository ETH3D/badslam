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

#include "badslam/trajectory_deformation.h"

namespace vis {

void RememberKeyframePoses(
    DirectBA* dense_ba,
    vector<SE3f>* original_keyframe_T_global) {
  original_keyframe_T_global->resize(dense_ba->keyframes().size());
  for (usize keyframe_index = 0; keyframe_index < dense_ba->keyframes().size(); ++ keyframe_index) {
    if (dense_ba->keyframes()[keyframe_index]) {
      original_keyframe_T_global->at(keyframe_index) = dense_ba->keyframes()[keyframe_index]->frame_T_global();
    }
  }
}


void ExtrapolateAndInterpolateKeyframePoseChanges(
    u32 start_frame,
    u32 end_frame,
    DirectBA* dense_ba,
    const vector<SE3f>& original_keyframe_T_global,
    RGBDVideo<Vec3u8, u16>* rgbd_video) {
  end_frame = std::min<int>(end_frame, rgbd_video->frame_count() - 1);
  
  usize prev_keyframe_index = 0;
  usize next_keyframe_index = 0;
  
  for (usize other_frame_index = start_frame; other_frame_index <= end_frame; ++ other_frame_index) {
    // Check whether prev_keyframe_index must be increased.
    while (next_keyframe_index < dense_ba->keyframes().size() &&
            dense_ba->keyframes()[next_keyframe_index]->frame_index() <= other_frame_index) {
      prev_keyframe_index = next_keyframe_index;
      ++ next_keyframe_index;
      
      while (next_keyframe_index < dense_ba->keyframes().size() &&
            !dense_ba->keyframes()[next_keyframe_index]) {
        ++ next_keyframe_index;
      }
    }
    
    Keyframe* prev_keyframe = dense_ba->keyframes()[prev_keyframe_index].get();
    Keyframe* next_keyframe = (next_keyframe_index < dense_ba->keyframes().size()) ? dense_ba->keyframes()[next_keyframe_index].get() : nullptr;
    
    if (prev_keyframe->frame_index() == other_frame_index) {
      // This is a keyframe, no interpolation or extrapolation is necessary.
      continue;
    }
    
    // Determine the offset for other_frame_index.
    SE3f new_global_T_other_frame;
    if (next_keyframe == nullptr ||  // Extrapolate at the end.
        prev_keyframe->frame_index() > other_frame_index) {  // Extrapolate at the start.
      SE3f old_kf_T_other_frame =
          original_keyframe_T_global[prev_keyframe_index] *
          rgbd_video->depth_frame_mutable(other_frame_index)->global_T_frame();
      new_global_T_other_frame =
          prev_keyframe->global_T_frame() *
          old_kf_T_other_frame;
    } else {
      // Interpolate.
      SE3f old_prev_kf_T_other_frame =
          original_keyframe_T_global[prev_keyframe_index] *
          rgbd_video->depth_frame_mutable(other_frame_index)->global_T_frame();
      SE3f new_global_T_other_frame_from_prev =
          prev_keyframe->global_T_frame() *
          old_prev_kf_T_other_frame;
      SE3f other_old_T_other_new_from_prev =
          rgbd_video->depth_frame_mutable(other_frame_index)->frame_T_global() *
          new_global_T_other_frame_from_prev;
      
      SE3f old_next_kf_T_other_frame =
          original_keyframe_T_global[next_keyframe_index] *
          rgbd_video->depth_frame_mutable(other_frame_index)->global_T_frame();
      SE3f new_global_T_other_frame_from_next =
          next_keyframe->global_T_frame() *
          old_next_kf_T_other_frame;
      SE3f other_old_T_other_new_from_next =
          rgbd_video->depth_frame_mutable(other_frame_index)->frame_T_global() *
          new_global_T_other_frame_from_next;
      
      u32 prev_keyframe_frame_index = prev_keyframe->frame_index();
      u32 next_keyframe_frame_index = next_keyframe->frame_index();
      float factor = (other_frame_index - prev_keyframe_frame_index) *
                      1.0f / (next_keyframe_frame_index - prev_keyframe_frame_index);
      
      SE3f interpolated_other_old_T_other_new;
      interpolated_other_old_T_other_new.translation() =
          (1 - factor) * other_old_T_other_new_from_prev.translation() +
          (factor) * other_old_T_other_new_from_next.translation();
      interpolated_other_old_T_other_new.setQuaternion(
          other_old_T_other_new_from_prev.unit_quaternion().slerp(
              factor, other_old_T_other_new_from_next.unit_quaternion()));
      
      new_global_T_other_frame =
          rgbd_video->depth_frame_mutable(other_frame_index)->global_T_frame() *
          interpolated_other_old_T_other_new;
    }
    
    rgbd_video->depth_frame_mutable(other_frame_index)->SetGlobalTFrame(new_global_T_other_frame);
    rgbd_video->color_frame_mutable(other_frame_index)->SetGlobalTFrame(new_global_T_other_frame);
  }
}

}
