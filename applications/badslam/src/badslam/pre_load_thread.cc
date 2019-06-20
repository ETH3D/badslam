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

#include "badslam/pre_load_thread.h"

namespace vis {

PreLoadThread::PreLoadThread(RGBDVideo<Vec3u8, u16>* rgbd_video) {
  rgbd_video_ = rgbd_video;
  
  done_ = true;
  preload_frame_index_ = -1;
  thread_exit_requested_ = false;
  thread_.reset(new thread(bind(&PreLoadThread::ThreadMain, this)));
}

PreLoadThread::~PreLoadThread() {
  RequestExitAndWaitForIt();
}

void PreLoadThread::RequestExitAndWaitForIt() {
  if (!thread_) {
    return;
  }
  
  unique_lock<mutex> input_lock(input_mutex_);
  thread_exit_requested_ = true;
  input_lock.unlock();
  new_input_available_condition_.notify_all();
  
  thread_->join();
  thread_.reset();
}

void PreLoadThread::ThreadMain() {
  int preload_frame_index;
  
  while (true) {
    // Wait until there is a preload request.
    // ### Input data lock start ###
    unique_lock<mutex> input_lock(input_mutex_);
    while (preload_frame_index_ == -1 && !thread_exit_requested_) {
      done_ = true;
      all_work_done_condition_.notify_all();
      new_input_available_condition_.wait(input_lock);
    }
    
    // Exit if requested.
    if (thread_exit_requested_) {
      done_ = true;
      all_work_done_condition_.notify_all();
      return;
    }
    
    preload_frame_index = preload_frame_index_;
    preload_frame_index_ = -1;
    input_lock.unlock();
    // ### Input data lock end ###
    
    rgbd_video_->color_frame_mutable(preload_frame_index)->GetImage().get();
    rgbd_video_->depth_frame_mutable(preload_frame_index)->GetImage().get();
  }
}

void PreLoadThread::PreLoad(int frame_index) {
  unique_lock<mutex> input_lock(input_mutex_);
  done_ = false;
  preload_frame_index_ = frame_index;
  input_lock.unlock();
  
  new_input_available_condition_.notify_all();
}

void PreLoadThread::WaitUntilDone() {
  unique_lock<mutex> input_lock(input_mutex_);
  while (!done_) {
    all_work_done_condition_.wait(input_lock);
  }
}

}
