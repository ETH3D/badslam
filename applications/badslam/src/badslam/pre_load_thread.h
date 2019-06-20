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

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <libvis/eigen.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video.h>

namespace vis {

// Represents a thread which pre-loads the next RGB-D frame in a dataset from
// disk while the current one is being processed to reduce the I/O-induced
// waiting time.
class PreLoadThread {
 public:
  // Constructor, starts the thread.
  PreLoadThread(RGBDVideo<Vec3u8, u16>* rgbd_video);
  
  ~PreLoadThread();
  
  // Signals the thread to exit and waits for this to happen.
  void RequestExitAndWaitForIt();
  
  // Requests pre-loading of the frame with the given number within the
  // RGBDVideo passed to the constructor.
  void PreLoad(int frame_index);
  
  // Waits until pre-loading of the last requested frame finished.
  void WaitUntilDone();
  
 private:
  void ThreadMain();
  
  RGBDVideo<Vec3u8, u16>* rgbd_video_;
  
  mutex input_mutex_;
  condition_variable new_input_available_condition_;
  atomic<bool> done_;
  condition_variable all_work_done_condition_;
  atomic<int> preload_frame_index_;
  
  atomic<bool> thread_exit_requested_;
  unique_ptr<thread> thread_;
};

}
