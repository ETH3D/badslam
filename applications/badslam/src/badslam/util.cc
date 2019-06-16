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


#include "badslam/util.h"

#include <cstdio>
#include <string.h>
#ifndef WIN32
#include <termios.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <libvis/cuda/cuda_util.h>

namespace vis {

#ifndef WIN32
// Implementation from: https://stackoverflow.com/questions/421860
char GetKeyInput() {
  char buf = 0;
  struct termios old = {0};
  if (tcgetattr(0, &old) < 0) {
    perror("tcsetattr()");
  }
  old.c_lflag &= ~ICANON;
  old.c_lflag &= ~ECHO;
  old.c_cc[VMIN] = 1;
  old.c_cc[VTIME] = 0;
  if (tcsetattr(0, TCSANOW, &old) < 0) {
    perror("tcsetattr ICANON");
  }
  if (read(0, &buf, 1) < 0) {
    perror ("read()");
  }
  old.c_lflag |= ICANON;
  old.c_lflag |= ECHO;
  if (tcsetattr(0, TCSADRAIN, &old) < 0) {
    perror ("tcsetattr ~ICANON");
  }
  return (buf);
}

int PollKeyInput() {
  int character;
  struct termios orig_term_attr;
  struct termios new_term_attr;
  
  /* set the terminal to raw mode */
  tcgetattr(fileno(stdin), &orig_term_attr);
  memcpy(&new_term_attr, &orig_term_attr, sizeof(struct termios));
  new_term_attr.c_lflag &= ~(ECHO|ICANON);
  new_term_attr.c_cc[VTIME] = 0;
  new_term_attr.c_cc[VMIN] = 0;
  tcsetattr(fileno(stdin), TCSANOW, &new_term_attr);
  
  /* read a character from the stdin stream without blocking */
  /*   returns EOF (-1) if no character is available */
  character = fgetc(stdin);
  
  if (character != EOF) {
    std::getchar();  // remove the character from the buffer
  }
  
  /* restore the original terminal attributes */
  tcsetattr(fileno(stdin), TCSANOW, &orig_term_attr);
  
  return character;
}
#endif

void PrintGPUMemoryUsage() {
  size_t free_bytes;
  size_t total_bytes;
  CUDA_CHECKED_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t used_bytes = total_bytes - free_bytes;
  
  constexpr double kBytesToMiB = 1.0 / (1024.0 * 1024.0);
  LOG(INFO) << "GPU memory used: " <<
               kBytesToMiB * used_bytes << " MiB, free: " <<
               kBytesToMiB * free_bytes << " MiB";
}

SE3f AveragePose(int count, SE3f* poses) {
  // TODO: Cast to double is probably not necessary?
  
  Eigen::Matrix3d accumulated_rotations;
  accumulated_rotations.setZero();
  Eigen::Vector3d accumulated_translations;
  accumulated_translations.setZero();
  
  for (int i = 0; i < count; ++ i) {
    accumulated_rotations += poses[i].so3().matrix().cast<double>();
    accumulated_translations += poses[i].translation().cast<double>();
  }
  
  Sophus::SE3f result;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(accumulated_rotations, Eigen::ComputeFullU | Eigen::ComputeFullV);
  result.setRotationMatrix((svd.matrixU() * svd.matrixV().transpose()).cast<float>());
  result.translation() = (accumulated_translations / (1.0 * count)).cast<float>();
  return result;
}

}
