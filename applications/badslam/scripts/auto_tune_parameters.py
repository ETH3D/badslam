# coding=utf-8
# Copyright 2019 ETH Zürich, Thomas Schöps
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import os
import subprocess
import sys


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python auto_tune_parameters.py <path_to_badslam_executable> <badslam_parameters>')
    
    # kernel_name : (width, height, runtime)
    best_results = dict()
    
    for tuning_iteration in range(0, 7):
        call = sys.argv[1:]
        call.append('--auto_tuning_iteration')
        call.append(str(tuning_iteration))
        
        print('Running: ' + ' '.join(call))
        proc = subprocess.Popen(call)
        return_code = proc.wait()
        del proc
        if return_code != 0:
            print('Program call failed (return code: ' + str(return_code) + ')')
            sys.exit(1)
        
        with open('auto_tuning_iteration_' + str(tuning_iteration) + '.txt', 'rb') as tuning_file:
            for line in tuning_file.readlines():
                if len(line) == 0:
                    continue
                words = line.rstrip('\n').split()
                kernel_name = words[0]
                width = words[1]
                height = words[2]
                runtime = float(words[3])
                
                if not kernel_name in best_results:
                    best_results[kernel_name] = (width, height, runtime)
                else:
                    # NOTE: It may happen that there are multiple results with
                    #       the same block size in different files. Those results
                    #       should be averaged before comparing them, instead of
                    #       comparing all of them individually.
                    prev_result = best_results[kernel_name]
                    if runtime < prev_result[2]:
                        best_results[kernel_name] = (width, height, runtime)
    
    with open('auto_tuning_result.txt', 'wb') as out_file:
        for kernel_name, kernel_info in best_results.iteritems():
            out_file.write(kernel_name + ' ' + kernel_info[0] + ' ' + kernel_info[1] + ' ' + str(kernel_info[2]) + '\n')
