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


#include "badslam/convergence_analysis.h"

#include <fstream>
#include <memory>

#include <libvis/eigen.h>
#include <libvis/logging.h>

namespace vis {

struct TrackingSample {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // Pose deltas
  vector<Eigen::Matrix<float, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 6, 1>>> x;
  // Residual sums before the iteration
  vector<float> residual_sums;
  // initial_global_T_frame is assumed to be identity
  SE3f final_global_T_frame;
  // Index of iteration after which this sample is assumed to be converged
  int converged_after_iteration;
  
  bool is_valid;
};


struct ConvergenceCriterion {
  virtual string Name() const = 0;
  virtual bool IsConverged(const TrackingSample& sample, int iteration, int scale, float scaling_factor, float damping) const = 0;
};

struct ConvergenceCriterion_DeltaThreshold : public ConvergenceCriterion {
  ConvergenceCriterion_DeltaThreshold(float translation_threshold, float rotation_threshold)
      : translation_threshold(translation_threshold),
        rotation_threshold(rotation_threshold) {}
  
  virtual string Name() const override {
    ostringstream o;
    o << "DeltaThreshold - translation: " << translation_threshold
      << ", rotation: " << rotation_threshold;
    return o.str();
  }
  
  virtual bool IsConverged(const TrackingSample& sample, int iteration, int /*scale*/, float scaling_factor, float /*damping*/) const override {
    // Scale the rotation part to the translation scale and then use the translation criterion on the whole vector norm.
    Eigen::Matrix<float, 6, 1> scaled_x = sample.x[iteration];
    scaled_x.bottomRows<3>() *= translation_threshold / rotation_threshold;
    return (scaled_x.squaredNorm() < scaling_factor * scaling_factor * translation_threshold);
  }
  
  float translation_threshold;
  float rotation_threshold;
};

struct ConvergenceCriterion_CostNonDecrease : public ConvergenceCriterion {
  ConvergenceCriterion_CostNonDecrease(int iteration_count)
      : iteration_count(iteration_count) {}
  
  virtual string Name() const override {
    ostringstream o;
    o << "ConvergenceCriterion_CostNonDecrease - for iterations: " << iteration_count;
    return o.str();
  }
  
  virtual bool IsConverged(const TrackingSample& sample, int iteration, int /*scale*/, float /*scaling_factor*/, float /*damping*/) const override {
    if (iteration < iteration_count) {
      return false;
    } else {
      for (int i = iteration; i > iteration - iteration_count; -- i) {
        if (sample.residual_sums[i] < sample.residual_sums[i - 1]) {
          return false;
        }
      }
      return true;
    }
  }
  
  int iteration_count;
};

struct ConvergenceCriterion_And : public ConvergenceCriterion {
  void AddCriterion(shared_ptr<ConvergenceCriterion> criterion) {
    criteria.push_back(criterion);
  }
  
  virtual string Name() const override {
    ostringstream o;
    o << "AND(";
    bool first = true;
    for (shared_ptr<ConvergenceCriterion> criterion : criteria) {
      if (first) {
        first = false;
      } else {
        o << ", ";
      }
      o << criterion->Name();
    }
    o << ")";
    return o.str();
  }
  
  virtual bool IsConverged(const TrackingSample& sample, int iteration, int scale, float scaling_factor, float damping) const override {
    for (shared_ptr<ConvergenceCriterion> criterion : criteria) {
      if (!criterion->IsConverged(sample, iteration, scale, scaling_factor, damping)) {
        return false;
      }
    }
    return true;
  }
  
  vector<shared_ptr<ConvergenceCriterion>> criteria;
};

struct ConvergenceCriterion_Or : public ConvergenceCriterion {
  void AddCriterion(shared_ptr<ConvergenceCriterion> criterion) {
    criteria.push_back(criterion);
  }
  
  virtual string Name() const override {
    ostringstream o;
    o << "OR(";
    bool first = true;
    for (shared_ptr<ConvergenceCriterion> criterion : criteria) {
      if (first) {
        first = false;
      } else {
        o << ", ";
      }
      o << criterion->Name();
    }
    o << ")";
    return o.str();
  }
  
  virtual bool IsConverged(const TrackingSample& sample, int iteration, int scale, float scaling_factor, float damping) const override {
    for (shared_ptr<ConvergenceCriterion> criterion : criteria) {
      if (criterion->IsConverged(sample, iteration, scale, scaling_factor, damping)) {
        return true;
      }
    }
    return false;
  }
  
  vector<shared_ptr<ConvergenceCriterion>> criteria;
};


void RunConvergenceAnalysis(const std::string& convergence_file_path) {
  std::ifstream infile(convergence_file_path, std::ios::in);
  if (!infile) {
    LOG(FATAL) << "Cannot read convergence file: " << convergence_file_path;
  }
  
  // Indexed by: [scale][sample_index].
  vector<vector<TrackingSample>> samples;
  samples.resize(1);
  
  TrackingSample* current_sample = nullptr;
  
  LOG(ERROR) << "Currently using hardcoded damping values that must match the actually used ones. TODO: log damping.";
  constexpr int kNumScales = 5;
  constexpr float kDamping[kNumScales] = {1.f, 1.f, 1.f, 0.5f, 0.25f};  // TODO: log this instead of having a copy here
  u32 scale = 0;
  
  u32 max_iterations = 0;
  
  while (!infile.eof() && !infile.bad()) {
    std::string line;
    std::getline(infile, line);
    if (line.size() == 0) {
      continue;
    }
    
    std::istringstream line_stream(line);
    std::string word;
    line_stream >> word;
    
    if (word == "scale") {
      // Start new multi-res tracking sample with the given scale
      line_stream >> word;
      scale = atoi(word.c_str());
      
      if (samples.size() < scale) {
        samples.resize(scale + 1);
      }
      
      samples[scale].emplace_back();
      current_sample = &samples[scale].back();
    } else if (word == "EstimateFramePose()") {
      // Start new original-res tracking sample
      scale = 0;
      samples[scale].emplace_back();
      current_sample = &samples[scale].back();
    } else if (word == "x") {
      Eigen::Matrix<float, 6, 1> x;
      line_stream >> x(0) >> x(1) >> x(2) >> x(3) >> x(4) >> x(5);
      current_sample->x.push_back(x);
      current_sample->final_global_T_frame = current_sample->final_global_T_frame * SE3f::exp(-kDamping[scale] * x);
      
      max_iterations = std::max<u32>(max_iterations, current_sample->x.size());
    } else if (word == "residual_sum") {
      float residual_sum;
      line_stream >> residual_sum;
      current_sample->residual_sums.push_back(residual_sum);
    }
  }
  
  // Find out when the samples converged.
  
  constexpr float kActualConvergenceThreshold = 1e-6f;
  
  // Indexed by: [iteration]
  vector<u32> converged_after_iteration_histogram(max_iterations, 0);
  
  u32 num_dropped_samples = 0;
  // Indexed by: [scale].
  vector<u32> valid_sample_count_for_scale(samples.size(), 0);
  
  for (scale = 0; scale < samples.size(); ++ scale) {
    for (u32 sample_index = 0; sample_index < samples[scale].size(); ++ sample_index) {
      TrackingSample& sample = samples[scale][sample_index];
      
      // Determine after which iteration this sample is assumed to be converged
      sample.converged_after_iteration = 0;
      SE3f current_global_T_frame = sample.final_global_T_frame;
      for (int iteration = sample.x.size() - 1; iteration >= 0; -- iteration) {
        // Undo this iteration
        current_global_T_frame = current_global_T_frame * SE3f::exp(-kDamping[scale] * sample.x[iteration]).inverse();
        
        // Was it too far away to be considered converged before this iteration?
        float squared_distance = (current_global_T_frame.inverse() * sample.final_global_T_frame).log().squaredNorm();
        if (squared_distance >= kActualConvergenceThreshold) {
          if (sample.converged_after_iteration > iteration) {
            // Convergence iteration determined and no 2-iteration jittering detected, break.
            break;
          }
          sample.converged_after_iteration = iteration;
        } else {
          // Reset converged iteration. In case of jittering where the pose
          // ends up at the final pose in every 2nd frame, this will continue the
          // search for the convergence iteration.
          sample.converged_after_iteration = 0;
        }
      }
      
      // If the sample only converged in the last 5 iterations, drop it.
      // This is because we do not know when it actually converged, so we do not
      // know the ground truth. It may be due to jittering or not reaching the
      // final pose at all.
      sample.is_valid = sample.converged_after_iteration < static_cast<int>(sample.x.size()) - 5;
      if (sample.is_valid) {
        ++ valid_sample_count_for_scale[scale];
        ++ converged_after_iteration_histogram[sample.converged_after_iteration];
      } else {
        ++ num_dropped_samples;
      }
    }
  }
  
  std::cout << "Dropped " << num_dropped_samples << " samples since their ground truth convergence iteration could not be determined." << std::endl << std::endl;
  
  std::cout << "Converged-after-iteration histogram:" << std::endl;
  for (usize iteration = 0; iteration < converged_after_iteration_histogram.size(); ++ iteration) {
    std::cout << "[" << iteration << "] " << converged_after_iteration_histogram[iteration] << std::endl;
  }
  
  std::cout << std::endl;
  
  // Test how well different convergence criteria would work
  // Indexed by: [scale].
  vector<float> best_cost_for_scale(samples.size(), std::numeric_limits<float>::infinity());
  vector<string> best_criterion_for_scale(samples.size());
  vector<u32> best_converged_too_early_for_scale(samples.size());
  vector<u32> best_converged_okay_for_scale(samples.size());
  vector<u32> best_converged_too_late_for_scale(samples.size());
  vector<u32> best_non_converged_for_scale(samples.size());
  
  constexpr int kOkayIterationsCount = 4;  // Number of *additional* iterations after the one in which the tracking converges which are considered ok
  
  vector<shared_ptr<ConvergenceCriterion>> criteria;
  
  constexpr float kCriteriaToTest[5] = {1e-5f, 1e-6f, 1e-7f, 1e-8f, 1e-9f};
  for (u32 criterion_index_for_translation = 0; criterion_index_for_translation < 5; ++ criterion_index_for_translation) {
    for (u32 criterion_index_for_rotation = 0; criterion_index_for_rotation < 5; ++ criterion_index_for_rotation) {
      criteria.emplace_back(new ConvergenceCriterion_DeltaThreshold(
          kCriteriaToTest[criterion_index_for_translation],
          kCriteriaToTest[criterion_index_for_rotation]));
      
      for (int i = 1; i <= 2; ++ i) {
        shared_ptr<ConvergenceCriterion_And> and_criterion(new ConvergenceCriterion_And());
        and_criterion->AddCriterion(shared_ptr<ConvergenceCriterion>(
            new ConvergenceCriterion_DeltaThreshold(
                kCriteriaToTest[criterion_index_for_translation],
                kCriteriaToTest[criterion_index_for_rotation])));
        and_criterion->AddCriterion(shared_ptr<ConvergenceCriterion>(
            new ConvergenceCriterion_CostNonDecrease(i)));
        criteria.emplace_back(and_criterion);
        
        shared_ptr<ConvergenceCriterion_Or> or_criterion(new ConvergenceCriterion_Or());
        or_criterion->AddCriterion(shared_ptr<ConvergenceCriterion>(
            new ConvergenceCriterion_DeltaThreshold(
                kCriteriaToTest[criterion_index_for_translation],
                kCriteriaToTest[criterion_index_for_rotation])));
        or_criterion->AddCriterion(shared_ptr<ConvergenceCriterion>(
            new ConvergenceCriterion_CostNonDecrease(i)));
        criteria.emplace_back(or_criterion);
      }
    }
  }
  
  criteria.emplace_back(new ConvergenceCriterion_CostNonDecrease(1));
  criteria.emplace_back(new ConvergenceCriterion_CostNonDecrease(2));
  criteria.emplace_back(new ConvergenceCriterion_CostNonDecrease(3));
  
  for (const shared_ptr<ConvergenceCriterion>& criterion : criteria) {
    std::cout << std::endl << "Testing criterion: " << criterion->Name() << std::endl;
    
    for (scale = 0; scale < samples.size(); ++ scale) {
      std::cout << "- Scale: " << scale << std::endl;
      
      int converged_too_early = 0;  // should be avoided, bad for quality
      int converged_okay = 0;       // defined to be converged within k iterations after the assumed convergence
      int converged_too_late = 0;   // bad for performance but not for quality
      int non_converged = 0;        // number of samples where the criterion did not signal convergence at all
      
      float scaling_factor = pow(2, scale);
      
      for (u32 sample_index = 0; sample_index < samples[scale].size(); ++ sample_index) {
        TrackingSample& sample = samples[scale][sample_index];
        if (!sample.is_valid) {
          continue;
        }
        
        SE3f current_global_T_frame;
        bool converged = false;
        for (int iteration = 0; iteration < static_cast<int>(sample.x.size()); ++ iteration) {
          // Simulate the iteration
          current_global_T_frame = current_global_T_frame * SE3f::exp(-kDamping[scale] * sample.x[iteration]);
          
          // Simulate the convergence check
          if (criterion->IsConverged(sample, iteration, scale, scaling_factor, kDamping[scale])) {
            // According to the criterion, the process is converged here.
            // Check whether this is correct or not.
            if (iteration < sample.converged_after_iteration) {
              ++ converged_too_early;
            } else if (iteration > sample.converged_after_iteration + kOkayIterationsCount) {
              ++ converged_too_late;
            } else {
              ++ converged_okay;
            }
            
            converged = true;
            break;
          }
        }
        
        if (!converged) {
          ++ non_converged;
        }
      }
      
      std::cout << "  converged_too_early: " << (converged_too_early / (0.01f * valid_sample_count_for_scale[scale])) << "%" << std::endl;
      std::cout << "  converged_okay: " << (converged_okay / (0.01f * valid_sample_count_for_scale[scale])) << "%" << std::endl;
      std::cout << "  converged_too_late: " << (converged_too_late / (0.01f * valid_sample_count_for_scale[scale])) << "%" << std::endl;
      std::cout << "  non_converged: " << (non_converged / (0.01f * valid_sample_count_for_scale[scale])) << "%" << std::endl;
      
      float cost =
          100 * converged_too_early +
            0 * converged_okay +
           50 * converged_too_late +
          100 * non_converged;
      if (cost < best_cost_for_scale[scale]) {
        best_cost_for_scale[scale] = cost;
        best_criterion_for_scale[scale] = criterion->Name();
        best_converged_too_early_for_scale[scale] = converged_too_early;
        best_converged_okay_for_scale[scale] = converged_okay;
        best_converged_too_late_for_scale[scale] = converged_too_late;
        best_non_converged_for_scale[scale] = non_converged;
      }
    }
  }
  
  // Output the best criterion for each scale
  std::cout << std::endl << "List of best criterion for each scale:" << std::endl;
  for (scale = 0; scale < samples.size(); ++ scale) {
    std::cout << "- Scale: " << scale << std::endl;
    
    std::cout << "  Criterion: " << best_criterion_for_scale[scale] << std::endl;
    
    std::cout << "  converged_too_early: " << (best_converged_too_early_for_scale[scale] / (0.01f * valid_sample_count_for_scale[scale])) << "%" << std::endl;
    std::cout << "  converged_okay: " << (best_converged_okay_for_scale[scale] / (0.01f * valid_sample_count_for_scale[scale])) << "%" << std::endl;
    std::cout << "  converged_too_late: " << (best_converged_too_late_for_scale[scale] / (0.01f * valid_sample_count_for_scale[scale])) << "%" << std::endl;
    std::cout << "  non_converged: " << (best_non_converged_for_scale[scale] / (0.01f * valid_sample_count_for_scale[scale])) << "%" << std::endl;
  }
}

}
