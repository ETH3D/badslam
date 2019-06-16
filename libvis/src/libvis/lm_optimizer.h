// Copyright 2017-2019 ETH Zürich, Thomas Schöps
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

#include <fstream>

// Only required for optional matrix multiplication on the GPU:
#include <cublasXt.h>

#include "libvis/eigen.h"
#include "libvis/libvis.h"
#include "libvis/logging.h"
#include "libvis/loss_functions.h"
#include "libvis/lm_optimizer_impl.h"
#include "libvis/lm_optimizer_update_accumulator.h"
#include "libvis/timing.h"

namespace vis {

// Generic class for non-linear continuous optimization with the
// Levenberg-Marquardt method.
//
// TODO: support vector residuals on which a cost function is applied to as a whole.
// The general form of the optimization problem is to minimize a cost function
// consisting of a sum of residuals r_i:
//   C(x) = \sum_{i} p(r_i(x)) .
// Here, p(r) is either the square function p(r) = r^2 or a robust loss function
// such as Huber's function or Tukey's biweight function. The
// Levenberg-Marquardt algorithm implemented by this class works best if the
// individual residuals r_i are zero-mean and small when close to the optimum.
// In general, the residuals must be continuous and differentiable as this is a
// gradient-based optimization method. However, if individual residuals do not
// fulfill this over the whole domain, it is usually negligible since the other
// residuals still provide sufficient information. The gradient must be non-zero
// for the optimization to work, as otherwise the direction of the update cannot
// be determined. The optimization will find a saddle point or a local minimum,
// which is only guaranteed to be the global minimum if the cost function is
// convex.
// 
// The derivation of the method is as follows.
// TODO: Derivation of the update.
// TODO: Derivation of iteratively re-weighted least squares.
// 
// The user of the class defines the cost function. Optionally, analytical
// Jacobian computation can also be provided by the user for increased
// performance.
// 
// The template parameters must be given as follows:
// * Scalar should be float or double, determining the numerical precision used
//   for computing the cost and Jacobians.
// * State holds the optimization problem state and is typically an
//   Eigen::Vector. However, for optimization using Lie algebras of SE3 or SO3,
//   for example, one can provide a custom type which stores the rotation
//   part(s) of the state as quaternion while applying updates using the
//   exponential map on the corresponding Lie algebra element. It is also
//   possible to define the state to be a wrapper on some existing data
//   structure whose contents are optimized, for example, the pixels of an
//   image. This avoids creating another representation for it.
// * CostFunction computes the cost for a given state and can
//   optionally provide analytical Jacobians for increased optimization speed.
//   Analytical Jacobians should always be used if performance is of concern.
// 
// The class passed for State must provide the following:
// 
// class State {
//  public:
//   // Either a copy constructor and operator= must be provided that actually
//   // copy the data (i.e., if the state contains pointers to external data,
//   // the data pointed to must be copied!), as follows:
//   State(const State& other);
//   State& operator=(const State& other);
//   
//   // Or the state must be reversible (i.e., doing "state -= x" is always
//   // reverted to the original value by following it up with "state -= -x" for
//   // any state and any x), and declare this by defining the following member
//   // function:
//   static constexpr bool is_reversible() { return true; }
//   // NOTE: The optimizer also assumes classes to be reversible which define a
//   // function "rows()". This is a HACK to support Eigen::Matrix vectors as
//   // states. TODO: Couldn't we do better and detect Eigen::Matrix directly
//   // using partial template specializations?
//   
//   // Returns the number of variables in the optimization problem. For
//   // example, for fitting a 2D line represented by the equation m * x + t,
//   // this should return 2 (as the parameters are m and t). Note that as an
//   // exception, this function can also be named rows(), which makes it
//   // possible to use an Eigen::Matrix vector for the state. TODO: See above,
//   // couldn't we detect Eigen::Matrix more directly?
//   int degrees_of_freedom() const;
//   
//   // Subtracts a delta vector from the state. The delta is computed by
//   // LMOptimizer and its row count equals the return value of
//   // degrees_of_freedom(). In the simplest case, the State class will subtract
//   // the corresponding delta vector component from each state variable,
//   // however for cases such as optimization over Lie groups, the
//   // implementation can differ.
//   template <typename Derived>
//   void operator-=(const MatrixBase<Derived>& delta);
// };
// 
// The class passed for CostFunction must provide the following:
//
// class CostFunction {
//  public:
//   // Computes the cost for a given state by providing the values of all
//   // residuals, and optionally the Jacobians of the residuals wrt. the
//   // variables if supported.
//   template<bool compute_jacobians, class Accumulator>
//   inline void Compute(
//       const State& state,
//       Accumulator* accumulator) const {
//     for (residual : residuals) {  // loop over all residuals
//       // To add a residual (r_i in the generic cost term above) to the cost,
//       // call the following, depending on whether compute_jacobians is true.
//       // This expects the non-squared residual. Jacobian computations should
//       // be omitted if compute_jacobians is false.
//       // TODO: Support optionally using Gauss-Newton only. Will this remove the need to have the !compute_jacobians case?
//       Scalar residual_value = ...;  // compute residual
//       if (compute_jacobians) {
//         int index = ...;  // get / compute Jacobian indices
//         Matrix<Scalar, rows, 1> jacobian = ...;  // compute jacobian
//         accumulator->AddResidualWithJacobian(index, residual_value, jacobian);
//       } else {
//         accumulator->AddResidual(residual_value);
//       }
//     }
//   }
// };
// 
// Step-by-step instructions to set up an optimization process with this class:
// 
// 1. Think about the optimization state (i.e., the optimized variables) and the
//    cost function.
//    
//    For example, for simple affine function fitting of a function
//    y = m * x + t, the cost would be a sum over pow(m * x + t - y, 2) for all
//    data points (x, y), and the state can be defined as (m, t)^T (or
//    equivalently, (t, m)^T).
// 
// 2. Choose / implement a class to hold the state, and a sequential indexing of
//    all variables within the state, if needed.
//    
//    In the line fitting example, an Eigen::Vector2f would be well-suited to
//    hold the state and the variables can be enumerated as (1 : m, 2: t) (or
//    the other way round).
//    
//    If the state contains many variables from an existing data structure, it
//    might be beneficial to define the state class as a wrapper on this
//    existing data structure (following the State scheme given above). This
//    avoids the need to copy the variables into the state before optimization
//    and back into their original place after optimization. Furthermore, this
//    allows the code to consistently access these variables in their original
//    place and form, even the cost function (residual and Jacobian) computation
//    code.
//    In this case, for the manual implementation of operator -=, the sequential
//    indexing is required to map components from the parameter of this function
//    (the delta vector) to the variables in the state.
//    
//    The variable indexing might be able to account for some structure in the
//    variables. One might want to list some variables first (having them first
//    is an assumption that the implementation makes without loss of generality)
//    that form a block-diagonal sub-matrix in the Hessian. This later enables
//    to use the Schur complement to solve for the update much faster.
//    
//    Make sure that there is no gauge freedom. If there is it has to be fixed,
//    for example by removing variables from the state. However, keep in mind
//    that the remaining variables have to be indexed sequentially.
// 
// 3. Implement the residual and Jacobian calculation according to the scheme
//    given above (CostFunction). The Compute() function must loop over all
//    residuals and add their values (and their Jacobians if compute_jacobians
//    is true) to the accumulator object using the functions it provides
//    (AddResidual, AddResidualWithJacobian, ...). The Jacobian indexing refers
//    to the variable ordering defined above in step 2.
// 
// 4. To set up and solve a problem, proceed as follows:
//      
//      CostFunction cost_function;
//      // Set up the cost function (e.g., collect residuals) ...
//      
//      State state;
//      // Initialize the state ...
//      
//      LMOptimizer<Scalar> optimizer;  // choose float or double for Scalar
//      optimizer.Optimize(
//          &state,
//          max_iteration_count,
//          cost_function,
//          print_progress);
//      
//      // Use the optimized state ...
// 
// 
// TODO:
// - Implement using numerical derivatives for optimization. How to efficiently
//   get the derivatives of individual residuals? Can we optionally require
//   additional methods for this? Is it possible to initiate the computations
//   in a variant of the AddJacobian() call?
// - Implement fast special case for a diagonal sub-matrix in H.
// - Implement an option to use sparse storage for H (if Schur complement not used).
template<typename Scalar>
class LMOptimizer {
 public:
  /// Helper struct which creates a copy of the object passed in the constructor
  /// if DoCopy == true, or simply references the existing object if DoCopy ==
  /// false. The copy or reference can be gotten with GetObject().
  template <bool DoCopy, typename T>
  struct OptionalCopy {
    // Implements the case DoCopy == false
    inline OptionalCopy(T* object)
        : object(object) {}
    
    inline T* GetObject() {
      return object;
    }
    
    T* object;
  };
  
  template <typename T>
  struct OptionalCopy<true, T> {
    // Implements the case DoCopy == true
    inline OptionalCopy(T* object)
        : object(*object) {}
    
    inline T* GetObject() {
      return &object;
    }
    
    T object;
  };
  
  class CostAccumulator {
   public:
    inline void AddInvalidResidual() {
      // TODO
    }
    
    template <typename LossFunctionT = QuadraticLoss>
    inline void AddResidual(
        Scalar residual,
        const LossFunctionT& loss_function = LossFunctionT()) {
      cost_ += loss_function.ComputeCost(residual);
    }
    
    template <typename LossFunctionT = QuadraticLoss,
            typename Derived>
    inline void AddResidualWithJacobian(
        Scalar residual,
        u32 /*index*/,
        const MatrixBase<Derived>& /*jacobian*/,
        const LossFunctionT& loss_function = LossFunctionT()) {
      cost_ += loss_function.ComputeCost(residual);
    }
    
    template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1>
    inline void AddResidualWithJacobian(
        Scalar residual,
        u32 /*index0*/,
        const MatrixBase<Derived0>& /*jacobian0*/,
        u32 /*index1*/,
        const MatrixBase<Derived1>& /*jacobian1*/,
        bool /*enable0*/ = true,
        bool /*enable1*/ = true,
        const LossFunctionT& loss_function = LossFunctionT()) {
      cost_ += loss_function.ComputeCost(residual);
    }
    
    template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1>
    inline void AddResidualWithJacobian(
        Scalar residual,
        const MatrixBase<Derived0>& /*indices*/,
        const MatrixBase<Derived1>& /*jacobian*/,
        const LossFunctionT& loss_function = LossFunctionT()) {
      cost_ += loss_function.ComputeCost(residual);
    }
    
    template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2,
            typename Derived3>
    inline void AddResidualWithJacobian(
        Scalar residual,
        u32 /*index0*/,
        const MatrixBase<Derived0>& /*jacobian0*/,
        u32 /*index1*/,
        const MatrixBase<Derived1>& /*jacobian1*/,
        const MatrixBase<Derived2>& /*indices2*/,
        const MatrixBase<Derived3>& /*jacobian2*/,
        bool /*enable0*/ = true,
        bool /*enable1*/ = true,
        const LossFunctionT& loss_function = LossFunctionT()) {
      cost_ += loss_function.ComputeCost(residual);
    }
    
    inline void Reset() { cost_ = 0; }
    
    inline Scalar cost() const { return cost_; }
    
   private:
    Scalar cost_ = 0;
  };
  
  class ResidualSumAndJacobianAccumulator {
   public:
    ResidualSumAndJacobianAccumulator(int degrees_of_freedom) {
      jacobian_.resize(degrees_of_freedom);
      jacobian_.setZero();
    }
    
    inline void AddInvalidResidual() {
      // TODO
    }
    
    template <typename LossFunctionT = QuadraticLoss>
    inline void AddResidual(
        Scalar residual,
        const LossFunctionT& loss_function = LossFunctionT()) {
      cost_ += loss_function.ComputeCost(residual);
      residual_sum_ += residual;
    }
    
    template <typename LossFunctionT = QuadraticLoss,
            typename Derived>
    inline void AddResidualWithJacobian(
        Scalar residual,
        u32 index,
        const MatrixBase<Derived>& jacobian,
        const LossFunctionT& loss_function = LossFunctionT()) {
      cost_ += loss_function.ComputeCost(residual);
      residual_sum_ += residual;
      
      jacobian_.template segment<Derived::ColsAtCompileTime>(index) += jacobian;
    }
    
    template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1>
    inline void AddResidualWithJacobian(
        Scalar residual,
        u32 index0,
        const MatrixBase<Derived0>& jacobian0,
        u32 index1,
        const MatrixBase<Derived1>& jacobian1,
        bool enable0 = true,
        bool enable1 = true,
        const LossFunctionT& loss_function = LossFunctionT()) {
      cost_ += loss_function.ComputeCost(residual);
      residual_sum_ += residual;
      
      if (enable0) {
        jacobian_.template segment<Derived0::ColsAtCompileTime>(index0) += jacobian0;
      }
      if (enable1) {
        jacobian_.template segment<Derived1::ColsAtCompileTime>(index1) += jacobian1;
      }
    }
    
    template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1>
    inline void AddResidualWithJacobian(
        Scalar residual,
        const MatrixBase<Derived0>& indices,
        const MatrixBase<Derived1>& jacobian,
        const LossFunctionT& loss_function = LossFunctionT()) {
      cost_ += loss_function.ComputeCost(residual);
      residual_sum_ += residual;
      
      for (int i = 0; i < indices.size(); ++ i) {
        jacobian_(indices[i]) += jacobian(i);
      }
    }
    
    template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2,
            typename Derived3>
    inline void AddResidualWithJacobian(
        Scalar residual,
        u32 index0,
        const MatrixBase<Derived0>& jacobian0,
        u32 index1,
        const MatrixBase<Derived1>& jacobian1,
        const MatrixBase<Derived2>& indices2,
        const MatrixBase<Derived3>& jacobian2,
        bool enable0 = true,
        bool enable1 = true,
        const LossFunctionT& loss_function = LossFunctionT()) {
      AddResidualWithJacobian(
          residual,
          index0, jacobian0,
          index1, jacobian1,
          enable0, enable1,
          loss_function);
      
      for (int i = 0; i < indices2.size(); ++ i) {
        jacobian_(indices2[i]) += jacobian2(i);
      }
    }
    
    inline Scalar residual_sum() const { return residual_sum_; }
    
    inline const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& jacobian() const { return jacobian_; }
    
    inline Scalar cost() const { return cost_; }
    
   private:
    Scalar cost_ = 0;
    Scalar residual_sum_ = 0;
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> jacobian_;
  };
  
  
  /// Tells the optimizer about the block-diagonal structure of the first
  /// (block_size * num_blocks) variables within the problem. This enables it to
  /// use the Schur complement when solving for the update to speed this step up
  /// greatly.
  /// TODO: Currently, the implementation assumes that all blocks are of the
  /// same size.
  void UseBlockDiagonalStructureForSchurComplement(int block_size, int num_blocks) {
    m_use_block_diagonal_structure = true;
    m_block_size = block_size;
    m_num_blocks = num_blocks;
  }
  
  /// Sets the rank deficiency of the Hessian (e.g., due to gauge freedom) that
  /// should be accounted for in solving for state updates by setting the
  /// corresponding number of least-constrained variable updates to zero.
  void AccountForRankDeficiency(int rank_deficiency) {
    m_rank_deficiency = rank_deficiency;
  }
  
  /// Removes a variable from the optimization, fixing it to its initialization.
  /// NOTE: The current implementation of this is potentially very slow! It is
  ///       recommended to use this for debug purposes only. Notice that removing
  ///       a variable in this way will always be slower than setting up the
  ///       problem without this variable (treating it as a constant) in the
  ///       first place. This function is provided for convenience only.
  void FixVariable(int index) {
    if (fixed_variables.size() < index + 1) {
      fixed_variables.resize(index + 1, false);
    }
    fixed_variables[index] = true;
  }
  
  /// Opposite of FixVariable().
  void FreeVariable(int index) {
    if (fixed_variables.size() < index + 1) {
      return;  // variables are free by default if there is no fixed_variables entry for them
    }
    fixed_variables[index] = false;
  }
  
  /// Runs the optimization until convergence is assumed. The initial state is
  /// passed in as pointer "state". This state is modified by the function and
  /// set to the final state after it returns. The final cost value is given as
  /// function return value.
  /// TODO: Allow to specify the strategy for initialization and update of
  ///       lambda. Can also easily provide a Gauss-Newton implementation then by
  ///       checking for the special case lambda = 0 and not retrying the update
  ///       then.
  template <class State, class CostFunction>
  Scalar Optimize(
      State* state,
      const CostFunction& cost_function,
      int max_iteration_count,
      int max_lm_attempts = 10,
      Scalar init_lambda = -1,
      Scalar init_lambda_factor = static_cast<Scalar>(0.001),
      bool print_progress = false) {
    constexpr bool is_reversible = IsReversibleGetter<State>::eval();
    constexpr bool kUseGradientDescent = false;  // TODO
    if (kUseGradientDescent) {
      return OptimizeWithGradientDescentImpl<State, CostFunction, is_reversible>(
          state,
          cost_function,
          max_iteration_count,
          max_lm_attempts,
          init_lambda,
          init_lambda_factor,
          print_progress);
    } else {
      return OptimizeImpl<State, CostFunction, is_reversible>(
          state,
          cost_function,
          max_iteration_count,
          max_lm_attempts,
          init_lambda,
          init_lambda_factor,
          print_progress);
    }
  }
  
  /// Verifies the analytical cost Jacobian provided by CostFunction
  /// by comparing it to the numerically calculated value. This is done for
  /// the current state. NOTE: This refers to the Jacobian of the total cost wrt.
  /// the state variables. It does not check each residual's individual Jacobian.
  /// TODO: Allow setting step size and precision threshold for each state
  ///       component.
  template <class State, class CostFunction>
  bool VerifyAnalyticalJacobian(
      State* state,
      Scalar step_size,
      Scalar error_threshold,
      const CostFunction& cost_function,
      int first_dof = 0,
      int last_dof = -1) {
    // Determine the variable count of the optimization problem.
    const int degrees_of_freedom = DegreesOfFreedomGetter<State>::eval(*state);
    CHECK_GT(degrees_of_freedom, 0);
    
    if (last_dof < 0) {
      last_dof = degrees_of_freedom - 1;
    }
    
    // Determine cost at current state.
    ResidualSumAndJacobianAccumulator helper(degrees_of_freedom);
    cost_function.template Compute<true>(*state, &helper);
    const Scalar base_residual_sum = helper.residual_sum();
    const Scalar base_cost = helper.cost();
    
    // NOTE: Using forward differences only for now.
    bool have_error = false;
    for (int variable_index = first_dof; variable_index <= last_dof; ++ variable_index) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> delta;
      delta.resize(degrees_of_freedom, Eigen::NoChange);
      delta.setZero();
      // Using minus step size since the delta will be subtracted.
      delta(variable_index) = -step_size;
      
      State test_state(*state);
      test_state -= delta;
      ResidualSumAndJacobianAccumulator test_helper(degrees_of_freedom);
      cost_function.template Compute<false>(test_state, &test_helper);
      const Scalar test_residual_sum = test_helper.residual_sum();
      const Scalar test_cost = test_helper.cost();
      
      Scalar analytical_jacobian_component = helper.jacobian()(variable_index);
      Scalar numerical_jacobian_component = (test_residual_sum - base_residual_sum) / step_size;
      
      Scalar error = fabs(analytical_jacobian_component - numerical_jacobian_component);
      if (error <= error_threshold) {
        LOG(1) << "VerifyAnalyticalJacobian(): Component " << variable_index << " ok (diff: " << fabs(analytical_jacobian_component - numerical_jacobian_component) << ")";
      } else {
        LOG(ERROR) << "VerifyAnalyticalJacobian(): Component " << variable_index
                   << " differs (diff: " << fabs(analytical_jacobian_component - numerical_jacobian_component)
                   << "): Analytical: " << analytical_jacobian_component
                   << ", numerical: " << numerical_jacobian_component
                   << " (base_cost: " << base_cost << ", test_cost: "
                   << test_cost << ")";
        have_error = true;
      }
    }
    return !have_error;
  }
  
  /// Verifies the cost computed by the cost function.
  template <class State, class CostFunction>
  Scalar VerifyCost(
      State* state,
      const CostFunction& cost_function) {
    CostAccumulator helper1;
    cost_function.template Compute<false>(*state, &helper1);
    
    CostAccumulator helper2;
    cost_function.template Compute<true>(*state, &helper2);
    
    if (helper1.cost() != helper2.cost()) {
      LOG(ERROR) << "Cost differs when computed with or without Jacobians:";
      LOG(ERROR) << "Without Jacobians: " << helper1.cost();
      LOG(ERROR) << "With Jacobians: " << helper2.cost();
    }
    
    return helper1.cost();
  }
  
  /// Tests whether the given state is likely an optimum by comparing its cost
  /// to that of some samples around it. Returns true if no lower cost state is
  /// found, false otherwise.
  /// TODO: For reduce_cost_if_possible, could also try to escape from saddle
  ///       points right away if one is detected (cost goes down for both
  ///       opposite directions), or make a step as soon as one is detected that
  ///       can reduce the cost to 95% or less.
  template <class State, class CostFunction>
  bool VerifyOptimum(
      State* state,
      const CostFunction& cost_function,
      Scalar step_size,
      bool reduce_cost_if_possible,
      int first_dof = 0,
      int last_dof = -1,
      Scalar* final_cost = nullptr) {
    constexpr bool is_reversible = IsReversibleGetter<State>::eval();
    CHECK(!is_reversible) << "This is currently only implemented for states with copy constructors";
    
    CostAccumulator helper;
    bool lower_cost_found = false;
    
    cost_function.template Compute<false>(*state, &helper);
    Scalar center_cost = helper.cost();
    LOG(1) << "Center cost in VerifyOptimum(): " << center_cost;
    
    int dof = DegreesOfFreedomGetter<State>::eval(*state);
    if (last_dof < 0) {
      last_dof = dof - 1;
    }
    Matrix<Scalar, Eigen::Dynamic, 1> delta;
    delta.resize(dof);
    delta.setZero();
    
    int best_i = -1;  // initialized only to silence the warning
    int best_d = 0;  // initialized only to silence the warning
    Scalar best_factor = 0;  // initialized only to silence the warning
    Scalar best_cost = numeric_limits<Scalar>::infinity();
    
    for (int i = first_dof; i <= last_dof; ++ i) {
      bool cost_reduced_for_first_direction = false;
      Scalar cost_for_first_direction = -1;  // initialized only to silence the warning
      Scalar factor_for_first_direction = 0;  // initialized only to silence the warning
      bool exit_search = false;
      
      for (int d = -1; d <= 1; d += 2) {
        delta(i) = d * step_size;
        State offset_state(*state);
        offset_state -= delta;
        
        helper.Reset();
        cost_function.template Compute<false>(offset_state, &helper);
        Scalar offset_cost = helper.cost();
        
        bool lower_cost = offset_cost < center_cost;
        lower_cost_found |= lower_cost;
        
        if (lower_cost) {
          LOG(WARNING) << "[" << i << " / " << dof << "] Lower cost found: " << helper.cost() << " < " << center_cost;
          
          if (reduce_cost_if_possible) {
            Scalar factor = 1;
            for (int f = 0; f < 10; ++ f) {
              Scalar new_factor = 2 * factor;
              State new_offset_state(*state);
              new_offset_state -= new_factor * delta;
              helper.Reset();
              cost_function.template Compute<false>(new_offset_state, &helper);
              if (helper.cost() >= offset_cost) {
                break;
              }
              
              offset_cost = helper.cost();
              factor = new_factor;
            }
            
            if (offset_cost < best_cost) {
              best_cost = offset_cost;
              best_i = i;
              best_d = d;
              best_factor = factor;
            }
            
            if (!cost_reduced_for_first_direction) {
              cost_for_first_direction = offset_cost;
              factor_for_first_direction = factor;
            } else {  // if (cost_reduced_for_first_direction)
              // Detected a saddle point (if it is not a maximum): the cost goes
              // down in two opposite directions. Try to escape this saddle point
              // by going in one of these directions.
              if (cost_for_first_direction < offset_cost) {
                best_cost = cost_for_first_direction;
                best_i = i;
                best_d = -1 * d;
                best_factor = factor_for_first_direction;
              } else {
                best_cost = offset_cost;
                best_i = i;
                best_d = d;
                best_factor = factor;
              }
              LOG(WARNING) << "Saddle point (or maximum) detected.";
              exit_search = true;
              break;
            }
            cost_reduced_for_first_direction = true;
          }
        } else {
          LOG(1) << "[" << i << " / " << dof << "] Higher cost: " << helper.cost();
        }
      }
      
      delta(i) = 0;
      if (exit_search) {
        break;
      }
    }
    
    if (reduce_cost_if_possible && !std::isinf(best_cost)) {
      delta(best_i) = best_d * step_size;
      *state -= best_factor * delta;
      LOG(WARNING) << "Cost reduced from " << center_cost << " to " << best_cost;
      *final_cost = best_cost;
      return false;
    }
    
    *final_cost = center_cost;
    return !lower_cost_found;
  }
  
  /// Returns the final value of lambda (of the Levenberg-Marquardt algorithm)
  /// after the last optimization.
  inline Scalar lambda() const { return m_lambda; }
  
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> step_scaling;  // TODO: TEST
  
  
 private:
  template <class State, class CostFunction, bool IsReversible>
  Scalar OptimizeWithGradientDescentImpl(
      State* state,
      const CostFunction& cost_function,
      int max_iteration_count,
      int max_lm_attempts,
      Scalar /*init_lambda*/,
      Scalar /*init_lambda_factor*/,
      bool print_progress) {
    const int degrees_of_freedom = DegreesOfFreedomGetter<State>::eval(*state);
    CHECK_GT(degrees_of_freedom, 0);
    
    Scalar last_cost = numeric_limits<Scalar>::quiet_NaN();
    bool applied_update = false;
    int iteration;
    for (iteration = 0; iteration < max_iteration_count; ++ iteration) {
      ResidualSumAndJacobianAccumulator jacobian_accumulator(degrees_of_freedom);
      cost_function.template Compute<true>(*state, &jacobian_accumulator);
      last_cost = jacobian_accumulator.cost();
      
      const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& jacobian = jacobian_accumulator.jacobian();
      CHECK_EQ(jacobian.cols(), step_scaling.cols());
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> step = step_scaling.cwiseProduct(jacobian.normalized()).transpose();
      
      Scalar step_factor = 1;
      applied_update = false;
      for (int lm_iteration = 0; lm_iteration < max_lm_attempts; ++ lm_iteration) {
        OptionalCopy<!IsReversible, State> optional_state_copy(state);
        State* updated_state = optional_state_copy.GetObject();
        *updated_state -= step_factor * step;
        
        // Test whether taking over the update will decrease the cost.
        UpdateEquationAccumulator<Scalar> test_cost(nullptr, nullptr, nullptr, nullptr, nullptr);
        cost_function.template Compute<false>(*updated_state, &test_cost);
        
        if (test_cost.cost() < jacobian_accumulator.cost()) {
          // Take over the update.
          if (IsReversible) {
            // no action required, keep the updated state
          } else {
            *state = *updated_state;
          }
          last_cost = test_cost.cost();
          applied_update = true;
          LOG(1) << "GDOptimizer: update accepted, new cost: " << last_cost;
          break;
        } else {
          if (IsReversible) {
            // Undo update. This may cause slight numerical inaccuracies.
            // TODO: Would it be better to combine undoing the old update and
            //       applying the new update (if not giving up) into a single step?
            *state -= -m_x;
          }
          
          step_factor *= 0.8;  // TODO: tune? Make configurable.
          LOG(1) << "GDOptimizer:   [" << (iteration + 1) << ", " << (lm_iteration + 1) << " of " << max_lm_attempts
                 << "] update rejected (bad cost: " << test_cost.cost()
                 << "), new step_factor: " << step_factor;
        }
      }
      
      if (!applied_update) {
        if (print_progress) {
          if (last_cost == 0) {
            LOG(INFO) << "GDOptimizer: Reached zero cost, stopping.";
          } else {
            LOG(INFO) << "GDOptimizer: Cannot find an update which decreases the cost, aborting.";
          }
        }
        iteration += 1;  // For correct display only.
        break;
      }
    }
    
    if (print_progress) {
      if (applied_update) {
        LOG(INFO) << "GDOptimizer: Maximum iteration count reached, stopping.";
      }
//       LOG(INFO) << "GDOptimizer: [" << iteration << "] Cost / Jacobian computation time: " << cost_and_jacobian_time_in_seconds << " seconds";
//       LOG(INFO) << "GDOptimizer: [" << iteration << "] Solve time: " << solve_time_in_seconds << " seconds";
      LOG(INFO) << "GDOptimizer: [" << iteration << "] Final cost:   " << last_cost;  // length matches with "Initial cost: "
    }
    
    return last_cost;
  }
  
  template <class State, class CostFunction, bool IsReversible>
  Scalar OptimizeImpl(
      State* state,
      const CostFunction& cost_function,
      int max_iteration_count,
      int max_lm_attempts,
      Scalar init_lambda,
      Scalar init_lambda_factor,
      bool print_progress) {
    CHECK_GE(init_lambda_factor, 0);
    
    float cost_and_jacobian_time_in_seconds = 0;
    float solve_time_in_seconds = 0;
    
    // Determine the variable count of the optimization problem, distributed
    // over the block-diagonal and dense parts of the Hessian.
    const int degrees_of_freedom = DegreesOfFreedomGetter<State>::eval(*state);
    CHECK_GT(degrees_of_freedom, 0);
    
    const int block_diagonal_degrees_of_freedom =
        m_use_block_diagonal_structure ? (m_block_size * m_num_blocks) : 0;
    
    const int dense_degrees_of_freedom = degrees_of_freedom - block_diagonal_degrees_of_freedom;
    CHECK_GE(dense_degrees_of_freedom, 0);
    
    // Set the size of H and b.
    // The structure of the update equation is as follows (both the top-left and
    // bottom-right blocks in H, and thus also the off-diagonal blocks and
    // corresponding blocks in the x and b vectors, could have size zero though):
    // 
    // +----------------+--------------+   +----------------+   +----------------+
    // | m_block_diag_H | m_off_diag_H |   |                |   | m_block_diag_b |
    // +----------------+--------------+ * | m_x            | = +----------------+
    // | m_off_diag_H^T | m_dense_H    |   |                |   | m_dense_b      |
    // +----------------+--------------+   +----------------+   +----------------+
    m_dense_H.resize(dense_degrees_of_freedom, dense_degrees_of_freedom);
    m_off_diag_H.resize(block_diagonal_degrees_of_freedom, dense_degrees_of_freedom);
    m_block_diag_H.resize(m_num_blocks);
    for (int i = 0; i < m_num_blocks; ++ i) {
      m_block_diag_H[i].resize(m_block_size, m_block_size);
    }
    
    m_x.resize(degrees_of_freedom);
    
    m_dense_b.resize(dense_degrees_of_freedom);
    m_block_diag_b.resize(block_diagonal_degrees_of_freedom);
    
    // Do optimization iterations.
    Scalar last_cost = numeric_limits<Scalar>::quiet_NaN();
    bool applied_update = true;
    int iteration;
    for (iteration = 0; iteration < max_iteration_count; ++ iteration) {
      // Compute cost and Jacobians (which get accumulated on H and b).
      // TODO: Support numerical Jacobian.
      // TODO: We can template OptimizeImpl() with a special type of accumulator
      //       to use. This way, we can make a special case if the Schur
      //       complement is not used to use a simpler (faster) accumulator.
      vector<Scalar> residual_cost_vector;
      residual_cost_vector.reserve(10000);  // TODO: make configurable
      vector<bool> residual_validity_vector;
      residual_validity_vector.reserve(10000);  // TODO: make configurable
      UpdateEquationAccumulator<Scalar> update_eq(
          &m_dense_H, &m_off_diag_H, &m_block_diag_H, &m_dense_b, &m_block_diag_b, &residual_cost_vector, &residual_validity_vector);
      Timer update_eq_timer("", /*construct_stopped*/ !print_progress);
      cost_function.template Compute<true>(*state, &update_eq);
      if (print_progress) {
        cost_and_jacobian_time_in_seconds += update_eq_timer.Stop(/*add_to_statistics*/ false);
      }
      last_cost = update_eq.cost();
      
//       // DEBUG: Output H matrix and b vector
//       if (print_progress) {
//         LOG(INFO) << "degrees_of_freedom: " << degrees_of_freedom;
//         LOG(INFO) << "block_diagonal_degrees_of_freedom: " << block_diagonal_degrees_of_freedom;
//         LOG(INFO) << "dense_degrees_of_freedom: " << dense_degrees_of_freedom;
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H;
//         Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b;
//         update_eq.GetHandB(&H, &b);
//         ofstream H_stream("debug_H.txt", std::ios::out);
//         H_stream << H << std::endl;
//         H_stream.close();
//         ofstream b_stream("debug_b.txt", std::ios::out);
//         b_stream << b << std::endl;
//         b_stream.close();
//         exit(1);
//       }
      
      // Determine numerical rank of H (TODO: move this somewhere / make it accessible in a nice way).
//       if (print_progress) {
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H;
//         Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b;
//         update_eq.GetHandB(&H, &b);
//         JacobiSVD<Matrix<Scalar, Dynamic, Dynamic>> svd_H(H, 0);
//         svd_H.singularValues();
//         double threshold = svd_H.singularValues()[0] * numeric_limits<Scalar>::epsilon() * std::max(H.rows(), H.cols());
//         int rank = 0;
//         for (; rank < H.rows(); ++ rank) {
//           if (svd_H.singularValues()[rank] < threshold) {
//             break;
//           }
//         }
//         LOG(WARNING) << "H matrix numerical rank default epsilon: " << threshold;
//         LOG(WARNING) << "H matrix numerical rank with default epsilon: " << rank;
//         LOG(WARNING) << "H matrix size: " << H.rows();
//         LOG(WARNING) << "H matrix free dimensions: " << (H.rows() - rank);
//         LOG(WARNING) << "H matrix 100 last (smallest) singular values:";
//         for (int i = 0; i < 100; ++ i) {
//           LOG(INFO) << "  " << (100 - i) << ": " << svd_H.singularValues()[svd_H.singularValues().size() - 100 + i];
//         }
//          // << svd_H.singularValues().template bottomRows<40>();
//       }
      
      if (print_progress) {
        if (iteration == 0) {
          LOG(INFO) << "LMOptimizer: [0] Initial cost: " << update_eq.cost();
        } else {
          LOG(1) << "LMOptimizer: [" << iteration << "] cost: " << update_eq.cost();
        }
      }
      
      applied_update = false;
      
      if (update_eq.cost() == 0) {
        if (print_progress) {
          LOG(INFO) << "LMOptimizer: Cost is zero, stopping.";
        }
        break;
      }
      
      // Initialize lambda based on the average diagonal element size in H.
      if (iteration == 0) {
        if (init_lambda >= 0) {
          m_lambda = init_lambda;
        } else {
          m_lambda = 0;
          for (usize i = 0; i < m_block_diag_H.size(); ++ i) {
            for (int k = 0; k < m_block_diag_H[i].rows(); ++ k) {
              m_lambda += m_block_diag_H[i](k, k);
            }
          }
          for (int i = 0; i < dense_degrees_of_freedom; ++ i) {
            m_lambda += m_dense_H(i, i);
          }
          m_lambda = static_cast<Scalar>(init_lambda_factor) * m_lambda / degrees_of_freedom;  // TODO: make the strategy for initializing m_lambda configurable?
        }
      }
      
//       LOG(INFO) << "lambda: " << m_lambda;
      
      for (int lm_iteration = 0; lm_iteration < max_lm_attempts; ++ lm_iteration) {
        Timer solve_timer("", /*construct_stopped*/ !print_progress);
        
        // Handle variable fixing (inefficiently!)
        bool have_fixed_variables = false;
        if (!fixed_variables.empty()) {
          int num_fixed_variables = 0;
          for (usize i = 0; i < fixed_variables.size(); ++ i) {
            if (fixed_variables[i]) {
              ++ num_fixed_variables;
            }
          }
          
          if (num_fixed_variables > 0) {
            have_fixed_variables = true;
            
            // Create a new H and b with the fixed variables removed.
            // Note that this might be very inefficient compared to the standard
            // code path if the latter uses the Schur complement, since here, the
            // Schur complement is not implemented.
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> full_H;
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> full_b;
            update_eq.GetHandB(&full_H, &full_b);
            
//             // DEBUG
//             LOG(INFO) << "num_fixed_variables: " << num_fixed_variables;
//             LOG(INFO) << "H/b size: " << full_H.rows() << " --> " << (full_H.rows() - num_fixed_variables);
            
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> thinned_H;
            thinned_H.resize(full_H.rows() - num_fixed_variables,
                             full_H.cols() - num_fixed_variables);
            
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> thinned_b;
            thinned_b.resize(full_b.size() - num_fixed_variables);
            
            int output_row = 0;
            for (int row = 0; row < full_H.rows(); ++ row) {
              if (row < static_cast<int>(fixed_variables.size()) && fixed_variables[row]) {
                continue;
              }
              
              int output_col = 0;
              for (int col = 0; col < full_H.cols(); ++ col) {
                if (col < static_cast<int>(fixed_variables.size()) && fixed_variables[col]) {
                  continue;
                }
                
                thinned_H(output_row, output_col) = full_H(row, col);
                
                ++ output_col;
              }
              CHECK_EQ(output_col, thinned_H.cols());
              
              thinned_b(output_row) = full_b(row);
              
              ++ output_row;
            }
            CHECK_EQ(output_row, thinned_H.rows());
            
//             // DEBUG: Output thinned H matrix and b vector
//             if (print_progress) {
//               ofstream H_stream("debug_thinned_H.txt", std::ios::out);
//               H_stream << thinned_H << std::endl;
//               H_stream.close();
//               ofstream b_stream("debug_thinned_b.txt", std::ios::out);
//               b_stream << thinned_b << std::endl;
//               b_stream.close();
//               exit(1);
//             }
            
            thinned_H.diagonal().array() += m_lambda;
            
            typedef LDLT<Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, Lower> LDLT_T;
            Matrix<Scalar, Eigen::Dynamic, 1> thinned_x = LDLT_T(thinned_H.template selfadjointView<Eigen::Upper>()).solve(thinned_b);
            
//             // DEBUG: Output thinned x vector
//             if (print_progress) {
//               ofstream x_stream("debug_thinned_x.txt", std::ios::out);
//               x_stream << thinned_x << std::endl;
//               x_stream.close();
//               exit(1);
//             }
            
            output_row = 0;
            for (int row = 0; row < full_H.rows(); ++ row) {
              if (row < static_cast<int>(fixed_variables.size()) && fixed_variables[row]) {
                m_x(row) = 0;
              } else {
                m_x(row) = thinned_x(output_row);
                ++ output_row;
              }
            }
          }
        }
        
        if (!have_fixed_variables) {
          // Add to the diagonal of H according to the Levenberg-Marquardt method.
          // TODO: Can we avoid copying all blocks of H that overlap with the diagonal?
          //       We could iteratively adjust lambda instead. It would introduce some tiny
          //       numerical inaccuracies though.
          // NOTE: We do not consider the variable fixing for initializing lambda,
          //       at the moment.
          vector<Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> block_diag_H_plus_I = m_block_diag_H;
          for (usize i = 0; i < m_block_diag_H.size(); ++ i) {
            m_block_diag_H[i].diagonal().array() += m_lambda;
          }
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense_H_plus_I = m_dense_H;
          dense_H_plus_I.diagonal().array() += m_lambda;
          
          if (block_diagonal_degrees_of_freedom > 0) {
            SolveWithSchurComplement(
                block_diagonal_degrees_of_freedom,
                dense_degrees_of_freedom,
                block_diag_H_plus_I,
                dense_H_plus_I);
          } else {
            // Standard solution: solve the whole system H * x = b at once, without using the Schur complement.
            typedef LDLT<Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, Lower> LDLT_T;
            m_x = LDLT_T(dense_H_plus_I.template selfadjointView<Eigen::Upper>()).solve(m_dense_b);
          }
        }
        
        if (print_progress) {
          solve_time_in_seconds += solve_timer.Stop(/*add_to_statistics*/ false);
        }
        
//       // DEBUG: Output x vector
//       if (print_progress) {
//         ofstream x_stream("debug_x.txt", std::ios::out);
//         x_stream << m_x << std::endl;
//         x_stream.close();
//         exit(1);
//       }
        
        // Apply the update to create a temporary state.
        // Note the inversion of the delta here.
        OptionalCopy<!IsReversible, State> optional_state_copy(state);
        State* updated_state = optional_state_copy.GetObject();
        *updated_state -= m_x;
        
        // Test whether taking over the update will decrease the cost.
        vector<Scalar> test_residual_cost_vector;
        test_residual_cost_vector.reserve(10000);  // TODO: make configurable
        vector<bool> test_residual_validity_vector;
        test_residual_validity_vector.reserve(10000);  // TODO: make configurable
        UpdateEquationAccumulator<Scalar> test_cost(nullptr, nullptr, nullptr, nullptr, nullptr, &test_residual_cost_vector, &test_residual_validity_vector);
        Timer cost_timer("", /*construct_stopped*/ !print_progress);
        cost_function.template Compute<false>(*updated_state, &test_cost);
        if (print_progress) {
          cost_and_jacobian_time_in_seconds += cost_timer.Stop(/*add_to_statistics*/ false);
        }
        
        if (/*test_cost.cost() < update_eq.cost()*/
            CostIsSmallerThan(
                test_residual_cost_vector,
                test_residual_validity_vector,
                residual_cost_vector,
                residual_validity_vector)) {
          // Take over the update.
          if (print_progress && lm_iteration > 0) {
            LOG(1) << "LMOptimizer:   [" << (iteration + 1) << "] update accepted";
          }
          if (IsReversible) {
            // no action required, keep the updated state
          } else {
            *state = *updated_state;
          }
          m_lambda = 0.5f * m_lambda;
          applied_update = true;
          last_cost = test_cost.cost();
          break;
        } else {
          if (IsReversible) {
            // Undo update. This may cause slight numerical inaccuracies.
            // TODO: Would it be better to combine undoing the old update and
            //       applying the new update (if not giving up) into a single step?
            *state -= -m_x;
          } else {
            // no action required, drop the updated state copy
          }
          
          m_lambda = 2.f * m_lambda;
          if (print_progress) {
            LOG(1) << "LMOptimizer:   [" << (iteration + 1) << ", " << (lm_iteration + 1) << " of " << max_lm_attempts
                   << "] update rejected (bad cost: " << test_cost.cost()
                   << "), new lambda: " << m_lambda;
          }
        }
      }
      
      if (!applied_update || last_cost == 0) {
        if (print_progress) {
          if (last_cost == 0) {
            LOG(INFO) << "LMOptimizer: Reached zero cost, stopping.";
          } else {
            LOG(INFO) << "LMOptimizer: Cannot find an update which decreases the cost, aborting.";
          }
        }
        iteration += 1;  // For correct display only.
        break;
      }
    }
    
    if (print_progress) {
      if (applied_update) {
        LOG(INFO) << "LMOptimizer: Maximum iteration count reached, stopping.";
      }
      LOG(INFO) << "LMOptimizer: [" << iteration << "] Cost / Jacobian computation time: " << cost_and_jacobian_time_in_seconds << " seconds";
      LOG(INFO) << "LMOptimizer: [" << iteration << "] Solve time: " << solve_time_in_seconds << " seconds";
      LOG(INFO) << "LMOptimizer: [" << iteration << "] Final cost:   " << last_cost;  // length matches with "Initial cost: "
    }
    
    return last_cost;
  }
  
  bool CostIsSmallerThan(
      vector<Scalar> left_residual_cost_vector,
      vector<bool> left_residual_validity_vector,
      vector<Scalar> right_residual_cost_vector,
      vector<bool> right_residual_validity_vector) {
    const usize size = left_residual_cost_vector.size();
    CHECK_EQ(size, left_residual_validity_vector.size());
    CHECK_EQ(size, right_residual_cost_vector.size());
    CHECK_EQ(size, right_residual_validity_vector.size());
    
    Scalar left_sum = 0;
    Scalar right_sum = 0;
    for (usize i = 0; i < size; ++ i) {
      if (left_residual_validity_vector[i] && right_residual_validity_vector[i]) {
        left_sum += left_residual_cost_vector[i];
        right_sum += right_residual_cost_vector[i];
      }
    }
    
    return left_sum < right_sum;
  }
  
  void SolveWithSchurComplement(
      int block_diagonal_degrees_of_freedom,
      int dense_degrees_of_freedom,
      vector<Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& block_diag_H_plus_I,
      const Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& dense_H_plus_I) {
    // Schur complement solution step 1:
    // Invert the block-diagonal matrix.
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block_I;
    block_I.resize(m_block_size, m_block_size);
    block_I.setZero();
    block_I.diagonal().array().setOnes();
    for (usize i = 0; i < block_diag_H_plus_I.size(); ++ i) {
      auto& matrix = block_diag_H_plus_I[i];
      matrix = matrix.template selfadjointView<Eigen::Upper>().ldlt().solve(block_I);  // (hopefully) fast SPD matrix inversion
      // TODO: Profile this compared to inversion as below:
      // matrix.template triangularView<Eigen::Lower>() = matrix.template triangularView<Eigen::Upper>().transpose();
      // matrix = matrix.inverse();
    }
    
    // Using the Schur complement, compute the small dense matrix to solve
    // Compute block_diagonal^(-1) * off-diagonal
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> D_inv_B;
    D_inv_B.resize(block_diagonal_degrees_of_freedom, dense_H_plus_I.cols());
    for (usize row = 0; row < block_diagonal_degrees_of_freedom; ++ row) {
      for (int col = 0; col < dense_H_plus_I.cols(); ++ col) {
        // We only need to take the non-zero diagonal block into account
        int block_index = row / m_block_size;
        int base_index = block_index * m_block_size;
        
        auto& H_block = block_diag_H_plus_I[block_index];
        
        Scalar result = 0;
        for (int k = 0; k < H_block.cols(); ++ k) {
          result += H_block(row - base_index, k) *
                    m_off_diag_H(base_index + k, col);
        }
        D_inv_B(row, col) = result;
      }
    }
    
    // Compute block_diagonal^(-1) * b_block_diagonal
    Matrix<Scalar, Eigen::Dynamic, 1> D_inv_b1;
    D_inv_b1.resize(block_diagonal_degrees_of_freedom);
    for (int row = 0; row < D_inv_b1.rows(); ++ row) {
      // We only need to take the non-zero diagonal block into account
      int block_index = row / m_block_size;
      int base_index = block_index * m_block_size;
      
      auto& H_block = block_diag_H_plus_I[block_index];
      
      Scalar result = 0;
      for (int k = 0; k < H_block.cols(); ++ k) {
        result += H_block(row - base_index, k) *
                  m_block_diag_b(base_index + k);
      }
      D_inv_b1(row) = result;
    }
    
    // Compute off-diagonal^T * block_diagonal^(-1) * off-diagonal
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> B_T_D_inv_B;
    B_T_D_inv_B.resize(dense_H_plus_I.rows(), dense_H_plus_I.cols());
    
    constexpr bool kRunOnGPU = true;
    if (kRunOnGPU) {
      cublasXtHandle_t cublasxthandle;
      CHECK_EQ(cublasXtCreate(&cublasxthandle), CUBLAS_STATUS_SUCCESS);
      
      int device_ids[1] = {0};  // Simply use the first device
      CHECK_EQ(cublasXtDeviceSelect(cublasxthandle, 1, device_ids), CUBLAS_STATUS_SUCCESS);
      
      constexpr bool kUseFloatOnGPU = false;
      if (kUseFloatOnGPU) {
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> left = m_off_diag_H.transpose().template cast<float>();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> right = D_inv_B.template cast<float>();
        
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> result;
        result.resize(left.rows(), right.cols());
        
        float alpha = 1;
        float beta = 0;
        CHECK_EQ(cublasXtSgemm(
            cublasxthandle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            left.rows(), right.cols(), left.cols(),
            &alpha,
            left.data(), left.rows(),
            right.data(), right.rows(),
            &beta,
            result.data(), result.rows()), CUBLAS_STATUS_SUCCESS);
        
        CHECK_EQ(cublasXtDestroy(cublasxthandle), CUBLAS_STATUS_SUCCESS);
        
        CHECK_EQ(B_T_D_inv_B.rows(), result.rows());
        CHECK_EQ(B_T_D_inv_B.cols(), result.cols());
        B_T_D_inv_B = result.cast<Scalar>();
      } else {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> left = m_off_diag_H.transpose().template cast<double>();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> right = D_inv_B.template cast<double>();
        
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> result;
        result.resize(left.rows(), right.cols());
        
        double alpha = 1;
        double beta = 0;
        CHECK_EQ(cublasXtDgemm(
            cublasxthandle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            left.rows(), right.cols(), left.cols(),
            &alpha,
            left.data(), left.rows(),
            right.data(), right.rows(),
            &beta,
            result.data(), result.rows()), CUBLAS_STATUS_SUCCESS);
        
        CHECK_EQ(cublasXtDestroy(cublasxthandle), CUBLAS_STATUS_SUCCESS);
        
        CHECK_EQ(B_T_D_inv_B.rows(), result.rows());
        CHECK_EQ(B_T_D_inv_B.cols(), result.cols());
        B_T_D_inv_B = result.cast<Scalar>();
      }
    } else {
      B_T_D_inv_B.template triangularView<Eigen::Upper>() = m_off_diag_H.transpose() * D_inv_B;
    }
    
    // Compute off-diagonal^T * block_diagonal^(-1) * b_block_diagonal
    Matrix<Scalar, Eigen::Dynamic, 1> B_T_D_inv_b;
    B_T_D_inv_b.resize(dense_H_plus_I.rows());
    B_T_D_inv_b = m_off_diag_H.transpose() * D_inv_b1;
    
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> schur_M;
    schur_M.resize(dense_H_plus_I.rows(), dense_H_plus_I.cols());
    schur_M.template triangularView<Eigen::Upper>() = dense_H_plus_I - B_T_D_inv_B;
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> schur_b = m_dense_b - B_T_D_inv_b;
    
    // TODO: Performance advice (from the LLT decomposition's documentation):
    // For best performance, it is recommended to use a column-major storage format with the Lower triangular part (the default), or, equivalently, a row-major storage format with the Upper triangular part. Otherwise, you might get a 20% slowdown for the full factorization step, and rank-updates can be up to 3 times slower.
    
    if (m_rank_deficiency > 0) {
      // TODO: We are currently not using the value of m_rank_deficiency apart
      //       from testing (m_rank_deficiency > 0). Could change to a bool if
      //       it stays like this.
      // Use the pseudoinverse as a means of Gauge fixing (it will take the shortest update).
      // Attention, in "BA - a modern synthesis" (Sec. 9.3), also a weight
      // matrix is used that might be useful here.
      schur_M.template triangularView<Eigen::Lower>() = schur_M.template triangularView<Eigen::Upper>().transpose();
      // Only available in more recent Eigen versions:
      m_x.bottomRows(dense_degrees_of_freedom) =
          schur_M.completeOrthogonalDecomposition().solve(schur_b);
//       m_x.bottomRows(dense_degrees_of_freedom) =
//           schur_M.colPivHouseholderQr().solve(schur_b);  // seems not to give good results
//       m_x.bottomRows(dense_degrees_of_freedom) =
//           schur_M.bdcSvd(ComputeThinU|ComputeThinV).solve(schur_b);  // not available in old Eigen
      // TODO: Using the pseudoinverse is extremely effective but also very slow.
      //       Try to find a faster method of gauge fixing.
      // TODO: Try schur_M.bdcSvd(ComputeThinU|ComputeThinV).solve(schur_b);
    } else {
      m_x.bottomRows(dense_degrees_of_freedom) = schur_M.template selfadjointView<Eigen::Upper>().ldlt().solve(schur_b);
    }
    
//     typedef LDLT<Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, Lower> LDLT_T;
//     auto dst = m_x.bottomRows(dense_degrees_of_freedom);
//     SolveWithSubsetSelection(
//         LDLT_T(schur_M.template selfadjointView<Eigen::Upper>()),
//         schur_b,
//         dst);
    
    m_x.topRows(block_diagonal_degrees_of_freedom) =
        D_inv_b1 - D_inv_B * m_x.bottomRows(dense_degrees_of_freedom);
  }
  
  // TODO: Not sure whether this makes sense. Does fixing arbitrary variables
  // really work? Couldn't this fix some variables to a bad state, preventing
  // convergence to the optimum?
  // 
//   /// Solves the equation ldlt * dst = rhs for rank-deficient H. Chooses the
//   /// least-constrained elements and fixes their update (i.e. the corresponding
//   /// element in x) to zero. The parameter rank_deficiency controls the number
//   /// of elements to be fixed. This is implemented according to the short
//   /// textual description in Sec. 9.5 of "Bundle Adjustment - A Modern Synthesis",
//   /// as a modification to LDLT<_MatrixType,_UpLo>::_solve_impl() from Eigen.
//   /// TODO: Eigen is MPL 2.0 licensed --> check the license and follow its terms.
//   template<typename LDLTType, typename RhsType, typename DstType>
//   void SolveWithSubsetSelection(const LDLTType& ldlt, const RhsType &rhs, DstType &dst) const {
//     eigen_assert(rhs.rows() == ldlt.rows());
//     
//     if (m_rank_deficiency == 0) {
//       dst = ldlt.solve(rhs);
//       return;
//     }
//     
//     // dst = P b
//     dst = ldlt.transpositionsP() * rhs;
//     
//     // Here, dst has the "pivoted" ordering where the least-constrained elements
//     // are at the end. With rd = rank_deficiency, we thus remove the last rd
//     // rows from dst here.
//     int rank = ldlt.rows() - m_rank_deficiency;
//     CHECK_GT(rank, 0);
//     
//     // dst = L^-1 (P b)
//     // Note: This implements ldlt.matrixL() for LDLT::UpLo == Lower based on
//     //       ldlt.matrixLDLT(). It seems that we need to do this manually since
//     //       it does not seem to be possible to apply .topLeftCorner() to the
//     //       result of matrixL().
//     // Original code: matrixL().solveInPlace(dst);
//     TriangularView<const typename LDLTType::MatrixType, UnitLower>(ldlt.matrixLDLT().topLeftCorner(rank, rank))
//         .solveInPlace(dst.topRows(rank));
//     
//     // dst = D^-1 (L^-1 P b)
//     // more precisely, use pseudo-inverse of D (see bug 241)
//     using std::abs;
//     const typename Diagonal<const typename LDLTType::MatrixType>::RealReturnType vecD(ldlt.vectorD());
//     // In some previous versions, tolerance was set to the max of 1/highest (or rather numeric_limits::min())
//     // and the maximal diagonal entry * epsilon as motivated by LAPACK's xGELSS:
//     // RealScalar tolerance = numext::maxi(vecD.array().abs().maxCoeff() * NumTraits<RealScalar>::epsilon(),RealScalar(1) / NumTraits<RealScalar>::highest());
//     // However, LDLT is not rank revealing, and so adjusting the tolerance wrt to the highest
//     // diagonal element is not well justified and leads to numerical issues in some cases.
//     // Moreover, Lapack's xSYTRS routines use 0 for the tolerance.
//     // Using numeric_limits::min() gives us more robustness to denormals.
//     typename LDLTType::RealScalar tolerance = (std::numeric_limits<typename LDLTType::RealScalar>::min)();
//     
//     for (typename LDLTType::Index i = 0; i < rank; ++ i) {
//       if(abs(vecD(i)) > tolerance)
//         dst.row(i) /= vecD(i);
//       else
//         dst.row(i).setZero();
//     }
//     
//     // dst = L^-T (D^-1 L^-1 P b)
//     // Original code: ldlt.matrixU().solveInPlace(dst);
//     TriangularView<const Block<typename LDLTType::MatrixType::AdjointReturnType>, UnitUpper>(ldlt.matrixLDLT().adjoint().topLeftCorner(rank, rank))
//         .solveInPlace(dst.topRows(rank));
//     
//     // Set the removed parts of dst to zero.
//     dst.bottomRows(m_rank_deficiency).setZero();
//     
//     // dst = P^-1 (L^-T D^-1 L^-1 P b) = A^-1 b
//     dst = ldlt.transpositionsP().transpose() * dst;
//   }
  
  /// Whether to use a block-diagonal structure for the Schur complement.
  bool m_use_block_diagonal_structure = false;
  
  /// Block size within block-diagonal part at the top-left of the Hessian.
  int m_block_size = 0;
  
  /// Number of blocks within block-diagonal part at the top-left of the Hessian.
  int m_num_blocks = 0;
  
  /// Rank deficiency of the Hessian (e.g., due to gauge freedom) that should
  /// be accounted for in solving for state updates by setting the
  /// least-constrained variable updates to zero.
  int m_rank_deficiency = 0;
  
  /// The last value of lambda used in Levenberg-Marquardt.
  Scalar m_lambda = -1;
  
  /// Dense part of the Gauss-Newton Hessian approximation.
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m_dense_H;
  /// Block-diagonal part of the Gauss-Newton Hessian approximation.
  vector<Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> m_block_diag_H;
  /// Off-diagonal part of the Gauss-Newton Hessian approximation.
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m_off_diag_H;
  
  /// Solution to the update equation.
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_x;
  
  /// Vector for the right hand side of the update linear equation system,
  /// corresponding to the dense part of H, m_dense_H.
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_dense_b;
  /// Part of b corresponding to m_block_diag_H.
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_block_diag_b;
  
  vector<bool> fixed_variables;
  
 friend class LMOptimizerTestHelper;
};

}
