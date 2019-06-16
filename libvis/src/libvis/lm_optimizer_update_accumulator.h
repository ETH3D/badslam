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

#include "libvis/eigen.h"
#include "libvis/libvis.h"

namespace vis {

template <typename Scalar>
class UpdateEquationAccumulator {
 public:
  UpdateEquationAccumulator(
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* dense_H,
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* off_diag_H,
      vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>* block_diag_H,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* dense_b,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* block_diag_b,
      vector<Scalar>* residual_cost_vector = nullptr,
      vector<bool>* residual_validity_vector = nullptr)
      : cost_(0),
        dense_H_(dense_H),
        off_diag_H_(off_diag_H),
        block_diag_H_(block_diag_H),
        dense_b_(dense_b),
        block_diag_b_(block_diag_b),
        residual_cost_vector_(residual_cost_vector),
        residual_validity_vector_(residual_validity_vector) {
    if (dense_H_) {
      dense_H_->setZero();
    }
    if (off_diag_H) {
      off_diag_H->setZero();
    }
    if (block_diag_H) {
      for (auto& matrix : *block_diag_H) {
        matrix.setZero();
      }
    }
    if (dense_b_) {
      dense_b_->setZero();
    }
    if (block_diag_b_) {
      block_diag_b_->setZero();
    }
  }
  
  // TODO: Need more AddResidualWithJacobian() variants (3 dense blocks,
  //       4 dense blocks, dense block + sparse index list, ...)
  
  inline void AddInvalidResidual() {
    residual_cost_vector_->emplace_back(-1);
    residual_validity_vector_->push_back(false);
  }
  
  /// To be called by CostFunction to add a residual to the cost.
  template <typename LossFunctionT = QuadraticLoss>
  inline void AddResidual(
      Scalar residual,
      const LossFunctionT& loss_function = LossFunctionT()) {
    Scalar residual_cost = loss_function.ComputeCost(residual); 
    cost_ += residual_cost;
    if (residual_cost_vector_) {
      residual_cost_vector_->emplace_back(residual_cost);
    }
    if (residual_validity_vector_) {
      residual_validity_vector_->push_back(true);
    }
  }
  
  /// To be called by CostFunction to add one scalar residual with
  /// its Jacobian. The Jacobian corresponds to the entries
  /// [index, index + jacobian.rows() - 1] of the state. If the Jacobian entries
  /// are all non-zero, it will result in one dense block in H. This function is
  /// for fixed-size Jacobians only and will not compile otherwise. If your
  /// Jacobian contains zeros, consider using another variant of this function
  /// to increase performance.
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived>
  inline void AddResidualWithJacobian(
      Scalar residual,
      u32 index,
      const MatrixBase<Derived>& jacobian,
      const LossFunctionT& loss_function = LossFunctionT()) {
    AddResidual(residual, loss_function);
    
    const Scalar weight = loss_function.ComputeWeight(residual);
    const Scalar weighted_residual = weight * residual;
    
    if (index < block_diag_b_->rows()) {
      int block_index = index / block_diag_H_->at(0).rows();
      int index_in_block = index - block_diag_H_->at(0).rows() * block_index;
      CHECK_LE(index_in_block + Derived::ColsAtCompileTime, block_diag_H_->at(0).rows())
          << "Overlapping Jacobians between the block-diagonal and dense parts"
              " is not allowed with this function. Overlapping Jacobians between"
              " multiple blocks is not allowed in general (otherwise, they would"
              " not be individual blocks).";
      
      AddToHandBOnDiagonal(&block_diag_H_->at(block_index), block_diag_b_, weight, weighted_residual,
                           index_in_block, index, jacobian);
    } else {
      u32 index_in_dense_part = index - block_diag_b_->rows();
      AddToHandBOnDiagonal(dense_H_, dense_b_, weight, weighted_residual,
                           index_in_dense_part, index_in_dense_part, jacobian);
    }
  }
  
  /// Variant of AddJacobian() for Jacobians consisting of two dense blocks
  /// with zeros in-between. Avoids processing the zeros and can therefore
  /// achieve higher performance in this case. It must hold: index0 < index1.
  /// The enableX parameters can be used to disable parts of the Jacbian, which
  /// allows to do this dynamically without calling different functions in
  /// different cases.
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
    if (enable0 && !enable1) {
      AddResidualWithJacobian(residual, index0, jacobian0, loss_function);
      return;
    } else if (!enable0 && enable1) {
      AddResidualWithJacobian(residual, index1, jacobian1, loss_function);
      return;
    } else if (!enable0 && !enable1) {
      AddResidual(residual, loss_function);
      return;
    }
    
    AddResidual(residual, loss_function);
    
    const Scalar weight = loss_function.ComputeWeight(residual);
    const Scalar weighted_residual = weight * residual;
    
    // TODO: The if-then-else cases below are a mess. If we introduce functions with more than two Jacobian parts, it would get much worse. Can we simplify this?
    if (index0 < block_diag_b_->rows()) {
      int block_index0 = index0 / block_diag_H_->at(0).rows();
      int index_in_block0 = index0 - block_diag_H_->at(0).rows() * block_index0;
      auto* H_block0 = &block_diag_H_->at(block_index0);
      
      AddToHandBOnDiagonal(H_block0, block_diag_b_, weight, weighted_residual,
                           index_in_block0, index0, jacobian0);
      
      if (index1 < block_diag_b_->rows()) {
        int block_index1 = index1 / block_diag_H_->at(0).rows();
        int index_in_block1 = index1 - block_diag_H_->at(0).rows() * block_index1;
        CHECK_EQ(block_index0, block_index1)
            << "Jacobian cannot span two blocks (otherwise they would not be individual blocks)";
        
        // Both parts of the Jacobian are in the same block-diagonal block of H.
        AddToHandBOnDiagonal(H_block0, block_diag_b_, weight, weighted_residual,
                             index_in_block1, index1, jacobian1);
        H_block0->template block<Derived0::ColsAtCompileTime, Derived1::ColsAtCompileTime>(index_in_block0, index_in_block1) +=
            (weight * jacobian0.transpose() * jacobian1).template cast<Scalar>();
      } else {
        // The first part of the Jacobian is in a block-diagonal block of H,
        // the second part is in the dense part.
        u32 index1_in_dense_part = index1 - block_diag_b_->rows();
        AddToHandBOnDiagonal(dense_H_, dense_b_, weight, weighted_residual,
                             index1_in_dense_part, index1_in_dense_part, jacobian1);
        off_diag_H_->template block<Derived0::ColsAtCompileTime, Derived1::ColsAtCompileTime>(index0, index1 - block_diag_b_->rows()) +=
            (weight * jacobian0.transpose() * jacobian1).template cast<Scalar>();
      }
    } else {
      // Both parts of the Jacobian are in the dense part of H.
      u32 index0_in_dense_part = index0 - block_diag_b_->rows();
      AddToHandBOnDiagonal(dense_H_, dense_b_, weight, weighted_residual,
                           index0_in_dense_part, index0_in_dense_part, jacobian0);
      u32 index1_in_dense_part = index1 - block_diag_b_->rows();
      AddToHandBOnDiagonal(dense_H_, dense_b_, weight, weighted_residual,
                           index1_in_dense_part, index1_in_dense_part, jacobian1);
      dense_H_->template block<Derived0::ColsAtCompileTime, Derived1::ColsAtCompileTime>(index0_in_dense_part, index1_in_dense_part) +=
          (weight * jacobian0.transpose() * jacobian1).template cast<Scalar>();
    }
  }
  
  /// Variant of AddJacobian() which takes equally-sized vectors indices and
  /// jacobian, where each corresponding pair of entries specifies a Jacobian
  /// component and its index. The indices must be in increasing order.
  // TODO: Would it be helpful to have a variant of this which allows to diable
  //       individual parts of the jacobian?
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1>
  inline void AddResidualWithJacobian(
      Scalar residual,
      const MatrixBase<Derived0>& indices,
      const MatrixBase<Derived1>& jacobian,
      const LossFunctionT& loss_function = LossFunctionT()) {
    AddResidual(residual, loss_function);
    
    const Scalar weight = loss_function.ComputeWeight(residual);
    const Scalar weighted_residual = weight * residual;
    
    if (block_diag_b_->rows() > 0) {
      for (int i = 0; i < indices.size(); ++ i) {
        for (int k = i; k < indices.size(); ++ k) {
          HAtSlow(indices[i], indices[k]) += (weight * jacobian[i] * jacobian[k]);
        }
        
        if (indices[i] < block_diag_b_->rows()) {
          (*block_diag_b_)(indices[i]) += weighted_residual * jacobian[i];
        } else {
          (*dense_b_)(indices[i] - block_diag_b_->rows()) += weighted_residual * jacobian[i];
        }
      }
    } else {
      for (int i = 0; i < indices.size(); ++ i) {
        for (int k = i; k < indices.size(); ++ k) {
          (*dense_H_)(indices[i], indices[k]) += (weight * jacobian[i] * jacobian[k]);
        }
        
        (*dense_b_)(indices[i]) += weighted_residual * jacobian[i];
      }
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
    // Add all blocks that do not involve the 3rd index block
    AddResidualWithJacobian(
        residual, index0, jacobian0, index1, jacobian1,
        enable0, enable1, loss_function);
    
    const Scalar weight = loss_function.ComputeWeight(residual);
    const Scalar weighted_residual = weight * residual;
    
    for (int i = 0; i < indices2.size(); ++ i) {
      // jacobian0.transpose() * jacobian2
      if (enable0) {
        for (int k = 0; k < Derived0::ColsAtCompileTime; ++ k) {
          HAtSlow(index0 + k, indices2[i]) += (weight * jacobian2[i] * jacobian0[k]);
        }
      }
      
      // jacobian1.transpose() * jacobian2
      if (enable1) {
        for (int k = 0; k < Derived1::ColsAtCompileTime; ++ k) {
          HAtSlow(index1 + k, indices2[i]) += (weight * jacobian2[i] * jacobian1[k]);
        }
      }
      
      // jacobian2.transpose() * jacobian2
      for (int k = i; k < indices2.size(); ++ k) {
        HAtSlow(indices2[i], indices2[k]) += (weight * jacobian2[i] * jacobian2[k]);
      }
      
      if (indices2[i] < block_diag_b_->rows()) {
        (*block_diag_b_)(indices2[i]) += weighted_residual * jacobian2[i];
      } else {
        (*dense_b_)(indices2[i] - block_diag_b_->rows()) += weighted_residual * jacobian2[i];
      }
    }
  }
  
  inline Scalar cost() const { return cost_; }
  
  /// For debugging purposes, outputs the complete H matrix and b vector.
  void GetHandB(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H,
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* b) {
    if (block_diag_b_->rows() == 0) {
      *H = *dense_H_;
      *b = *dense_b_;
    } else {
      int H_size = block_diag_b_->rows() + dense_b_->rows();
      
      H->resize(H_size, H_size);
      int block_size = block_diag_H_->at(0).rows();
      H->topLeftCorner(block_size * block_diag_H_->size(),
                       block_size * block_diag_H_->size()).setZero();
      for (usize i = 0; i < block_diag_H_->size(); ++ i) {
        H->block(i * block_size, i * block_size, block_size, block_size) =
            block_diag_H_->at(i);
      }
      H->topRightCorner(off_diag_H_->rows(), off_diag_H_->cols()) = *off_diag_H_;
      H->bottomRightCorner(dense_H_->rows(), dense_H_->cols()) = *dense_H_;
      
      b->resize(H_size);
      b->topRows(block_diag_b_->rows()) = *block_diag_b_;
      b->bottomRows(dense_b_->rows()) = *dense_b_;
    }
    
    H->template triangularView<Eigen::Lower>() = H->template triangularView<Eigen::Upper>().transpose();
  }
  
 private:
  inline Scalar& HAtSlow(int row, int col) {
    if (row < block_diag_b_->rows() &&
        col < block_diag_b_->rows()) {
      // Access in block-diagonal part (block_diag_H_).
      int block_index = row / block_diag_H_->at(0).rows();
      int row_in_block = row - block_diag_H_->at(0).rows() * block_index;
      int col_in_block = col - block_diag_H_->at(0).rows() * block_index;
      auto* H_block = &block_diag_H_->at(block_index);
      CHECK_GE(col_in_block, 0) << "Trying to access invalid matrix element (not following block-diagonal structure)";
      CHECK_LT(col_in_block, H_block->cols()) << "Trying to access invalid matrix element (not following block-diagonal structure)";
      return (*H_block)(row_in_block, col_in_block);
    } else if (row < block_diag_b_->rows()) {
      // Access in upper-right part (off_diag_H_).
      u32 col_in_dense_part = col - block_diag_b_->rows();
      CHECK_LT(col_in_dense_part, off_diag_H_->cols());
      return (*off_diag_H_)(row, col_in_dense_part);
    } else {
      // Access in lower-right part (dense_H_).
      u32 row_in_dense_part = row - block_diag_b_->rows();
      u32 col_in_dense_part = col - block_diag_b_->rows();
      CHECK_LT(row_in_dense_part, dense_H_->rows());
      CHECK_LT(col_in_dense_part, dense_H_->cols());
      return (*dense_H_)(row_in_dense_part, col_in_dense_part);
    }
  }
  
  template <typename Derived>
  inline void AddToHandBOnDiagonal(
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* b,
      Scalar weight,
      Scalar weighted_residual,
      u32 index_in_H,
      u32 index_in_b,
      const MatrixBase<Derived>& jacobian) {
    // TODO: Limit these checks to debug compile modes
    CHECK_GT(Derived::ColsAtCompileTime, 0);
    CHECK_LE(index_in_H + Derived::ColsAtCompileTime, H->rows());
    CHECK_LE(index_in_b + Derived::ColsAtCompileTime, b->rows());
    
    H->template block<Derived::ColsAtCompileTime, Derived::ColsAtCompileTime>(index_in_H, index_in_H)
        .template triangularView<Eigen::Upper>() +=
            (weight * jacobian.transpose() * jacobian).template cast<Scalar>();
    
    b->template segment<Derived::ColsAtCompileTime>(index_in_b) +=
        (weighted_residual * jacobian.transpose()).template cast<Scalar>();
  }
  
  Scalar cost_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* dense_H_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* off_diag_H_;
  vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>* block_diag_H_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* dense_b_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* block_diag_b_;
  
  vector<Scalar>* residual_cost_vector_;
  vector<bool>* residual_validity_vector_;
};

}
