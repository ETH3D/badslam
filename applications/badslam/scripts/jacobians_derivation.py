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

import math
import sys
import time

from sympy import *

from jacobian_functions import *


# ### Math functions ###

# Implementation of Eigen::QuaternionBase<Derived>::toRotationMatrix(void).
# The quaternion q is given as a list [qw, qx, qy, qz].
def QuaternionToRotationMatrix(q):
  tx  = 2 * q[1]
  ty  = 2 * q[2]
  tz  = 2 * q[3]
  twx = tx * q[0]
  twy = ty * q[0]
  twz = tz * q[0]
  txx = tx * q[1]
  txy = ty * q[1]
  txz = tz * q[1]
  tyy = ty * q[2]
  tyz = tz * q[2]
  tzz = tz * q[3]
  return Matrix([[1 - (tyy + tzz), txy - twz, txz + twy],
                 [txy + twz, 1 - (txx + tzz), tyz - twx],
                 [txz - twy, tyz + twx, 1 - (txx + tyy)]])


# Implementation of Sophus::SO3Group<Scalar> expAndTheta().
# Only implementing the first case (of very small rotation) since we take the Jacobian at zero.
def SO3exp(omega):
  theta = omega.norm()
  theta_sq = theta**2
  
  half_theta = theta / 2
  
  theta_po4 = theta_sq * theta_sq
  imag_factor = Rational(1, 2) - Rational(1, 48) * theta_sq + Rational(1, 3840) * theta_po4;
  real_factor = 1 - Rational(1, 2) * theta_sq + Rational(1, 384) * theta_po4;
  
  # return SO3Group<Scalar>(Eigen::Quaternion<Scalar>(
  #     real_factor, imag_factor * omega.x(), imag_factor * omega.y(),
  #     imag_factor * omega.z()));
  qw = real_factor
  qx = imag_factor * omega[0]
  qy = imag_factor * omega[1]
  qz = imag_factor * omega[2]
  
  return QuaternionToRotationMatrix([qw, qx, qy, qz])


# Implementation of Sophus::SE3Group<Scalar> exp().
# Only implementing the first case (of small rotation) since we take the Jacobian at zero.
def SE3exp(tangent):
  omega = Matrix(tangent[3:6])
  V = SO3exp(omega)
  rotation = V
  translation = V * Matrix(tangent[0:3])
  return rotation.row_join(translation)


# Matrix-vector multiplication with homogeneous vector:
def MatrixVectorMultiplyHomogeneous(matrix, vector):
  return matrix * vector.col_join(Matrix([1]))


# Multiplication of two 3x4 matrices where the last rows are assumed to be [0, 0, 0, 1].
def MatrixMatrixMultiplyHomogeneous(left, right):
  return left * right.col_join(Matrix([[0, 0, 0, 1]]))


# Inverse of SE3 3x4 matrix.
# Derivation: solve (R * x + t = y) for x:
# <=>  x + R^(-1) t = R^(-1) y
# <=>  x = R^(-1) y - R^(-1) t
def SE3Inverse(matrix):
  R_inv = Matrix([[matrix[0, 0], matrix[1, 0], matrix[2, 0]],
                  [matrix[0, 1], matrix[1, 1], matrix[2, 1]],
                  [matrix[0, 2], matrix[1, 2], matrix[2, 2]]])
  t_inv = -R_inv * Matrix([[matrix[0, 3]], [matrix[1, 3]], [matrix[2, 3]]])
  return R_inv.row_join(t_inv)


# 3-Vector dot product:
def DotProduct3(vector1, vector2):
  return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]

# 3D point projection onto a pinhole image:
def Project(point, fx, fy, cx, cy):
  return Matrix([fx * point[0] / point[2] + cx,
                 fy * point[1] / point[2] + cy])

# 3D point projection to a stereo observation (x, y, right_x) the way ORB-SLAM2 does it:
def ProjectStereo(point, fx, fy, cx, cy, bf):
  x = fx * point[0] / point[2] + cx
  return Matrix([x,
                 fy * point[1] / point[2] + cy,
                 x - bf / point[2]])

# Point un-projection from image to 3D:
def Unproject(x, y, depth, fx_inv, fy_inv, cx_inv, cy_inv):
  return Matrix([[depth * (fx_inv * x + cx_inv)],
                 [depth * (fy_inv * y + cy_inv)],
                 [depth]])

# Depth correction function (takes inverse depth, but returns non-inverse depth):
def CorrectDepth(cfactor, a, inv_depth):
  return 1 / (inv_depth + cfactor * exp(- a * inv_depth))

# Simple model for the fractional-part function used for bilinear interpolation
# which leaves the function un-evaluated. Ignores the discontinuities when
# computing the derivative. They do not matter.
class frac(Function):
  # Returns the first derivative of the function.
  # A simple model for the function within the range between two discontinuities is:
  # f(x) = x - c, with a constant c. So f'(x) = 1.
  def fdiff(self, argindex=1):
    if argindex == 1:
      return S.One
    else:
      raise ArgumentIndexError(self, argindex)

# Bilinear interpolation using the fractional-part function model from above.
# x and y are expected to be in the range between 0 and 1.
# (x, y) == (0, 0) would return the value top_left,
# (x, y) == (1, 1) would return the value bottom_right, etc.
def InterpolateBilinear(x, y, top_left, top_right, bottom_left, bottom_right):
  fx = frac(x)
  fy = frac(y)
  return (1 - fy) * ((1 - fx) * top_left + fx * top_right) + fy * ((1 - fx) * bottom_left + fx * bottom_right)


# ### Cost function setup and Jacobian computation ###

if __name__ == '__main__':
  init_printing()
  
  # Depth residual (with deltas T and t):
  # inv_sigma * dot(surfel_normal, global_T_frame * exp(hat(T)) * unproject(x, y, correct(x, y, depth)) - (surfel_pos + t * surfel_normal))
  #
  # NOTE: Can be reformulated to local space:
  # inv_sigma * dot(local_surfel_normal, unproject(x, y, correct(x, y, depth)) - SE3Inverse(exp(hat(T))) * frame_T_global * (surfel_pos + t * surfel_normal))
  #
  # with correct(x, y, depth) = 1 / (inv_depth + cfactor(x, y) * exp(- a * inv_depth))
  #      cfactor : correction factor that can be defined per-pixel or for blocks of pixels (i.e., on a lower resolution image).
  #      a : global parameter for the depth correction, to be optimized.
  
  # Define variables
  surfel_normal = Matrix(3, 1, lambda i,j:Symbol('n_%d' % (i), real=True))
  global_T_frame = Matrix(3, 4, lambda i,j:Symbol('gtf_%d_%d' % (i, j), real=True))
  frame_T_global = Matrix(3, 4, lambda i,j:Symbol('ftg_%d_%d' % (i, j), real=True))
  local_point = Matrix(3, 1, lambda i,j:Symbol('l_%d' % (i), real=True))  # unproject(x, y, correct(depth))
  global_point = Matrix(3, 1, lambda i,j:Symbol('g_%d' % (i), real=True))  # global_T_frame * unproject(x, y, correct(depth))
  surfel_pos = Matrix(3, 1, lambda i,j:Symbol('s_%d' % (i), real=True))
  local_surfel_pos = Matrix(3, 1, lambda i,j:Symbol('ls_%d' % (i), real=True))
  fx = Symbol("fx", real=True)
  fy = Symbol("fy", real=True)
  cx = Symbol("cx", real=True)
  cy = Symbol("cy", real=True)
  fx_inv = Symbol("fx_inv", real=True)
  fy_inv = Symbol("fy_inv", real=True)
  cx_inv = Symbol("cx_inv", real=True)
  cy_inv = Symbol("cy_inv", real=True)
  x = Symbol("x", real=True)
  y = Symbol("y", real=True)
  t = Symbol("t", real=True)
  depth = Symbol("depth", real=True)
  raw_inv_depth = Symbol("raw_inv_depth", real=True)
  cfactor = Symbol("cfactor", real=True)
  a = Symbol("a", real=True)
  top_left = Symbol("top_left", real=True)
  top_right = Symbol("top_right", real=True)
  bottom_left = Symbol("bottom_left", real=True)
  bottom_right = Symbol("bottom_right", real=True)
  surfel_gradmag = Symbol("surfel_gradmag", real=True)
  
  determine_depth_jacobians = True
  if not determine_depth_jacobians:
    print('Determining depth jacobians is deactivated')
    print('')
  if determine_depth_jacobians:
    # TODO: multiplication with inv_sigma is not included here!
    
    # Jacobian of depth residual wrt. global_T_frame changes (using delta: T):
    # dot(surfel_normal, global_T_frame * exp(hat(T)) * unproject(x, y, correct(depth)) - surfel_pos)
    print('### Jacobian of depth residual wrt. global_T_frame changes ###')
    functions = [lambda point: DotProduct3(surfel_normal, point),
                 lambda point : point - surfel_pos,
                 lambda point : MatrixVectorMultiplyHomogeneous(global_T_frame, point),
                 lambda matrix : MatrixVectorMultiplyHomogeneous(matrix, local_point),
                 SE3exp]
    parameters = Matrix(6, 1, lambda i,j:var('T_%d' % (i)))
    parameter_values = zeros(6, 1)
    ComputeJacobian(functions, parameters, parameter_values)
    print('')
    print('')
    
    # Jacobian of depth residual wrt. surfel_pos changes (using delta: t)
    # Rewrite cost term with delta as:
    # dot(surfel_normal, global_T_frame * unproject(x, y, correct(depth)) - (surfel_pos + t * surfel_normal))
    print('### Jacobian of depth residual wrt. surfel position changes ###')
    functions = [lambda point: DotProduct3(surfel_normal, point),
                 lambda surfel : global_point - surfel,
                 lambda pos_delta : surfel_pos + pos_delta,
                 lambda t : t * surfel_normal]
    ComputeJacobian(functions, t, 0)
    print('')
    print('')
    
    # Jacobian of depth residual wrt. intrinsics (fx_inv, fy_inv, cx_inv, cy_inv) changes
    print('### Jacobian of depth residual wrt. intrinsics changes ###')
    functions = [lambda point: DotProduct3(surfel_normal, point),
                 lambda point : point - surfel_pos,
                 lambda local_point : MatrixVectorMultiplyHomogeneous(global_T_frame, local_point),
                 lambda intrinsics : Unproject(x, y, depth, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3])]
    parameters = Matrix([[fx_inv], [fy_inv], [cx_inv], [cy_inv]])
    ComputeJacobian(functions, parameters, parameters)
    print('')
    print('')
    
    # Jacobian of depth residual wrt. depth correction changes
    print('### Jacobian of depth residual wrt. depth correction changes ###')
    functions = [lambda point: DotProduct3(surfel_normal, point),
                 lambda point : point - surfel_pos,
                 lambda local_point : MatrixVectorMultiplyHomogeneous(global_T_frame, local_point),
                 lambda depth : Unproject(x, y, depth, fx_inv, fy_inv, cx_inv, cy_inv),
                 lambda parameters : CorrectDepth(parameters[0], parameters[1], raw_inv_depth)]
    parameters = Matrix([[cfactor], [a]])
    ComputeJacobian(functions, parameters, parameters)
    print('')
    print('')
  
  
  # Descriptor residual based on gradient magnitude (gradmag; with deltas T and t):
  # NOTE: The SE3Inverse() makes this T delta the same as in the depth
  #       residual above such that both types of residuals can be combined. It
  #       could be read "old_frame_T_new_frame".
  # interp_bilinear(gradmag_image, Project(SE3Inverse(exp(hat(T))) * frame_T_global * (surfel_pos + t * surfel_normal))) - surfel_gradmag
  
  determine_descriptor_jacobians = True
  if not determine_descriptor_jacobians:
    print('Determining descriptor jacobians is deactivated')
    print('')
  if determine_descriptor_jacobians:
    # Jacobian of descriptor residual wrt. global_T_frame changes (using delta: T):
    print('### Jacobian of descriptor residual wrt. global_T_frame changes ###')
    functions = [lambda descriptor : descriptor - surfel_gradmag,
                 lambda point : InterpolateBilinear(point[0], point[1], top_left, top_right, bottom_left, bottom_right),
                 lambda point : Project(point, fx, fy, cx, cy),
                 lambda left : MatrixVectorMultiplyHomogeneous(left, local_surfel_pos),
                 SE3Inverse,
                 SE3exp]
    parameters = Matrix(6, 1, lambda i,j:var('T_%d' % (i)))
    parameter_values = zeros(6, 1)
    ComputeJacobian(functions, parameters, parameter_values)
    print('')
    print('')
    
    # Jacobian of descriptor residual wrt. surfel_pos changes (using delta: t)
    # Rewrite cost term with delta as:
    # interp_bilinear(gradmag_image, Project(frame_T_global * (surfel_pos + t * surfel_normal))) - surfel_gradmag
    print('### Jacobian of descriptor residual wrt. surfel position changes ###')
    functions = [lambda descriptor : descriptor - surfel_gradmag,
                 lambda point : InterpolateBilinear(point[0], point[1], top_left, top_right, bottom_left, bottom_right),
                 lambda point : Project(point, fx, fy, cx, cy),
                 lambda global_surfel : MatrixVectorMultiplyHomogeneous(frame_T_global, global_surfel),
                 lambda pos_delta : surfel_pos + pos_delta,
                 lambda t : t * surfel_normal]
    ComputeJacobian(functions, t, 0)
    print('')
    print('')
    
    # Jacobian of descriptor residual wrt. intrinsics (fx_inv, fy_inv, cx_inv, cy_inv) changes
    print('### Jacobian of descriptor residual wrt. intrinsics changes ###')
    functions = [lambda descriptor : descriptor - surfel_gradmag,
                 lambda point : InterpolateBilinear(point[0], point[1], top_left, top_right, bottom_left, bottom_right),
                 lambda intrinsics : Project(local_surfel_pos, 1 / intrinsics[0], 1 / intrinsics[1], -intrinsics[2] / intrinsics[0], -intrinsics[3] / intrinsics[1])]
    parameters = Matrix([[fx_inv], [fy_inv], [cx_inv], [cy_inv]])
    ComputeJacobian(functions, parameters, parameters)
    print('')
    print('')
  
  
  # NOTE: The sparse keypoint residual below is not used in the BAD SLAM application currently.
  # 
  # Sparse keypoint residual (with delta T):
  # TODO: Include depth correction with CorrectStereo() for 3D keypoint observations
  # 
  # For 3D keypoint observations:
  #   e_sigma_normalized := inv_sigma_octave * (CorrectStereo(keypoint_3d_observation) - ProjectStereo(SE3Inverse(exp(hat(T))) * frame_T_global * keypoint_pos_3d, bf))
  # For 2D keypoint observations:
  #   e_sigma_normalized := inv_sigma_octave * (keypoint_2d_observation - Project(SE3Inverse(exp(hat(T))) * frame_T_global * keypoint_pos_3d))
  # 
  # Weight based on: sqrt(e_raw^T * InformationMatrix * e_raw)
  # In practice, ORB-SLAM2 uses a diagonal information matrix with all diagonal values being equal: inv_sigma_octave^2.
  # This means we can treat this case the same way as multiplying the raw residual with inv_sigma_octave (without the square).
  determine_sparse_jacobians = False
  if not determine_sparse_jacobians:
    print('Determining sparse jacobians is deactivated')
    print('')
  if determine_sparse_jacobians:
    inv_sigma_octave = Symbol("inv_sigma_octave", real=True)
    keypoint_2d_observation = Matrix(2, 1, lambda i,j:var('obs2d_%d' % (i)))
    keypoint_3d_observation = Matrix(3, 1, lambda i,j:var('obs3d_%d' % (i)))
    bf = Symbol("bf", real=True)
    local_keypoint_pos = Matrix(3, 1, lambda i,j:var('lkp_%d' % (i)))
    
    # Jacobian of sparse 3D residual wrt. global_T_frame changes (using delta: T):
    print('### Jacobian of sparse 3D keypoint observation residual wrt. global_T_frame changes ###')
    functions = [lambda projected_point : inv_sigma_octave * (keypoint_3d_observation - projected_point),
                 lambda point : ProjectStereo(point, fx, fy, cx, cy, bf),
                 lambda left : MatrixVectorMultiplyHomogeneous(left, local_keypoint_pos),
                 SE3Inverse,
                 SE3exp]
    parameters = Matrix(6, 1, lambda i,j:var('T_%d' % (i)))
    parameter_values = zeros(6, 1)
    #start_time = time.time()
    ComputeJacobian(functions, parameters, parameter_values)
    print('')
    #print("RUNTIME for chain of functions: %s seconds" % round(time.time() - start_time, 2))
    print('')
    
    # Jacobian of sparse 2D residual wrt. global_T_frame changes (using delta: T):
    print('### Jacobian of sparse 2D keypoint observation residual wrt. global_T_frame changes ###')
    functions = [lambda projected_point : inv_sigma_octave * (keypoint_2d_observation - projected_point),
                 lambda point : Project(point, fx, fy, cx, cy),
                 lambda left : MatrixVectorMultiplyHomogeneous(left, local_keypoint_pos),
                 SE3Inverse,
                 SE3exp]
    parameters = Matrix(6, 1, lambda i,j:var('T_%d' % (i)))
    parameter_values = zeros(6, 1)
    ComputeJacobian(functions, parameters, parameter_values)
    print('')
    print('')
    
    # Jacobian of sparse 3D residual wrt. keypoint 3D position changes
    print('### Jacobian of sparse 3D keypoint observation residual wrt. keypoint 3D position changes ###')
    functions = [lambda projected_point : inv_sigma_octave * (keypoint_3d_observation - projected_point),
                 lambda point : ProjectStereo(point, fx, fy, cx, cy, bf),
                 lambda right : MatrixVectorMultiplyHomogeneous(frame_T_global, right)]
    parameters = Matrix(3, 1, lambda i,j:var('kp_%d' % (i)))
    ComputeJacobian(functions, parameters, parameters)
    print('')
    print('')
    
    # Jacobian of sparse 2D residual wrt. keypoint 3D position changes
    print('### Jacobian of sparse 2D keypoint observation residual wrt. keypoint 3D position changes ###')
    functions = [lambda projected_point : inv_sigma_octave * (keypoint_2d_observation - projected_point),
                 lambda point : Project(point, fx, fy, cx, cy),
                 lambda right : MatrixVectorMultiplyHomogeneous(frame_T_global, right)]
    parameters = Matrix(3, 1, lambda i,j:var('kp_%d' % (i)))
    ComputeJacobian(functions, parameters, parameters)
    print('')
    print('')
