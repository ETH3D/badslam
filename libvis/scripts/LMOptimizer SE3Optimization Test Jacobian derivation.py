from sympy import *


# Implementation of QuaternionBase<Derived>::toRotationMatrix(void).
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


# Implementation of SO3Group<Scalar> expAndTheta().
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


# Implementation of SE3Group<Scalar> exp().
# Only implementing the first case (of small rotation) since we take the Jacobian at zero.
def SE3exp(tangent):
  omega = Matrix(tangent[3:6])
  V = SO3exp(omega)
  rotation = V
  translation = V * Matrix(tangent[0:3])
  return rotation.row_join(translation)


# Main
init_printing(use_unicode=True)
print('Variant 1')
print('')

# Define the tangent vector with symbolic elements T_0 to T_5.
# (For a matrix, use: Matrix(3, 1, lambda i,j:var('S_%d%d' % (i,j))) )
T = Matrix(6, 1, lambda i,j:var('T_%d' % (i)))

# Compute transformation matrix from tangent vector.
T_matrix = SE3exp(T)

# Define the vector current_T * src:
S = Matrix(3, 1, lambda i,j:var('S_%d' % (i)))

# Matrix-vector multiplication with homogeneous vector:
result = T_matrix * S.col_join(Matrix([1]))

# Compute Jacobian:
# (Note: The transpose is needed for stacking the matrix columns (instead of rows) into a vector.)
jac = result.transpose().reshape(result.rows * result.cols, 1).jacobian(T)

# Take Jacobian at zero:
jac_subs = jac.subs([(T[0], 0), (T[1], 0), (T[2], 0), (T[3], 0), (T[4], 0), (T[5], 0)])

# Simplify and output:
jac_subs_simple = simplify(jac_subs)
pprint(jac_subs_simple)






print('')
print('')
print('Variant 2')
print('')

# Treat the function of which we want to determine the derivative as a list of nested functions.
# This makes it easier to compute the derivative of each part, simplify it, and concatenate the results
# using the chain rule.

### Define the function of which the Jacobian shall be taken ###

# Matrix-vector multiplication with homogeneous vector:
def MatrixVectorMultiplyHomogeneous(matrix, vector):
  return matrix * vector.col_join(Matrix([1]))

# Define the vector current_T * src:
S = Matrix(3, 1, lambda i,j:var('S_%d' % (i)))

# The list of nested functions. They will be evaluated from right to left
# (this is to match the way they would be written in math: f(g(x)).)
functions = [lambda matrix : MatrixVectorMultiplyHomogeneous(matrix, S), SE3exp]


### Define the variables wrt. to take the Jacobian, and the position for evaluation ###

# Chain rule:
# d(f(g(x))) / dx = (df/dy)(g(x)) * dg/dx

# Define the parameter with respect to take the Jacobian, y in the formula above:
parameters = Matrix(6, 1, lambda i,j:var('T_%d' % (i)))

# Set the position at which to take the Jacobian, g(x) in the formula above:
parameter_values = zeros(6, 1)


### Automatic Jacobian calculation, no need to modify anything beyond this point ###

# Jacobian from previous step, dg/dx in the formula above:
previous_jacobian = 1

# TODO: Test whether this works with non-matrix functions.
def ComputeValueAndJacobian(function, parameters, parameter_values):
  # Evaluate the function.
  values = function(parameter_values)
  # Compute the Jacobian.
  symbolic_values = function(parameters)
  symbolic_values_vector = symbolic_values.transpose().reshape(symbolic_values.rows * symbolic_values.cols, 1)
  parameters_vector = parameters.transpose().reshape(parameters.rows * parameters.cols, 1)
  jacobian = symbolic_values_vector.jacobian(parameters_vector)
  # Set in the evaluation point.
  for row in range(0, parameters.rows):
    for col in range(0, parameters.cols):
      jacobian = jacobian.subs(parameters[row, col], parameter_values[row, col])
  # Simplify the jacobian.
  jacobian = simplify(jacobian)
  return (values, jacobian)


# Print info about initial state.
print('Taking the Jacobian of these functions (sorted from inner to outer):')
for i in range(len(functions) - 1, -1, -1):
  print(str(functions[i]))
print('with respect to:')
pprint(parameters)
print('at position:')
pprint(parameter_values)
print('')

# Loop over all functions:
for i in range(len(functions) - 1, -1, -1):
  # Compute value and Jacobian of this function.
  (values, jacobian) = ComputeValueAndJacobian(functions[i], parameters, parameter_values)
  
  # Update parameter_values
  parameter_values = values
  # Update parameters (create a new symbolic vector of the same size as parameter_values)
  parameters = Matrix(values.rows, values.cols, lambda i,j:var('T_%d%d' % (i,j)))
  # Concatenate this Jacobian with the previous one according to the chain rule:
  previous_jacobian = jacobian * previous_jacobian
  
  # Print intermediate result
  print('Intermediate step ' + str(len(functions) - i) + ', for ' + str(functions[i]))
  print('Position after function evaluation (function value):')
  pprint(parameter_values)
  print('Jacobian of this function wrt. its input only:')
  pprint(jacobian)
  print('Cumulative Jacobian wrt. the innermost parameter:')
  pprint(previous_jacobian)
  print('')

# Print final result
print('Final result:')
pprint(previous_jacobian)
