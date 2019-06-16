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

from sympy import *
from sympy.printing.cxxcode import CXX11CodePrinter


# TODO: Test whether this works with non-matrix functions.
def ComputeValueAndJacobian(function, parameters, parameter_values):
  # Evaluate the function.
  values = function(parameter_values)
  
  # Compute the Jacobian.
  symbolic_values = function(parameters)
  
  if isinstance(symbolic_values, Matrix):
    symbolic_values_vector = symbolic_values.transpose().reshape(symbolic_values.rows * symbolic_values.cols, 1)
  else:
    symbolic_values_vector = diag(symbolic_values)  # Create 1-element matrix to get .jacobian()
  
  if isinstance(parameters, Matrix):
    parameters_vector = parameters.transpose().reshape(parameters.rows * parameters.cols, 1)
  else:
    parameters_vector = diag(parameters)  # Create 1-element matrix to be able to call .jacobian()
  
  jacobian = symbolic_values_vector.jacobian(parameters_vector)
  
  # Set in the evaluation point.
  if isinstance(parameters, Matrix):
    for row in range(0, parameters.rows):
      for col in range(0, parameters.cols):
        jacobian = jacobian.subs(parameters[row, col], parameter_values[row, col])
  else:
    jacobian = jacobian.subs(parameters, parameter_values)
  
  # Simplify the jacobian (after setting in, since this may simplify it a lot).
  jacobian = simplify(jacobian)
  return (values, jacobian)


def MakeInputParameterList(symbols):
  text = ''
  
  for variable_name in sorted([str(a) for a in symbols]):
    if len(text) > 0:
      text = text + ', '
    
    text = text + 'Scalar ' + variable_name
  
  return text


# Takes a list of expressions (and their names) to be computed by the function.
# The expressions should already be simplified beforehand, no further simplification is applied.
def GenerateFunction(function_name, expressions, expression_names, write_rowwise):
  printer = CXX11CodePrinter()
  
  function_parameters = set()
  for expression in expressions:
    function_parameters = function_parameters.union(expression.free_symbols)
  function_parameters = MakeInputParameterList(function_parameters)
  
  text = ''
  
  (replacements, reduced_expressions) = cse(expressions, numbered_symbols('term'))
  
  for replacement in replacements:
    text += 'const Scalar ' + str(replacement[0]) + ' = ' + ccode(replacement[1]) + ';\n'
  if len(replacements) > 0:
    text += '\n'
  
  for reduced_expression, expression_name, write_expression_rowwise in zip(reduced_expressions, expression_names, write_rowwise):
    if isinstance(reduced_expression, ImmutableDenseMatrix):
      if write_expression_rowwise:
        for row in range(0, reduced_expression.rows):
          row_name = (expression_name if reduced_expression.rows == 1 else expression_name + '_row_' + str(row))
          result = MatrixSymbol(row_name, 1, reduced_expression.cols)
          text += printer.doprint(reduced_expression[row, :], assign_to=result) + '\n'
          function_parameters += ', Scalar* ' + row_name
      else:
        result = MatrixSymbol(expression_name, reduced_expression.rows, reduced_expression.cols)
        text += printer.doprint(reduced_expression, assign_to=result) + '\n'
        function_parameters += ', Scalar* ' + expression_name
    else:
      text += '*' + expression_name + ' = ' + ccode(reduced_expression) + ';\n'
      function_parameters += ', Scalar* ' + expression_name
  
  print('')
  print('void ' + function_name + '(' + function_parameters + ') {')
  for line in text.split('\n')[:-1]:  # The last line in text is empty
    print('  ' + line)
  print('}')


def ComputeJacobian(functions, parameters, parameter_values, verbose=False):
  # Print info about initial state.
  if verbose:
    print('Taking the Jacobian of these functions (sorted from inner to outer):')
    for i in range(len(functions) - 1, -1, -1):
      print(str(functions[i]))
    print('with respect to:')
    pprint(parameters)
    print('at position:')
    pprint(parameter_values)
    print('')

  # Jacobian from previous step, dg/dx in the formula above:
  previous_jacobian = 1
  
  current_parameters = parameters
  current_parameter_values = parameter_values

  # Loop over all functions:
  for i in range(len(functions) - 1, -1, -1):
    # Compute value and Jacobian of this function.
    (values, jacobian) = ComputeValueAndJacobian(functions[i], current_parameters, current_parameter_values)
    
    # Update current_parameter_values
    current_parameter_values = values
    
    if verbose:
      print('Intermediate step ' + str(len(functions) - i) + ', for ' + str(functions[i]))
      print('Position after function evaluation (function value):')
      pprint(current_parameter_values)
      print('Jacobian of this function wrt. its input only:')
      pprint(jacobian)
    
    # Update current_parameters (create a new symbolic vector of the same size as current_parameter_values)
    if isinstance(values, Matrix):
      current_parameters = Matrix(values.rows, values.cols, lambda i,j:var('T_%d%d' % (i,j)))
    else:
      current_parameters = symbols("T")
    
    # Concatenate this Jacobian with the previous one according to the chain rule:
    if previous_jacobian != 1:  # Only do this if it matters (to improve performance)
      previous_jacobian = jacobian * previous_jacobian
      previous_jacobian = simplify(previous_jacobian)
    else:
      previous_jacobian = jacobian
    
    # Print intermediate result
    if verbose:
      print('Cumulative Jacobian wrt. the innermost parameter:')
      pprint(previous_jacobian)
      print('')
  
  final_jacobian = previous_jacobian
  
  
  # Compute residual expression
  residual_values = parameter_values
  for i in range(len(functions) - 1, -1, -1):
    residual_values = functions[i](residual_values)
    residual_values = simplify(residual_values)
  
  
  # Print residual
  print('')
  print('Residual at evaluation point (latex style):')
  print(latex(residual_values))
  
  # Print final Jacobian
  print('')
  print('Jacobian (pretty-printed):')
  pprint(final_jacobian)
  
  print('')
  print('Jacobian (Python style):')
  print(final_jacobian)
  
  
  # Generate functions to compute the residual and Jacobian
  GenerateFunction('ComputeJacobian', [final_jacobian], ['jacobian'], [True])
  GenerateFunction('ComputeResidual', [residual_values], ['residuals'], [False])
  GenerateFunction('ComputeResidualAndJacobian', [residual_values, final_jacobian], ['residuals', 'jacobian'], [False, True])
