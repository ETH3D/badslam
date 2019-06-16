# Creates sample points within the area of a 2D sphere within [-1, 1] x [-1, 1].
# Inteded to be used for defining sample points within a stereo matching window.

import argparse
import errno
import math
import os
from pylab import *  # matplotlib
import random
import shutil
import subprocess
import sys


if __name__ == '__main__':
  context_radius = 4
  num_samples = 150  # (2 * context_radius + 1) * (2 * context_radius + 1)
  
  x_list = []
  y_list = []
  
  random.seed(7)
  
  print('constexpr float kSamples[' + str(num_samples) + '][2] = {')
  
  for i in range(0, num_samples):
    while True:
      x = 2 * random.random() - 1
      y = 2 * random.random() - 1
      dist = math.sqrt(x * x + y * y)
      if dist < 1:
        break
    
    x_list.append(x)
    y_list.append(y)
    
    print('    {' + str(x) + 'f, ' + str(y) + 'f}' + (',' if i < num_samples - 1 else ''))
  
  print('};')
  
  # Plot samples
  FIGSIZE_X = 4
  FIGSIZE_Y = 4
  
  fig = plt.figure(figsize=(FIGSIZE_X, FIGSIZE_Y))
  axes = plt.subplot()
  
  axes.plot(x_list, y_list, 'o')
  
  plt.show()
  
  # fig.tight_layout()
  #fig.savefig(file_name, dpi=300, bbox_inches='tight')  # , bbox_extra_artists=(lgd,)   pad_inches=0 )
  
  #close(fig)
  