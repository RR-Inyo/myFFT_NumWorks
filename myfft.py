# myfft.py
# Implementation of FFT and DFT for NumWorks calculators
# (c) 2021 @RR_Inyo
# Released under the MIT lisence
# https://opensource.org/licenses/mit-license.php

import cmath

# DFT, straight-forward Discrete Fourier Transform 
# Caution! Original input data is overwritten and destroyed.
def DFT(f):
  N = len(f)
  y = [0] * N # List to contain results, zero-initialized
  
  # DFT core calculation
  for i in range(N):
    for j in range(N):
      # Prepare complex sinusoid
      W = cmath.exp(-2j * cmath.pi * i * j / N) 
      
      # multiply-accumulate computation
      y[i] += W * f[j]
  
  # Overwrite results to input
  for i in range(N):
    f[i] = y[i]

# FFT, non-recursive, decimation-in-frequency version
# Bit-reversal reordering
# Caution! Original input data is overwritten and destroyed.
def FFT(f):
  N = len(f)
  p = int((cmath.log10(N) / cmath.log10(2)).real)
  
  # FFT core calculation
  for i in range(0, p):     # Number of time-domain data divisions
    n = N // (2**(i + 1))
    for j in range(2 ** i): # Counter for divided time-domain data
      for k in range(n):    # Counter for each item in divided time-domain data
        # Prepare complex sinusoid, rotation operator
        W = cmath.exp(-2j * cmath.pi * k / (2 * n))
      
        # Butterfly computation
        f_tmp = f[n * 2 * j + k] - f[n * (2 * j + 1) + k]
        f[n * 2 * j + k] = f[n * 2 * j + k] + f[n * (2 * j + 1) + k]
        f[n * (2 * j + 1) + k] = W * f_tmp
  
  # Bit-reverse reordering
  i = 0
  for j in range(1, N - 1):
    k = N >> 1
    while(True):
      i ^= k
      if k > i:
        k >>= 1
        continue
      else:
        break 
    if j < i:
      f_tmp = f[j]
      f[j] = f[i]
      f[i] = f_tmp

# Test bench
def testFFT(points = 64):
  
  # Import modules used for test
  import time
  import matplotlib.pyplot as plt
  
  # Number of data points
  if points & (points - 1) != 0:
    print('Error: Not power of 2')
    exit()
  else:
    N = points

  k = range(N)  

  # Define function to calculate input digital signal
  def calc_f(i, N):
    f = 0.8 * cmath.exp(1 * 2j * cmath.pi * i / N) \
      + 0.6 * cmath.exp(7 * 2j * cmath.pi * i / N + 1j * cmath.pi / 2) \
      + 0.4 * cmath.exp(12 * 2j * cmath.pi * i / N + 1j * cmath.pi / 2) \
      + 0.3 * cmath.exp(17 * 2j * cmath.pi * i / N - 1j * cmath.pi / 6)
    return f
  
  # Prepare input digital signal
  f = [calc_f(i, N) for i in k]
  
  # Perform DFT and measure time
  F = f.copy()
  t0 = time.monotonic()
  DFT(F)
  t1 = time.monotonic()
  print('Time for DFT: {:.2f} ms'.format((t1 - t0) * 1000))

  # Perform FFT and measure time
  F = f.copy()
  t0 = time.monotonic()
  FFT(F)
  t1 = time.monotonic()
  print('Time for FFT: {:.2f} ms'.format((t1 - t0) * 1000))
  
  # Plot input time-domain signal
  #plt.plot([val.real for val in f])
  #plt.plot([val.imag for val in f])
  #plt.show()
  
  # Plot output spectrum
  plt.plot([val.real for val in F])
  plt.plot([val.imag for val in F])
  plt.show()
