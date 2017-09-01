#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import numpy as np


"""
Solve a simple SE problem.
Philip Mocz (2017)
Harvard University

i d_t psi + nabla^2/2 psi -x^2 psi/2 = 0
Domain: [-inf,inf]
Potential: 1/2 x^2
Initial condition: particle in SHO
(hbar = 1, M = 1)

usage: python quantumsph.py
"""

def main():
  """ Main Loop.
  Evolve the time-dependent SE and plot solutions (as *.pdf file)
  """
  
  # Particle in SHO - c.f. Mocz & Succi (2015) Fig. 2
  # parameters
  n = 100               # number of particles
  dt = 0.02             # timestep
  nt = 100              # number of timesteps
  nt_setup = 400        # number of timesteps to set up simulation
  n_out = 25            # plot solution every nout steps
  b = 4                 # velocity damping for acquiring initial condition
  m = 1/n               # mass of SPH particle ( m * n = 1 normalizes |wavefunction|^2 to 1)
  h = 40/n              # smoothing length
  t = 0.                # time

  # plot potential
  xx = np.linspace(-4.0, 4.0, num=400)
  xx = np.reshape(xx,(xx.size,1))
  fig = plt.plot(xx, 0.5*xx**2, linewidth=5, color=[0.7, 0.7, 0.9])
  
  # initialize
  x = np.linspace(-3.0, 3.0, num=n)
  x = np.reshape(x,(n,1))
  u = np.zeros((n,1))
  
  rho = density( x, m, h )
  P = pressure( x, rho, m, h )
  a = acceleration( x, u, m, rho, P, b, h )

  # get v at t=-0.5*dt for the leap frog integrator using Euler's method
  u_mhalf = u - 0.5 * dt * a

  # main loop (time evolution)
  for i in np.arange(-nt_setup, nt):   # negative time (t<0, i<0) is used to set up initial conditions

    # leap frog
    u_phalf = u_mhalf + a*dt
    x = x + u_phalf*dt
    u = 0.5*(u_mhalf+u_phalf)
    u_mhalf = u_phalf
    if (i >= 0):
      t = t + dt
    print("%.2f" % t)
    
    if (i == -1 ):  # switch off damping before t=0
      u = np.zeros((n,1)) + 1.0
      u_mhalf = u
      b = 0  # switch off damping at time t=0
    
    # update densities, pressures, accelerations
    rho = density( x, m, h )
    P = pressure( x, rho, m, h )
    a = acceleration( x, u, m, rho, P, b, h)
 
    # plot solution every n_out steps
    if( (i >= 0) and (i % n_out) == 0 ):
      xx = np.linspace(-4.0, 4.0, num=400)
      xx = np.reshape(xx,(xx.size,1))
      rr = probeDensity(x, m, h, xx)
      rr_exact = 1./np.sqrt(np.pi) * np.exp(-(xx-np.sin(t))**2/2.)**2
      fig = plt.plot(xx, rr_exact, linewidth=2, color=[.6, .6, .6])
      fig = plt.plot(xx, rr, linewidth=2, color=[1.*i/nt, 0, 1.-1.*i/nt], label='$t='+"%.2f" % t +'$')
    # plot the t<0 damping process for fun
    if( i==-nt_setup or i==-nt_setup*3/4 or i==-nt_setup/2 ):
      xx = np.linspace(-4.0, 4.0, num=400)
      xx = np.reshape(xx,(xx.size,1))
      rr = probeDensity(x, m, h, xx)
      fig = plt.plot(xx, rr, linewidth=1, color=[0.9, 0.9, 0.9])
  
  plt.legend()
  plt.xlabel('$x$')
  plt.ylabel('$|\psi|^2$')
  plt.axis([-2, 4, 0, 0.8])
  plt.savefig('solution.pdf', aspect = 'normal', bbox_inches='tight', pad_inches = 0)
  plt.close()
   
   
   
   
def kernel(r, h, deriv):
  """ SPH Gaussian smoothing kernel (1D).
  Input: distance r, scaling length h, derivative order deriv
  Output: weight
  """
  return {
    '0': h**-1 / np.sqrt(np.pi) * np.exp(-r**2/h**2),
    '1': h**-3 / np.sqrt(np.pi) * np.exp(-r**2/h**2) * (-2*r),
    '2': h**-5 / np.sqrt(np.pi) * np.exp(-r**2/h**2) * ( 4*r**2 - 2*h**2),
    '3': h**-7 / np.sqrt(np.pi) * np.exp(-r**2/h**2) * (-8*r**3 + 12*h**2*r)
  }[deriv]
     
   
   
def density(x, m, h):
  """ Compute density at each of the particle locations using smoothing kernel
  Input: positions x, SPH particle mass m, scaling length h
  Output: density
  """
  
  n = x.size
  rho = np.zeros((n,1))
  
  for i in range(0, n):
    # calculate vector between two particles
    uij = x[i] - x
    # calculate contribution due to neighbors
    rho_ij = m*kernel( uij, h, '0' )
    # accumulate contributions to the density
    rho[i] = rho[i] + np.sum(rho_ij)
    
  return rho


 
def pressure(x, rho, m, h):
  """Compute ``pressure'' at each of the particles using smoothing kernel
  P = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
  Input: positions x, densities rho, SPH particle mass m, scaling length h
  Output: pressure
  """
  
  n = x.size
  drho = np.zeros((n,1))
  ddrho = np.zeros((n,1))
  P = np.zeros((n,1))
  
  # add the pairwise contributions to 1st, 2nd derivatives of density
  for i in range(0, n):
    # calculate vector between two particles
    uij = x[i] - x
    # calculate contribution due to neighbors
    drho_ij  = m * kernel( uij, h, '1' )
    ddrho_ij = m * kernel( uij, h, '2' )
    # accumulate contributions to the density
    drho[i]  = np.sum(drho_ij)
    ddrho[i] = np.sum(ddrho_ij)

  # add the pairwise contributions to the quantum pressure
  for i in range(0, n):
    # calculate vector between two particles
    uij = x[i] - x
    # calculate contribution due to neighbors
    P_ij = 0.25 * (drho**2 / rho - ddrho) * m / rho * kernel( uij, h, '0' )
    # accumulate contributions to the pressure
    P[i] = np.sum(P_ij)

  return P
 
 
 
def acceleration( x, u, m, rho, P, b, h):
  """ Calculates acceletaion of each particle due to quantum pressure, harmonic potential, velocity damping
  Input: positions x, velocities u, SPH particle mass m, densities rho, pressure P, damping coeff b, scaling length h
  Output: accelerations
  """
	
  n = x.size
  a = np.zeros((n,1))

  for i in range(0, n):
    
    # damping & harmonic potential (0.5 x^2)
    a[i] = a[i] - u[i]*b - x[i]

    # quantum pressure (pairwise calculation)
    x_js = np.delete(x,i)
    P_js = np.delete(P,i)
    rho_js = np.delete(rho,i)
    # first, calculate vector between two particles
    uij = x[i] - x_js
    # calculate acceleration due to pressure
    fac = -m * (P[i]/rho[i]**2 + P_js/rho_js**2)
    pressure_a = fac * kernel( uij, h, '1' )
    # accumulate contributions to the acceleration
    a[i] = a[i] + np.sum(pressure_a)

  return a



def probeDensity(x, m, h, xx):
  """ Probe the density at arbitrary locations
  Input: positions x, SPH particle mass m, scaling length h, probe locations xx
  Output: density at evenly spaced points
  """	

  nxx  = xx.size
  rr = np.zeros((nxx,1))

  n = x.size

  # add the pairwise contributions to density
  for i in range(0, nxx):
      # calculate vector between two particles
      uij = xx[i] - x
      # calculate contribution due to neighbors
      rho_ij = m * kernel( uij, h, '0' )
      # accumulate contributions to the density
      rr[i] = rr[i] + np.sum(rho_ij)

  return rr




if __name__ == "__main__":
  main()
