"""	Langevin dynamics with holonomic constraints.
	The constraints are can be divided between two systems 
        where one is for a solvent type part and the other a
        /the subsystem of interest. 
	Bonds are updated iteratively following the RATTLE scheme.

	The constraints need to be solved for both the positions 
	as well as the relevant velocity compononents."""

import sys
import numpy as np
from numpy.random import standard_normal
from gpaw.md import MolecularDynamics

# Random forces need to be distributed among the cpus

if '_gpaw' in sys.modules:
	from gpaw.mpi import world as gpaw_world

else:
	gpaw_world = None

class LangevinC(MolecularDynamics):
	""" Same usage as with the old langevin with one exception.
	### FIX DESCRIPTION #############################################################
	#constraints
	#	if True then the positions and velocities are iteratively updated to
	#	fix the solvent molecules to experimental bond lengths and angles.
        #
	#Constraints are over all water molecules, and are described as three pair (i,j)
	#bond lengths. Two hydrogen to oxygen bonds and a fictious hydrogen-hydrogen bond 
	#which fixes the angle.
        #################################################################################
	The constraint updating follows the RATTLE procedure: 
	H. C. Andersen, J. Comp. Phys. 52, 24 (1983)

	The two step discretized integrator is from: 
	E. V.-Eijnden, and G. Ciccotti, Chem. Phys. Lett. 429, 310 (2006)

	The latter paper presents and proves that the constraint procedure is as accurate as
	the integrator. In this case it is accurate to order 2 in the increment."""

	def __init__(self, atoms, timestep, temperature, friction, fixcm=True, constr=False,
		index=0, sol_index=3, list_sub=None, list_sol=None,  trajectory=None, logfile=None, 
                loginterval=1, communicator=gpaw_world):
	    MolecularDynamics.__init__(self, atoms, timestep, trajectory, logfile,
		loginterval)
	    self.temp = temperature
	    self.frict = friction
	    self.fixcm = fixcm
	    self.constr = constr
	    self.index = index
	    self.list_sub = list_sub
            self.list_sol = list_sol
            self.sol_index = sol_index
	    self.communicator = communicator
	    self.updatevars()

	def set_temperature(self, temperature):
	    self.temp = temperature
	    self.updatevars()

	def set_friction(self, friction):
	    self.frict = friction
	    self.updatevars()

	def set_timestep(self, timestep):
	    self.dt = timestep
            self.updatevars()

	def updatevars(self):
	    """ Gets constant variables and prepares for a time step

	    A single step amounts to:

	    x(n+1) = x(n) + dt*v(n) + A(n)
	    v(n+1) = v(n) + 0.5*dt*(f(x(n+1))+f(x(n))) - dt*y*v(n) + dt**0.5*o*xi(n) + y*A(n)

	    where: 
	    A(n) = 0.5*dt**2(f(x(n))-y*v(n)) + o*dt**3/2(0.5*xi(n)-(2*3**0.5)**-1*eta(n))

	    y is the friction coeff, o(sigma) is (2*kB*T*m_i*y)**1/2

	    xi and eta are the random variables with mean 0 and covariance.

	    However, to allow for the possibility of constraints we rewrite the equations the following way:

	    x(n+1) = x(n) + dt*p(n)
	    v(n+1) = p(n) - 0.5*dt*y*v(n) - y*A(n) - o*dt**0.5*(2*3**0.5)**-1*eta(n) + 0.5*dt*f(n+1) + 0.5*dt**0.5*o*xi(n)

	    where:
	    p(n) = v(n) + A(n)*dt**-1. """
	    dt = self.dt
	    T = self.temp
	    fr = self.frict

	    masses = self.masses
	    sigma = np.sqrt(2*T*fr/masses)
	    #sigma.shape = (-1,1)		
	    c1 = 0.5*dt**2
	    c2 = c1 * fr
	    c3 = sigma*dt*dt**0.5/2.0
	    c4 = sigma*dt*dt**0.5/(2.0*np.sqrt(3))
	    v1 = 0.5*dt
	    v2 = c3/dt
	    v3 = c4/dt

	    # Pass them on
	    self.c1 = c1
	    self.c2 = c2
	    self.c3 = c3
	    self.c4 = c4
	    self.v1 = v1
	    self.v2 = v2
	    self.v3 = v3
	    self.fr = fr

	def step(self, f):
	    index = self.index
	    list_sub = self.list_sub
            list_sol = self.list_sol
	    atoms = self.atoms
	    v = atoms.get_velocities()

            xi = standard_normal(size=(len(atoms), 3))
            eta = standard_normal(size=(len(atoms), 3))
	 
            if self.communicator is not None:
            	 self.communicator.broadcast(xi, 0)
           	 self.communicator.broadcast(eta, 0)

	    # Begin calculating A
	    A = self.c1*f/self.masses - self.c2*v + self.c3*xi + self.c4*eta
	    
	    # Make P and A/dt
	    A2 = A/self.dt
	    P = v + A2

	    # Update the position of the QM/subsystem subunits - Need to run over the list objects
	   
	    for iter in range(500):
	         for n in range(len(list_sub)):
		      for i,j,r in list_sub[n]:
			   old = atoms.get_positions()
			   s = old[i] + self.dt*P[i] - old[j] - self.dt*P[j]
			   ro = old[i] - old[j]
			   m1 = self.masses[i]
			   m2 = self.masses[j]
			   d = r
			   if np.absolute((s**2).sum()-d**2) < 0.002:
			      mu = 0
			   else:
			      mu = ((s**2).sum() - d**2) / (2 * self.dt * np.dot(s,ro) * (m1**-1 + m2**-1))
			   if mu == 0:
			      continue
			   else:
			      P[i] -= mu*ro/m1
			      P[j] += mu*ro/m2

	    # Update the positions of the water molecules
	    bonds = self.list_sol
            nm = self.sol_index # no. atoms per solvent molecule

	    for iter in range(200):
		 for i,j,r in bonds:
		      old = atoms.get_positions()
		      s = old[(i+index)::nm] + self.dt*P[(i+index)::nm] - old[(j+index)::nm] - self.dt*P[(j+index)::nm]
		      ro = old[(i+index)::nm] - old[(j+index)::nm]
		      m1 = self.masses[i+index]
		      m2 = self.masses[j+index]
		      d = r
		      mu = np.zeros(len(s[:,]))
		      for k in range(len(mu)):
			      if np.absolute((s[k]**2).sum() - d**2) < 0.002:
				 continue
			      else:			  
				 mu[k] = ((s[k]**2).sum() - d**2) / (2 * self.dt * np.dot(s[k,:],ro[k,:]) * (m1**-1 + m2**-1))		     
		      for k in range(len(mu)):
			      if mu[k] == 0:
				 continue
			      else:
	              		 P[(i+index)::nm][k] -= (mu[k]*ro[k])/m1
		      		 P[(j+index)::nm][k] += (mu[k]*ro[k])/m2
		      		   
	    # set pos: x^n to x^(n+1)
	    atoms.set_positions(atoms.get_positions() + self.dt*P)

	    # Update the forces
	    f = atoms.get_forces()

	    # Update the velocities
	    P += self.v2*xi + self.v1*f/self.masses - self.fr*A - self.fr*self.v1*v - self.v3*eta
	    atoms.set_velocities(P)

	    # Run through the velocities and determine if the velocity components of each constraint
	    # is within the accepter tolerance... if not update - 
	    # Subsystem 
	    for iter in range(100):
		 for n in range(len(list_sub)):
		      for i,j,r in list_sub[n]:
			   pos = atoms.get_positions()
		 	   old = atoms.get_velocities()
			   dif = old[i] - old[j]
			   ro = pos[i] - pos[j]
			   m1 = self.masses[i]
			   m2 = self.masses[j]
			   d = r
			   if np.absolute(np.dot(dif,ro)) < 0.002:
			      mu = 0
			   else:
			      mu = np.dot(ro,dif/(d**2 * (m1**-1 + m2**-1)))
			   if mu == 0:
			      continue
			   else:
			      old[i] -= mu*ro/m1
			      old[j] += mu*ro/m2
			   atoms.set_velocities(old)
	    atoms.set_velocities(atoms.get_velocities())

	    # solvent
	    for iter in range(200):
		 for i,j,r in bonds:
		      pos = atoms.get_positions()
		      old = atoms.get_velocities()
		      dif = old[(i+index)::nm] - old[(j+index)::nm]
		      ro = pos[(i+index)::nm] - pos[(j+index)::nm]
		      m1 = self.masses[i+index]
		      m2 = self.masses[j+index]
		      d = r
		      mu = np.zeros(len(dif[:,]))
		      for k in range(len(mu)):
			      if np.absolute(np.dot(dif[k],ro[k])) < 0.002:
				 continue
			      else:
				 mu[k]=np.dot(ro[k],dif[k]/(d**2 * (m1**-1+m2**-1)))
		      for k in range(len(mu)):
			      if mu[k] == 0:
				 continue
			      else:
		      		 old[(i+index)::nm][k] -= mu[k]*ro[k]/m1
				 old[(j+index)::nm][k] += mu[k]*ro[k]/m2
		      atoms.set_velocities(old)

	    atoms.set_velocities(atoms.get_velocities())

	    return f

	
