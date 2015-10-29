""" A general classical interpotential class. Input are the
    Lennard-Jones parameters and index of the no. atoms per
    classical molecule. This calculator is attached to an 
    atoms object were the classical charges are set beforehand
    (atoms.set_charges(#ARRAY)). """

import numpy as np
import ase.units as unit

# Electrostatic constant:
k_c = 332.1 * unit.kcal / unit.mol

class Inter_Pot:

    def __init__(self, LJ, index, type=None):
        self.energy = None
        self.forces = None
        self.LJ = LJ
        self.index = index
        
        """ Form of the LJ parameters should be : array((2,index))
            where (0,:) are the epsilon values and (1,:) the sigma 
            values. 

            Type refers to the periodic boundary conditions and
            how they are treated.

            'None' --> just 1 / r for the object as it is
                       with minimum image convention along
                       pbc !=0 
                       OK for cells with l   > 18.0
                                          xyz

            'Wolf' --> Cheap Ewald summation method, OK for
                       cells with l   > 9.0 
                                   xyz

            'Ewald'--> Ewald summation method, OK for all?

            All methods accept pbc(0-1,0-1,0-1) etc.

        """

    def calculate(self, atoms):
        self.atoms = atoms
        self.cell = atoms.get_cell()
        self.pbc = atoms.get_pbc()
        self.numbers = atoms.get_atomic_numbers()
        self.positions = atoms.get_positions()

        N = self.index

        natoms = len(atoms)
        nmol = natoms // self.index

        self.energy = 0.0
        self.forces = np.zeros((natoms,3))
        
        C = self.cell.diagonal()

        # Works for pbc = False or pbc = (1,1,1)
        # Cubic cell only, and if pbc then minimum image convention is used
        # hence a minimal cell is about 18.0**3 Ang
        #if atoms.pbc.all() == True:
        #    assert (C >= 18.0).all() 
            #assert self.pbc.all()
        #assert not (self.cell - np.diag(C)).any()

        Z = self.numbers.reshape((-1,N))
        # Assert all molecules are aranged the same atom by atom        
        #for i in range(1, nmol):
        #    assert (Z[:N,:] == Z[i*N:(i+1)*N,:])

        # Get dx,dy,dz from first atom of each molecule to same atom of all other 
        # and find minimum distance. Everything moves according to this analysis.
        for a in range(nmol-1):
            D = self.positions[(a+1)*N::N] - self.positions[a*N]
            n = np.rint(D / C) * self.pbc
            q_v = self.atoms[(a+1)*N:].get_initial_charges()
            
            # Min. img. position list as seen for molecule !a!
            position_list = np.zeros(((nmol-1-a)*N,3))

            for j in range(N):
                position_list[j::N] += self.positions[(a+1)*N+j::N] - n*C

            self.energy_and_forces(a, position_list, q_v, nmol)

    def energy_and_forces(self, a, position_list, q_v, nmol):
        """ The combination rules for the LJ terms follow Waldman-Hagler:
            J. Comp. Chem. 14, 1077 (1993)
        """
        N = self.index
        LJ = self.LJ

        for i in range(N): 
            D = position_list - self.positions[a*N+i]
            d = (D**2).sum(axis=1)

            # Create arrays to hold on to epsilon and sigma
            epsilon = np.zeros(N)
            sigma = np.zeros(N)

            for j in range(N):
                if self.LJ[1,i] * self.LJ[1,j] == 0:
                    epsilon[j] = 0
                else:
                    epsilon[j] = 2 * LJ[1,i]**3 * LJ[1,j]**3 \
                              * np.sqrt(LJ[0,i] * LJ[0,j]) \
                              / (LJ[1,i]**6 + LJ[1,j]**6)

                sigma[j] = ((LJ[1,i]**6 + LJ[1,j]**6) / 2)**(1./6)

            # Create list of same length as position_list
            epsilon_list = np.tile(epsilon, (nmol-1-a))
            sigma_list = np.tile(sigma, (nmol-1-a))

            self.energy += (k_c * q_v[i] * q_v / d**0.5 + 4 * epsilon_list * \
                           (sigma_list**12 / d**6 - sigma_list**6 / d**3)).sum()

            F = ((k_c * q_v[i] * q_v / d**0.5 + 4 * epsilon_list * \
                 (12 * sigma_list**12 / d**6 - 6 * sigma_list**6 / d**3))\
                 / d)[:, np.newaxis] * D

            self.forces[a*N+i] -= F.sum(axis=0)
            self.forces[(a+1)*N:] += F

    def get_potential_energy(self, atoms):
        self.update(atoms)
        return self.energy

    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces

    def get_stress(self, atoms):
        raise NotImplementedError

    def update(self, atoms):
        if (self.energy is None or
            len(self.numbers) != len(atoms) or
            (self.numbers != atoms.get_atomic_numbers()).any()):
            self.calculate(atoms)

        elif ((self.positions != atoms.get_positions()).any() or
              (self.pbc != atoms.get_pbc()).any() or
              (self.cell != atoms.get_cell()).any()):
            self.calculate(atoms)
