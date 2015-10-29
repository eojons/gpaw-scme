""" A general classical interpotential class. 

    This class describes intermolecular energy and forces 
    between point charges. 

    Inputs are:
    Lennard-Jones parameters : 
                           Arrays of shape (2,len(atoms))
                           where (0,:) are epsilon values
                           and (1,:) are sigma values.

    index of the no. atoms per classical molecule:
    ensures that atoms only see complete molecules, 
    given any cut-off or pbc.

    type of periodic boundary conditions:
                 'Wolf' : a ewald summation type method
                          Should account nicely for both
                          short and long range electrost. 

                 'MIC'  : minimum image convention. The 
                          larger the MM bulk the better.

                 None

    This calculator is attached to an atoms object were the 
    point charges are set beforehand e.g.:

    atoms.set_charges([.2,-.2,...]) 

    it is also important to account for which directions are
    periodic via. atoms.set_pbc((x,y,z)). Default is 1,1,1 if 
    'MIC' or 'Wolf'.

    Does not support mixed systems atm.

"""

import ase.units as unit
import numpy as np

from math import pi, exp
from scipy.special import erf

def erfc(x):
    v = 1 - erf(x)
    return v

# Electrostatic constant in [au]
k_c = 332.1 * unit.kcal / unit.mol

class Inter_Pot:

    def __init__(self, LJ = None, noa = 1, sys_pbc = None, fix_pbc = None):
        self.energy = None
        self.forces = None
        self.LJ = LJ           # Lennard-Jones parameters
        self.noa = noa         # No. of atoms per molecule
        self.sys_pbc = sys_pbc # PBC method
        
    def calculate(self, atoms):
        """ Basically two pbc schemes. Either a direct coulomb
            sum over the point charges, or the Wolf summation.
        
            Separate calculator for the two schemes. Calc 1 
            takes care of pbc = 'MIC' or None.

        """
        self.atoms = atoms
        self.positions = atoms.positions
        LJ = self.LJ

        """ Combination of LJ terms follows Waldman-Hagler:
                             J. Comp. Chem. 14, 1077 (1993)
        """

        # If LJ is None we still pass forth an empty array
        if LJ is None:
            LJ = np.zeros((2,noa))
        
        # Expand LJ to encompass whole MM array
        LJ = np.tile(LJ, (1, len(atoms)/self.noa))     
        
        self.energy = 0

        self.forces = np.zeros((len(self.atoms), 3))

        if self.sys_pbc == 2 or self.sys_pbc is None:
            self.calculate_1(LJ)

        elif self.sys_pbc == 1:
            self.calculate_2(LJ)

    def calculate_1(self, LJ):
        pbc = self.atoms.pbc

        # Make sure the sys_pbc and atoms.pbc make sense
        if self.sys_pbc == 2 and pbc.sum() == 0:
            pbc = (1,1,1)

        elif self.sys_pbc is None and pbc.sum() > 0:
            if fix_pbc is True:
                pbc = (1,1,1)
            else:
                pbc = (0,0,0)
  
        # Track number of molecules
        N = self.noa
        natoms = len(self.atoms)
        nmol = natoms / N

        C = self.atoms.cell.diagonal()

        print pbc

        # MISSING ASSERTIONS/WARNINGS

        # Get dx, dy, dz from first atom of each molecule to same atom
        # of all other molecules, find minimum distance given the pbc.
        for a in range(nmol - 1):
            D = self.positions[(a+1)*N::N] - self.positions[a*N]
            n = np.rint(D / C) * pbc
            q_v = self.atoms[(a+1)*N:].get_charges()

            # Depleting list of possible interactions, how 
            # molecule N sees all other N - 1 molecules etc.
            position_list = np.zeros(((nmol-1-a)*N,3))

            for j in range(N):
                position_list[j::N] += \
                                     self.positions[(a+1)*N+j::N] - n*C

            self.energy_and_forces(a, position_list, q_v, nmol, LJ)

    def energy_and_forces(self, a, position_list, q_v, nmol, LJ):
        N = self.noa
        
        # Get E and F for each atom of molecule "a" seeing all others
        for i in range(N):
            D = position_list - self.positions[a*N+i]
            d = (D**2).sum(axis=1)

            A = 1e-9  # Avoid singul. in division (see below)

            # Create arrays to hold on to epsilons and sigmas
            # particular to atom i of molecule a
            epsilon = np.zeros(N)
            sigma = np.zeros(N)

            for j in range(N):
                epsilon[j] = 2 * LJ[1,i]**3 * LJ[1,j]**3 \
                             * np.sqrt(LJ[0,i] * LJ[1,j]) \
                             / (LJ[1,i]**6 + LJ[1,j]**6 + A)
                
                sigma[j] = ((LJ[1,i]**6 + LJ[1,j]**6) / 2)**(1./6)

            # Create same length arrays
            epsilon_list = np.tile(epsilon, (nmol-1-a))
            sigma_list = np.tile(sigma, (nmol-1-a))

            self.energy += (k_c * q_v[i] * q_v / d**0.5 + 4 \
                            * epsilon_list * (sigma_list**12 / d**6 - \
                            sigma_list**6 / d**3)).sum() 

            F = ((k_c * q_v[i] * q_v / d**0.5 + 4 * epsilon_list * \
                 (12 * sigma_list**12 / d**6 - 6 * sigma_list**6 \
                 / d**3)) / d)[:, np.newaxis] * D

            self.forces[a*N+i] -= F.sum(axis=0)
            self.forces[(a+1)*N:] += F

    def calculate_2(self, LJ):
        """ sys_pbc == 'Wolf'

            Decomposed short and long range electrostatics,
            with a cut-off at 9.0 [Ang] (found to be A-OK 
            for water, a aprotic highly polar solvent).

            see: J. Phys. Chem. 106, 10725 (2005)
                 Mol. Sim. 31(11), 739, (2005)

            Terms: 
            E_mm = 1 / r_eff - 1 / r - 1 / Rc_eff + 1 / Rc
                   + erfc(a*r) / r - erfc(a*Rc) / Rc + E_self

                 = 0 if r > Rc

            r_eff gives the point charges a Gaussian spread
                 = (y**4 - r**4) / (y**5 - r**5), y = 0.1 [Ang]

            Limits at Rc truncate potential and forces to zero

            Rc_eff is : r_eff as r --> Rc

            Lennard-Jones terms have very short range, hence given 
            a value everywhere.

            NOT IMPLEMENTED:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            Furthermore, since simple solvent molecules are charge
            neutral, and the fact that the code does molecule - to
            - molecule interactions there is no need for charge
            neutralization (see refs).
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        """

        # The system at the origin (0,0,0)        
        mm    = self.atoms
        pbc   = mm.get_pbc()
        mm_c  = mm.get_charges()

        N = self.noa

        # Expand the origin to encompass nearest neighbours
        mme_p, mme_c, LJe, counter = self.get_expansion(mm, LJ, pbc, N)

        # Wolf, smearing and smoothing parameters
        Rc = 9.0     # Potential cut-off [Ang]
        a = 0.22     # Gauss smearing, [Ang**-1]
        y = 0.10     # Smooting parameters, [Ang]

        # Following constants are for the charge neutralization
        # Constants; Energy expression
        c1 = (y**4 - Rc**4) / (y**5 - Rc**5)
        c2 = 1.0 / Rc
        c3 = erfc(a * Rc) / Rc

        # Constants, Force expression
        c1_f = - (4 * Rc**3 * (y**5 - Rc**5) - 5 * Rc**4 *  \
                 (y**4 - Rc**4)) / (y**5 - Rc**5)**2
        c2_f = 1. / Rc**2
        c3_f = erfc(a * Rc) / Rc**2
        c4_f = 2 * a / pi**0.5 * exp(-a**2 * Rc**2) / Rc

        # Self energy
        E_self = k_c * (mm_c**2 * (0.5 * (c1 + c2 + c3) \
                  + a / pi**0.5)).sum()

        # Two parts: One where the system interacts within the origin
        # and the other where it interacts with the surroundings!
        
        # iterate over origin and interact with the whole
        for i in range(len(mm)/N):
            for j in range(N):
                D = mm[i*N+j].position - mme_p[(i+1)*N:len(mm)]
                d = (D**2).sum(axis=1)

                I = d**0.5 <= Rc # Keep track of cut-off
          
                A = 1.0e-9

                # Mixing of LJ terms
                epsilon = 2 * LJ[1,j]**3 * LJe[1,(i+1)*N:len(mm)]**3 \
                      * np.sqrt(LJ[0,j] * LJe[0,(i+1)*N:len(mm)]) \
                      / (LJ[1,j]**6 + LJe[1,(i+1)*N:len(mm)]**6 + A)

                sigma = ((LJ[1,j]**6 + LJe[1,(i+1)*N:len(mm)]**6) / 2)**(1./6)

                self.energy += (mm_c[j] * mme_c[(i+1)*N:len(mm)] * I  * k_c * ((y**4 - d**2) \
                            / (y**5 - d**2.5) - 1. / d**0.5 + \
                            erfc(a * d**0.5) / d**0.5 - c1 + c2 \
                            - c3) + 2 * epsilon * (sigma**12 / \
                            d**6 - sigma**6 / d**3)).sum()

                # Forces: get forces on a single qm nuclei due to mm.
                # The opp. forces on to the mm point charges have to
                # be contracted to the origin.
                top = 4. * (d**4 - y**5 * d**1.5) + 5. * \
                       (y**4 * d**2 - d**4)
                bot = (y**5 - d**2.5)**2

                Forces = (mm_c[j] * mme_c[(i+1)*N:len(mm)] * I * k_c * (top / bot + \
                     1. / d - erfc(a * d**0.5) / d - 2 * a / pi**0.5 \
                     * np.exp(-a**2 * d) / d**0.5 - c1_f - c2_f + c3_f \
                     + c4_f) + 4 * epsilon * (12 * sigma**12 / d**6.5 \
                     - 6 * sigma**6 / d**3.5))[:, np.newaxis] * D


                self.forces[i*N+j] += Forces.sum(axis=0)
                self.forces[(i+1)*N:len(mm)] -= Forces


        # iterate over origin and interact with the whole 
        for i in range(len(mm)):
            D = mm[i].position - mme_p[len(mm):]
            d = (D**2).sum(axis=1)

            I = d**0.5 <= Rc # Keep track of cut-off
          
            A = 1.0e-9

            # Mixing of LJ terms
            epsilon = 2 * LJ[1,i]**3 * LJe[1,len(mm):]**3 \
                      * np.sqrt(LJ[0,i] * LJe[0,len(mm):]) \
                      / (LJ[1,i]**6 + LJe[1,len(mm):]**6 + A)

            sigma = ((LJ[1,i]**6 + LJe[1,len(mm):]**6) / 2)**(1./6)

            self.energy += (mm_c[i] * mme_c[len(mm):] * I  * k_c * ((y**4 - d**2) \
                            / (y**5 - d**2.5) - 1. / d**0.5 + \
                            erfc(a * d**0.5) / d**0.5 - c1 + c2 \
                            - c3) + 2 * epsilon * (sigma**12 / \
                            d**6 - sigma**6 / d**3)).sum()

            # Forces: get forces on a single qm nuclei due to mm.
            # The opp. forces on to the mm point charges have to
            # be contracted to the origin.
            top = 4. * (d**4 - y**5 * d**1.5) + 5. * \
                       (y**4 * d**2 - d**4)
            bot = (y**5 - d**2.5)**2

            Forces = (mm_c[i] * mme_c[len(mm):] * I * k_c * (top / bot + \
                     1. / d - erfc(a * d**0.5) / d - 2 * a / pi**0.5 \
                     * np.exp(-a**2 * d) / d**0.5 - c1_f - c2_f + c3_f \
                     + c4_f) + 4 * epsilon * (12 * sigma**12 / d**6.5 \
                     - 6 * sigma**6 / d**3.5))[:, np.newaxis] * D


            self.forces[i] += Forces.sum(axis=0)
            self.forces[:len(mm)] -= self.get_contraction(Forces,counter)

        self.energy += E_self
        del mme_p, mme_c, LJe

    def get_contraction(self, Forces, counter):
        # Force array is a #counter multiple of the origin
        # Need to add forces to mm[i:], and only pieces which
        # belong to [i:] within each no. of expansions
        no = len(self.atoms)
        new_Forces = Forces[:no]

        for j in range(1, counter-1):
            new_Forces += Forces[no*j:(j+1)*no]

        return new_Forces

    def get_expansion(self, mm, LJ, pbc, noa):
        # Expand position array, charge array, and LJ array
        # to simulate an origin surrounded by nearest neighbours   
        C = mm.cell.diagonal()
        no = len(mm)

        mm_p = mm.get_positions()
        mm_c = mm.get_charges()

        counter = 0

        LJ_new = LJ

        for i in range(3):
            if pbc[i] == True:
                if counter == 0:
                    counter += 3
                elif counter > 0:
                    counter *= 3
                mm_c = np.tile(mm_c, 3)
                mm_p = np.tile(mm_p, (3, 1))
                LJ_new = np.tile(LJ_new, (1, 3))

                cc = counter / 3
                mm_p[no*cc:no*2*cc, i] -= C[i]
                mm_p[no*2*cc:no*3*cc, i] += C[i]

        return mm_p, mm_c, LJ_new, counter

    def get_potential_energy(self, atoms):
        self.update(atoms)
        return self.energy

    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces

    def get_stress(self, atoms):
        raise NotImplementedError

    def update(self, atoms):
        self.calculate(atoms)
        #if self.energy is None:
            #len(self.numbers) != len(atoms) or
            #(self.numbers != atoms.get_atomic_numbers()).any()):
        #    self.calculate(atoms)

        #elif (self.positions != atoms.get_positions()).any(): #or
             # (self.pbc != atoms.get_pbc()).any() or
             # (self.cell != atoms.get_cell()).any()):
        #    self.calculate(atoms)
 
