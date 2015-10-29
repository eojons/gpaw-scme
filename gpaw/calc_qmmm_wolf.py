import ase.units as unit
import numpy as np

from math import pi, exp 
from scipy.special import erf

from gpaw.lcao.tools import world, rank, MASTER

def erfc(x):
    v = 1 - erf(x)
    return v

# Electrostatic constant in [au]
k_c = 332.1 * unit.kcal / unit.mol

class calc_qmmm:
    """ Class to calculate the nuclei to point charge electrostatics and 
        Lennard-Jones btw. a QM and a MM part.

        Given the periodic boundary conditions they either interact via.
        the coulomb potential 1/r or the damped and truncated Wolf
        potential (basic form: erf/x - erfc/x).

        The LJ potential goes quickly to zeros, hence, there is no
        truncation.

        Wolf potential is an approximate Ewald summation scheme, which
        requires at least the nearest neighbours of the 0 cell. If
        cell side lengths are sufficient (lx,lz,ly > 9.0 Ang), only
        a single (27x) expansion of the smaller part is required.

    """
    
    def __init__(self, mm_sub = None, qm_sub = None, 
                 index = None, LJ_mm = None, LJ_qm = None, 
                 qmmm_pbc = None, comp_charge = None):

        self.energy = None
        self.forces = None
        self.mm = mm_sub   # MM part, sent from ase_qmmm, after processing
        self.qm = qm_sub   # QM part, sent from ase_qmmm, after processing
        # processing: the MM system has already been shifted if pbc='MIC'
        # and the QM has been shifted to the origin having a minimal or 
        # some fixed cell. If pbc='Wolf' the only thing that might have 
        # changed is the pbc(x,y,z).
        self.index = index 
        self.LJ_mm = LJ_mm # Lennar-Jones terms passed on from ase_qmmm
        self.LJ_qm = LJ_qm 
        self.qmmm_pbc = qmmm_pbc
        self.comp = comp_charge # 'Pseudo' QM nuclei charge, passed on
        # from the calc_qm object (GPAW) 

    def calculate(self, atoms):
        """ Two possible pbc schemes. Either a direct coulomb sum over
            the point charges and QM nuclei, or the Wolf sum.
        """
 
        # Get LJ terms in order first. If LJ_qm is None we still pass
        # on an empty array to either calculator.
        LJ_mm = self.LJ_mm
        LJ_qm = self.LJ_qm

        if LJ_qm == None:
            LJ_mm = np.zeros((2,len(self.mm)))
            LJ_qm = np.zeros((2,len(self.qm)))

        """ Either 'Wolf' or 'MIC' case the LJ terms are mixed btw. the QM
            and MM part via the Waldman-Hagler combination rule: 
 
            J. Comp. Chem. 14, 1077 (1993)
        """
 
        self.energy = 0
        self.forces = np.zeros((len(self.mm) + len(self.qm), 3))

        if self.qmmm_pbc == 2 or self.qmmm_pbc is None:

            self.calculate_1(LJ_mm, LJ_qm)

        elif self.qmmm_pbc == 1:

            self.calculate_2(LJ_mm, LJ_qm)        

    def calculate_1(self, LJ_mm, LJ_qm):
        """ qmmm_pbc = 'MIC' or None

        """
        mm = self.mm
        qm = self.qm
        mm_c = self.mm.get_charges()
        #qm_c = self.comp get_charges()

        index = self.index

        # Single sum, over QM: whole MM array taken care of
        for j in range(len(qm)):
            D = qm[j].position - mm.positions
            d = (D**2).sum(axis=1)

            A = 1e-9 # Avoid singul. in division (see below)

            # Mixing of LJ terms to make epsilon_ij and sigma_ij
            epsilon = 2 * LJ_mm[1,:]**3 * LJ_qm[1,j]**3 \
                      * np.sqrt(LJ_mm[0,:] * LJ_qm[0,j]) \
                      / (LJ_mm[1,:]**6 + LJ_qm[1,j]**6 + A)

            sigma = ((LJ_mm[1,:]**6 + LJ_qm[1,j]**6) / 2)**(1./6)

            self.energy += (mm_c * qm_c[j] * k_c / d**0.5 + \
                            4 * epsilon * (sigma**12 / d**6 -\
                            sigma**6 / d**3)).sum()

            Force = ((mm_c * qm_c[j] * k_c / d**0.5 + 4 * epsilon \
                    * (12 * sigma**12 / d**6 - 6 * sigma**6 / d**3))\
                    / d)[:, np.newaxis] * D


            self.forces[j] += Force.sum(axis=0)
            self.forces[index:] -= Force

    def calculate_2(self, LJ_mm, LJ_qm):
        """ qmmm_pbc = 'Wolf'

            Composed of long and short range parts, with truncation
            at the cut-off, Rc = 9.0 [Ang] 

            see: J. Phys. Chem. 106, 10725 (2005)
                 Mol. Sim. 31(11), 739 (2005)

            Terms:
            E_qmmm = 1 / r_eff - 1 / r - 1 / Rc_eff + 1 / Rc 
                     + erfc(a*r) / r - erfc (a*Rc) / Rc

                   = 0 if r > Rc

            r_eff ensures this is smooth close to 0:
                  = (y**4 - r**4) / (y**5 - r**5), y = 0.1 [Ang]

            Limits at Rc truncate potential to zero at Rc.

            Forces: derivative and limit assumed to commute.

            Rc_eff is: r_eff as r --> Rc

            a: (alpha in lit.) The Gaussian width parameter
            for point charge smearing. Value: 0.22 [Ang**-1]
 
            Lennard-Jones is a very short lived potential and hence 
            simply given a value everywhere.
 
        """
        qm = self.qm
        pbc = qm.get_pbc()
        qm_c = self.comp

        index = self.index

        ## Check which cluster is smaller!!! ##

        # Expand MM pos, charge and LJ arrays to nearest 
        # neighbours. Track no. expansions (counter).
        mm_p, mm_c, LJ_mm, counter, M = self.get_expansion(LJ_mm, pbc)

        # Wolf, smearing and smoothing parameters
        Rc = 9.0     # Potential cut-off [Ang]
        a = 0.22     # Gauss smearing, [Ang**-1]
        y = 0.10     # Smoothing parameter, [Ang]

        # Constants; Energy expression
        c1 = (y**4 - Rc**4) / (y**5 - Rc**5)
        c2 = 1.0 / Rc
        c3 = erfc(a * Rc) / Rc

        # Constants; Forces expression
        c1_f = - (4 * Rc**3 * (y**5 - Rc**5) - 5 * Rc**4 * \
                  (y**4 - Rc**4)) / (y**5 - Rc**5)**2
        c2_f = 1. / Rc**2
        c3_f = erfc(a * Rc) / Rc**2
        c4_f = 2 * a / pi**0.5 * exp(-a**2 * Rc**2) / Rc

        # Iterate over all QM nuclei and interact with whole
        # classical array
        for j in range(len(qm)):
            D = qm[j].position - mm_p
            d = (D**2).sum(axis=1)

            I = d**0.5 <= Rc  # Keep track of cut-off (which are 0)
            I = I*np.ones(len(I))

            A = 1.0e-9

            # Mixing of LJ terms, epsilon_ij and sigma_ij
            epsilon = 2 * LJ_mm[1,:]**3 * LJ_qm[1,j]**3 \
                      * np.sqrt(LJ_mm[0,:] * LJ_qm[0,j]) \
                      / (LJ_mm[1,:]**6 + LJ_qm[1,j]**6 + A)
     
            sigma = ((LJ_mm[1,:]**6 + LJ_qm[1,j]**6) / 2.)**(1./6)

            Energy = (mm_c * qm_c[j] * I * k_c * ((y**4 - d**2) \
                            / (y**5 - d**2.5) - 1. / d**0.5 + \
                            erfc(a * d**0.5) / d**0.5 - c1 + c2 \
                            - c3) + 4 * I * epsilon * (sigma**12 / \
                            d**6 - sigma**6 / d**3)).sum()
            self.energy += Energy
            # Forces: get forces on a single qm nuclei due to mm.
            # The opp. forces on to the mm point charges have to
            # be contracted to the origin.
            top = 4. * (d**4 - y**5 * d**1.5) + 5. * \
                       (y**4 * d**2 - d**4) 
            bot = (y**5 - d**2.5)**2
             
            Forces = ((mm_c * qm_c[j] * I * k_c * (top / bot + \
                     1 / d - erfc(a * d**0.5) / d - 2 * a / pi**0.5 \
                     * np.exp(- a**2 * d) / d**0.5 - c1_f - c2_f + c3_f \
                     + c4_f) - I * 4 * epsilon * (12 * sigma**12 / d**6.5 \
                     - 6 * sigma**6 / d**3.5))* 1. / d**0.5)[:, np.newaxis] * D

            self.forces[j] -= Forces.sum(axis=0)
            self.forces[index:] += self.get_contraction(Forces, 
                                                        counter, M)

        del mm_p, mm_c, LJ_mm

    def get_expansion(self, LJ_mm, pbc):
        # Expand lists to encompass nearest neighbour cells
        C = self.mm.cell.diagonal() 
        no = len(self.mm)
        mm_p = self.mm.get_positions()
        mm_c = self.mm.get_charges()
        
        M = np.zeros((no, 3))

        counter = 0

        LJ_new = LJ_mm

        for i in range(3):
           if pbc[i] == True:
               if counter == 0:
                   counter += 3 
               elif counter > 0:
                   counter *= 3
               mm_c = np.tile(mm_c, 3)
               mm_p = np.tile(mm_p, (3,1))
               M = np.tile(M, (3, 1))   
               LJ_new = np.tile(LJ_new, (1,3))
               
               cc = counter / 3
               mm_p[no*cc:no*2*cc,i] -= C[i]
               mm_p[no*cc*2:no*3*cc,i] += C[i]
               M[no*cc:no*2*cc,i] += 1
               M[no*cc*2:no*3*cc,i] -= 1
        
        #if rank == MASTER:
        #    print M

        return mm_p, mm_c, LJ_new, counter, M
        
    def get_contraction(self, Forces, counter, M):
        # The Forces array is #counter multiple of the mm system
        # Need to add forces together and contract the array
        # to encompass only the origin... hence, effectively
        # look at the effect of nearest neighbour QM images.
        # The forces on the classical part due to periodic 
        # images of the quantum part can be extracted from 
        # the expansion through the matrix M. It changes sign
        # of the force components along the translation axis
        no = len(self.mm)
        new_Forces = np.zeros((no, 3))
        new_Forces += Forces[:no]

        for j in range(1, counter - 1):
            new_Forces += Forces[j*no:(j+1)*no]  #- \
                          #2 * Forces[j*no:(j+1)*no] * M[j*no:(j+1)*no]
        
        return new_Forces

    def update(self, atoms):
        # Missing check functions - if things have not changed, do not
        # calculate...

        index = self.index
        mm = self.mm
        qm = self.qm

        if self.energy is None:
            self.calculate(atoms)


    def get_energy_and_forces(self, atoms):
        self.update(atoms)
        return self.energy, self.forces 
