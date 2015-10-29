import ase.units as unit
import numpy as np
from math import pi

from gpaw.lcao.tools import world, rank, MASTER

# Point charges and LJ terms must be set beforehand.
k_c = 332.1 * unit.kcal / unit.mol

class calc_qmmm:
    def __init__(self, mm_subsystem=None, qm_subsystem=None, 
                 density=None, index=None, mp = None,
                 comp_charge = None, origin = None,
                 LJ_mm = None, LJ_qm = None):
        """ Need to feed it the mm_subsystem (coordinates and charges),
            the qm_subsystem (coordinates), the all electron density,
            and the index of the mm system. Everything has been shifted.
            """
        self.energy = None
        self.forces = None
        self.mm = mm_subsystem
        self.qm = qm_subsystem
        self.density = density # Bottleneck atm
        self.index = index
        self.mp = mp
        self.comp = comp_charge
        self.origin = origin
        self.LJ_mm = LJ_mm
        self.LJ_qm = LJ_qm

        """ The E_qmmm is:

            E_qmmm = sum_i q_i int_n n(r)/(r-r_i)dr + 
                     sum_i sum_a q_i Z_a / (r_a - r_i) + (LJ)

            (LJ) = 4 * epsilon * (sigma**12 / R**12 - sigma**6 / R**6) 
             
            Hence we split it into two parts... One which deals with 
            the point charge to electron density (V_ext in GPAW), 
            and the other which is basically a TIP3P type potential .
            If pseudo density interaction compensation charges must
            follow, else only nuclei charges on MM forces are required.

            As of now: Only compensation charge to MM coulomb energy added. LJ
            QM to MM energy and forces.

        """
        
    def calculate_1(self):
        # q_i*Z_a/(r_a-r_i) and LJ
        mm = self.mm
        qm = self.qm

        LJ_mm = self.LJ_mm
        LJ_qm = self.LJ_qm

        if LJ_qm == None:
            LJ_mm = np.zeros((2,len(mm)))
            LJ_qm = np.zeros((2,len(qm)))
        """ The combination rules for the LJ terms follow Waldman-Hagler:
            J. Comp. Chem. 14, 1077 (1993)
        """

        mm_c = self.mm.get_initial_charges()
        index = self.index
        for i in range(len(mm)):
            for j in range(len(qm)):
                D = qm[j].position - mm[i].position
                d = (D**2).sum()

                if LJ_mm[1,i] * LJ_qm[1,j] == 0:
                    epsilon = 0
                else:
                    epsilon = 2 * LJ_mm[1,i]**3 * LJ_qm[1,j]**3 \
                              * np.sqrt(LJ_mm[0,i] * LJ_qm[0,j]) \
                              / (LJ_mm[1,i]**6 + LJ_qm[1,j]**6)

                sigma = ((LJ_mm[1,i]**6 + LJ_qm[1,j]**6) / 2)**(1./6)

                self.energy  += (mm_c[i] * self.comp[j] * k_c / d**0.5 + \
                                4 * epsilon * (sigma**12 / d**6 - sigma**6 / d**3))     

                Forces  = (mm_c[i] * self.comp[j] * k_c / d**0.5 + \
                           4 * epsilon * (12 * sigma**12 / d**6 - \
                           6 * sigma**6 / d**3)) * D / d

                self.forces[index+i,:] -= Forces
                self.forces[j,:] += Forces          


    def update(self, atoms):
        index = self.index
        mm = self.mm
        qm = self.qm
        origin = self.origin
        if self.energy is None:
            self.calculate(atoms)
        elif ((mm.positions + origin != atoms[index:].get_positions()).any() or
            (qm.positions + origin != atoms[:index].get_positions()).any()):
            self.calculate(atoms)

    def calculate(self, atoms):
        self.energy = 0
        self.forces = np.zeros((len(self.mm)+len(self.qm),3))
        self.calculate_1()

    def get_energy_and_forces(self, atoms):
        self.update(atoms)
        return self.energy, self.forces


