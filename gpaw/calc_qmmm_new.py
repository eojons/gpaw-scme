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
                 LJ_mm = None, LJ_qm = None, type = 'cc',
                 self.dipole = None, self.quad = None):
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

        # NEW #
        self.type = type # Are these charge-charge 'cc' or
                         # charge-dipole,quadpole 'dc' interactions
        self.dipole = dipole
        self.quad   = quad

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

                self.energy  += (mm_c[i] * self.comp[j] *\
                                 k_c / d**0.5 + 4 * epsilon *\
                                 (sigma**12 / d**6 - sigma**6 / d**3))

                Forces  = (mm_c[i] * self.comp[j] * k_c / d**0.5 + \
                           4 * epsilon * (12 * sigma**12 / d**6 - \
                           6 * sigma**6 / d**3)) * D / d

                self.forces[index+i,:] -= Forces
                self.forces[j,:] += Forces          


    def calculate_2(self):
        # Dipole to charge interactions - need to evaluate energy and
        # forces on the qm nuclei due to the presence of dipoles and 
        # quadrupoles of the MM part. LJ terms included.

        mm = self.mm
        qm = self.qm

        LJ_mm = self.LJ_mm
        LJ_qm = self.LJ_qm

        dipoles = self.dipoles
        quad    = self.quad

        Za = self.comp_char
        mp = self.mp

        for i in range(len(mm)):

            for j in range(len(qm)):

                cm = mm[i*mp:(i+1)*mp].get_center_of_mass()

                r   = qm[j].position - cm
                rd  = np.sqrt((r**2).sum())
                mUr = dipole[i].dot(r)
                Q   = quad[:,:,i]

                # Make LJ
                epsilon, sigma = make_LJ(LJ_qm[:,i], LJ_mm[:,j])

                # Factors
                c1 = 4. * epsilon

                self.energy += c1 * (sigma**12/rd**12 \
                                   - sigma**6/rd**6) + \
                                     mUr/rd**3 * Za[j]
                Force = c1 * (12. * sigma**12/rd**12 - \
                              6. * sigma**6/rd**6) * r/rd**2 \
                             - Za[j] * (3. * mUr * r / rd**5 \
                                        + mUr / rd**3)

                for k in range(3):
                    for l in range(3):
                        self.energy += Q[k,l]*r[k]*r[l] / rd**5 * Za[j]



    def make_LJ(self, LJ_qm, LJ_mm):
        """ Combination rules of Waldman-Hagler:
            J. Comp. Chem. 14, 1077 (1993) 

        """
        if LJ_mm[1] * LJ_qm[1] == 0:
            epsilon = 0
        else:
            epsilon = 2 * LJ_mm[1]**3 * LJ_qm[1]**3 \
                      * np.sqrt(LJ_mm[0] * LJ_qm[0]) \
                      / (LJ_mm[1]**6 + LJ_qm[1]**6)

        sigma = ((LJ_mm[1]**6 + LJ_qm[1]**6) / 2.0)**(1./6)

        return epsilon, sigma


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
        if self.type == 'cc':
            self.forces = np.zeros((len(self.mm)+len(self.qm),3))
            self.calculate_1()
        elif self.type == 'dc':
            self.forces = np.zeros((len(self.qm),3))
            self.calculate_2()
        else:
            raise NotImplementedError


    def get_energy_and_forces(self, atoms):
        self.update(atoms)
        return self.energy, self.forces


