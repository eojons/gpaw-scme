import numpy as np
from gpaw.calc_qmmm_wolf import *
from gpaw.point_charges_wolf import *

from gpaw.lcao.tools import world, rank, MASTER

# DELETE AFTER CHECK
from ase.io import write
from ase.visualize import view

class ase_qmmm:
    """ Class to handle a GPAW calculation under the influence of classical
        point charges, and the influence of the resulting QM electronic 
        density on the classical system. Classical-classical interactions 
        included, both intra- and inter- (depends on calc_mm details).
        All in all: a QM/MM calculator
    """

    def __init__(self, atoms, index, calc_mm = None,
                 calc_qm = None, rcut = 7, mm_ind = None,
                 LJ_qm = None, LJ_mm = None, cell = None,
                 qmmm_pbc = None, mm_pbc = np.array([1,1,1]), 
                 n_fix = None, fixqm = False):
        """ There are three options for periodic boundary conditions available:
            qmmm_pbc = 'Wolf', 'MIC' or None. 

            'Wolf':
            is an Ewald summation method, and hence treats the QM and MM 
            periodic boundaries on equal footing. The PBC of the QM system
            can be set to any (n,n,n) where n = 0,1, and the QM/MM and MM
            treatment follows.

            If qmmm_pbc = 'Wolf' but qm.pbc = (0,0,0) then by default qm.pbc is
            set to (1,1,1). The QM, QM/MM and MM cells must be equal along the 
            periodic directions.

            HENCE the pbc directions in QM, QM/MM and MM will follow the 
            total_sys.pbc set in ASE beforehand.

            'MIC': 
            is the minimum image convention. The idea is that the QM system 
            is embedded in a very large (cubic, oblong) cluster of classical 
            molecules, completely decoupled from other QM images (no pbc). 
            The classical molecules then only interact among each other through
            minimum distance between cells.

            Recommended that the total QM/MM cell is at least 20 Ang in pbc=1. The QM  #FIX TEXT
            part in the system is then given a minimal cell with rcut, or the cell 
            is given beforehand (and fixed) with qm_cell. The QM system does not 
            have to be in the center of the QM/MM cell, it is translated to the MM
            center in any case.

            The default for rcut is 7 [Ang] (minimal for grid based calculations),
            but should be set by hand just above basis set cut off in LCAO mode.

            ALL cases: the QM atoms should be indexed in order, and the index 
            variable is to index the first MM atom (every atom after that is MM).

            The classical point charges should be set beforehand, e.g.
            system[index:].set_charges(#ARRAY)
        """
	self.atoms = atoms
        self.index = index     # Indexes the first MM atom
        self.positions = atoms.get_positions() # Need this for later?
        self.calc_mm = calc_mm # A general point charge and LJ interaction script is
                               # available (calc_mm_general).
        self.calc_qm = calc_qm # GPAW calculator object
        self.qm = None
        self.mm = None 
        self.rcut = rcut       # Determines the minimal QM cell (if 'MIC')
        self.density = None    # If density is fixed in some interval
        self.n_fix = n_fix     # No. steps (e.g. for MD) where the same qm density is
                               # used. After the N steps the qm density is once again
                               # updated. Great for thermadynamics stuff.
        self.dummy = None      # Dummy variable to holf on to steps within n_fix region
	self.energy = None     # E_tot (total energy): E_qm + E_qmmm + E_mm
        self.forces = None     # f_a (total forces: f_qmmm + f_qm + f_mm)
        self.comp_char = None  # Compensation charge (or pseudo core charge) for QM/MM
                               # interactions. Coulomb between MM and QM
        self.mm_ind = mm_ind   # No. of atoms in each classical molecule
        self.LJ_qm = LJ_qm 
        self.LJ_mm = LJ_mm 
        """ Lennard-Jones parameters: Form of the arrays should be array((2,:index))
            where :index runs over all atoms belonging to the qm part (and index: 
            the mm part). (0,:) are the epsilon values and (1,:) are the sigma values. 
            The qm and mm parameters are mixed according to: 
            Waldman-Hagler, J. Comp. Chem. 14, 1077 (1993)
	""" 
        self.cell = cell      # Determines fixed QM cell (if 'MIC')
        self.qmmm_pbc = qmmm_pbc # Determines pbc: 'Wolf', 'MIC or None allowed!
                                 # ADD ERROR MSG IF PBC IS NOT SET PROPERLY!!! (NO DEFAULT)
        self.mm_pbc = mm_pbc     # Which axis are treated with MIC, default (1,1,1)                        
        self.origin = None
        self.qm_energy = 0    # This variable needs to be held for the n_fix
        self.fixqm = fixqm


    def get_qm_subsystem(self):
	""" All atoms in :index are treated as the qm subsystem. Need to handle two
            MIC cases. Fixed cell or rcut cell. Even if n_fix is on still have to center 
            QM part to MM part as it is updated.

	"""

        index = self.index
        rcut = self.rcut
        pos = self.atoms[:index].get_positions()

        qmmm_pbc = self.qmmm_pbc

        if qmmm_pbc == 1: #WOLF
	    self.qm = self.atoms[:index]

            if self.qm.pbc.sum() == 0: # Makes no sense without any pbc
                self.qm.set_pbc((1,1,1))  

        elif qmmm_pbc is None: # No pbc, no need to shift
            self.qm = self.atoms[:index]
            self.qm.set_pbc((0,0,0)) 

	elif qmmm_pbc == 2: # MIC
	    cell = self.cell

            if cell is None: # Use rcut to fin minimal cell and shift QM system. 
                             # Keep track of shift with origin - pass on to MM.
	        C = np.zeros((3,3))
                xmin = pos[:,0].min(); xmax = pos[:,0].max()
                C[0,0] += xmax - xmin + 2 * rcut
                ymin = pos[:,1].min(); ymax = pos[:,1].max()
                C[1,1] += ymax - ymin + 2 * rcut
                zmin = pos[:,2].min(); zmax = pos[:,2].max()
                C[2,2] += zmax - zmin + 2 * rcut
                origin = np.array([xmin, ymin, zmin]) - rcut

	    else:
	        C = self.cell
                throw = self.atoms[:index]
                pos_old = self.atoms[0].position
                throw.set_cell(C)
                throw.center()
                pos_new = throw[0].position
                origin = pos_old - pos_new
                del throw

            # With a well defined cell the qm subsystem is shifted to the origin
            qm_subsystem = self.atoms[:index]
            pos -= origin
            qm_subsystem.set_positions(pos)
            qm_subsystem.set_cell(C)
            qm_subsystem.set_pbc((0,0,0)) 

            # Hold on to the shift and qm_subsys
            self.origin = origin
            self.qm = qm_subsystem


    def get_mm_subsystem(self):
        """ All atoms in [index:] considered as classical. Everything stays
            more or less the same if qmmm_pbc = 'Wolf' or None. 

        """
        # ADD ERROR MSG IF CHARGES.sum() = 0
        index = self.index
        qm = self.qm
        pbc = qm.get_pbc()

        # Make sure the qm part is defined
	if self.qm is None:
            self.update_qm()

        if self.qmmm_pbc == 1:           
            self.mm = self.atoms[index:]
            # Make sure the MM have the same pbc as the QM part
            self.mm.set_pbc(pbc)
            
        elif self.qmmm_pbc is None:
            self.mm = self.atoms[index:]
            self.mm.set_pbc((0,0,0)) 

        elif self.qmmm_pbc == 2:
            # The QM part has been shifted to origin. Need to move
            # the MM system accordingly, and then arrange it around 
            # the center of the QM system using MIC. This embeds the
            # QM part completely
            pos = self.atoms[index:].get_positions()

            pos -= self.origin

            mm_ind = self.mm_ind
            mm_pbc = self.mm_pbc
            
            # Minimum image conv. relative to the center of the QM cell
            n = np.zeros(np.shape(pos))
            c_mid = self.qm.cell.diagonal() * 0.5
      
            n[::mm_ind] = np.rint((c_mid - pos[::mm_ind]) \
                          / self.atoms.cell.diagonal())

            # Grab all atoms of the classical molecule
            for i in range(1, mm_ind):
                n[i::mm_ind] += n[::mm_ind]

            mmpbc = np.zeros(3)
            for i in range(3):
                if mm_pbc[i] == True:
                    mmpbc[i] += 1

            # Translate molecules to center the QM system along the periodic dir
            pos += n * self.atoms.cell.diagonal() * mmpbc

            mm_subsystem = self.atoms[index:]
            mm_subsystem.set_positions(pos)
            mm_subsystem.set_pbc(mm_pbc)
            self.mm = mm_subsystem


    def calculate_mm(self):
        # Match pbc description # FIX!!!
        # self.calc_mm.set(pbc=self.qmmm_pbc)

        self.mm.set_calculator(self.calc_mm)
        self.mm_energy = 0
        self.mm_forces = np.zeros((len(self.mm),3))

        self.mm_energy += self.mm.get_potential_energy()
        self.mm_forces += self.mm.get_forces()


    def update_mm(self):
        self.get_mm_subsystem()
        

    def calculate_qm(self):
        self.qm_energy = 0
        self.qm_forces = np.zeros((len(self.qm),3))
        self.mm_forces = np.zeros((len(self.mm), 3))

        # If Density is fixed - update dummy variable but re-use density
        if self.dummy == 1 or self.dummy is None:
            # Pass point charge and periodicity information to the external
            # potential class
            self.calc_qm.set(external=PointCharges(self.mm, self.qmmm_pbc))
            self.qm.set_calculator(self.calc_qm)
            self.qm_forces += self.qm.get_forces()

        self.qm_energy += self.qm.get_potential_energy()

        # Grab (pseudo) QM nuclei charge 
        self.comp_char = self.calc_qm.get_compensation_charges(self.index)
        self.mm_forces = self.calc_qm.get_point_charge_forces(
                                                      mm_subsystem = self.mm,
                                                      all_pbc = self.qmmm_pbc) # FIX!!!


    def update_qm(self):
        # if density is fixed, do not ask to update QM
        if self.dummy == 1: # Works even if None and None
            self.get_qm_subsystem()

        elif self.dummy is None:
            self.get_qm_subsystem()


    def calculate_qmmm(self, atoms):
        if self.comp_char is None:
            self.calculate_qm()

        self.qmmm_energy = 0
        self.qmmm_forces = np.zeros((len(atoms),3))

        # Call QM/MM interface object. This is nuclei - point charge interactions
        # and Lennard-Jones interactions. NO induced dipole in mm part.
        self.qmmm_energy, self.qmmm_forces = calc_qmmm(mm_sub = self.mm,
            qm_sub = self.qm, index = self.index, LJ_mm = self.LJ_mm,
            LJ_qm = self.LJ_qm, comp_charge = self.comp_char,
            qmmm_pbc = self.qmmm_pbc).get_energy_and_forces(atoms)


    def get_energy_and_forces(self, atoms):
        """ Calls all routines and collects energy and force terms into E_tot and
            F_i,tot

        """

        # RECORD positions (for update check)
        self.positions = atoms.get_positions()

        # Make sure subsystems are defined
        self.update_qm() 

        self.update_mm()

        index = self.index

        # E_tot, F_tot,i
        self.energy = 0
        self.forces = np.zeros((len(atoms), 3))

        # Classical part: E_MM, F_MM,i
        self.calculate_mm()
        self.energy += self.mm_energy
        self.forces[index:,:] += self.mm_forces

        # QM part, and part of classical: E_QM, F_QM,a, F_MM,i
        # also the electronic energy part due to V_ext: E_QM/MM,ext
        self.calculate_qm()
        self.energy += self.qm_energy
        self.forces[:index,:] += self.qm_forces
        self.forces[index:,:] += self.mm_forces

        # QM/MM nuclei-point charge, and LJ terms: E_QM/MM, F_QM/MM,i,a
        self.calculate_qmmm(atoms)
        self.energy += self.qmmm_energy

        if self.dummy == 1 or self.dummy is None:
            self.forces += self.qmmm_forces

        else:
            self.forces[index:,:] += self.qmmm_forces[index:,:]


    def get_potential_energy(self, atoms):

        self.update_all(atoms)
        return self.energy


    def get_forces(self, atoms):

        self.update_all(atoms)

        if  self.fixqm is True:

            index = self.index
            self.forces[:index] *= 0

        return self.forces

    def get_energy(self, atoms):
   
        return self.energy


    # ADD functionality to get specific parts of the energy
    def update_all(self, atoms):

        if self.n_fix != None:

            if self.dummy == self.n_fix:
                self.dummy = 0

                if self.energy is None:
                    self.get_energy_and_forces(atoms)
          
                elif (self.positions != atoms.get_positions()).any():
                    self.get_energy_and_forces(atoms)

            else:

                if self.dummy is None:
		     self.dummy = 1

                     if self.energy is None:
                         self.get_energy_and_forces(atoms)

                     elif (self.positions != atoms.get_positions()).any():
                         self.get_energy_and_forces(atoms)

                else:
                     self.dummy += 1
                      
                     if self.energy is None:
                         self.get_energy_and_forces(atoms)

                     elif (self.positions != atoms.get_positions()).any():
                         self.get_energy_and_forces(atoms)


        # If n_fix is not none we keep track of self.dummy
        # counting towards the no. steps until n_fix is reached.
        if self.energy is None:
            self.get_energy_and_forces(atoms)
        elif (self.positions != atoms.get_positions()).any():
            self.get_energy_and_forces(atoms)

    def get_stress(self, atoms):

        raise NotImplementedError
