import os.path #WHY!

import numpy as np

from ase.units import Bohr

import _gpaw

from gpaw import debug
from gpaw.external_potential import ElectrostaticPotential

from gpaw.lcao.tools import rank, MASTER

class PointCharges:
    def __init__(self, atoms, pbc):
        """ Need a list of point charges (atoms) and the periodic
            boundary conditions.

            Keyword sent from ase_qmmm_NEW, allowed are 'Wolf', 'MIC' 
            and None.

            In case of None and 'MIC' no additional expansions are
            needed. The external potential is created using the list
            of atoms as it is (adjusted in ase_qmmm beforehand).

            'Wolf' requires additional expansion. The list of classical
            atoms is translated into 26 boxes around the origin (27 total)
            and the external potential placed on the grid using Wolf
            summations (truncated Ewald summation).

            The same principle is behind forces and nuclei-point charge
            interactions: see calc_qmmm and ase_interface.

        """

        self.pc_nc = None    # List of charge positions sent to c script
        self.charge_n = None # List of charges sent to c script
        self.atoms = atoms   
        self.qmmm_pbc = pbc  # Not to overwrite the qm.pbc

    def get_potential(self, gd=None):
        """ Place the external potential on the grid. 

        """

        if hasattr(self, 'potential'):
            if gd == self.gd or gd is None:
                # Nothing changed
                return self.potential

        if gd is None:
            gd = self.gd

        n = len(self.atoms)
        pc_nc = np.empty((n, 3))
        charge_n = np.empty((n))

        for a, pc in enumerate(self.atoms):
            pc_nc[a] = pc.position / Bohr
            charge_n[a] = pc.charge
        self.pc_nc = pc_nc
        self.charge_n = charge_n

        potential = self.pc_potential(gd.beg_c, gd.end_c, gd.h_cv)
 
        # Hold on to potential and grid descriptor for future use
        self.potential = potential
        self.gd = gd

        # empty point charge lists
        self.pc_nc, self.charge_n = None, None

        return potential

    def get_nuclear_energy(self, nucleus):
        return -1. * nucleus.setup.Z * self.get_value(spos_c = nucleus.spos_c)

    def get_value(self, positions=None, spos_c=None):
        """ The potential value as seen by an electron at a certain grid point.

            positions [Bohr] or
            spos_c : scaled position on the grid.

        """
        if position is None:
            vr = spos_c * self.gd.h_cv * self.gd.N_c
        else:
            vr = position

        if self.pc_nc is None or self.charge_n is None:
            n = len(self.atoms)
            pc_nc = np.empty((n,3))
            charge_n = np.empty((n))
            for a, pc in enumerate(self.atoms):
                pc_nc[a] = pc.positions / Bohr
                charge_n[a] = pc.charge

            self.pc_nc = pc_nc
            self.charge_n = charge_n
            

        if self.qmmm_pbc == 2 or self.qmmm_pbc is None:
            v = _gpaw.pc_potential_value(vr, self.pc_nc, self.charge_n)

        elif self.qmmm_pbc == 1:
            # When placing the potential on the grid the expansion is in
            # wolf.c. Here we translate the classical pcs around the 
            # origin (0,0,0).
            self.pc_nc, self.charge_n = self.get_expansions()
            v = _gpaw.wolf_potential_value(vr, self.pc_nc, self.charge_n)

       
        # Delete arrays (remade) 
        self.pc_nc, self.charge_n = None, None
        return v

    def get_taylor(self, position=None, spos_c=None):
        """ No expansion; the point charges only interact 
            with the pseudo electronic charge (and hence
            compensations charges, pseudo nuclei charges).

        """
        return [[0]]

    def __eq__(self, other):
        """ ASE atoms object does not compare charges. Hence, when
            calling GPAW.set(external=...) two identical PointCharge 
            objects with different charges will not trigger a 
            reinitialization of the Hamiltonian object. 

        """
        try:
            return Atoms.__eq__(self, other) and \
                   np.all(self.get_charges() == other.get_charges())
        except:
            return NotImplemented

    def pc_potential(self, beg_c, end_c, h_cv):
        """ Set the potential up on the grid. Create the empty potential
            grid and pass on to the appropriate c code
            
        """
        # All values in [Bohr]
        potential = np.zeros(end_c - beg_c)
        h = np.array([h_cv[0,0],h_cv[1,1],h_cv[2,2]])
        C = self.atoms.cell.diagonal() / Bohr

        # Only cubic cells work (ATM)

        if self.qmmm_pbc == 2 or self.qmmm_pbc is None:
            _gpaw.pc_potential(potential, self.pc_nc, self.charge_n, \
                               beg_c, end_c, h)

        elif self.qmmm_pbc == 1:
            _gpaw.wolf_potential(potential, self.pc_nc, self.charge_n, \
                                 beg_c, end_c, h, C)
        # Missing PBC input in WOLF schemes!

        return potential

    def get_expansion(self):
        # Expand lists to nearest neighbours
        # ADD ALARM if lx ly or lz are too short (if pbc)
        C = self.atoms.cell.diagonal() / Bohr
        pbc = self.atoms.pbc
        no = len(self.atoms)
        charge_n = self.charge_n
        pc_nc = self.pc_nc

        counter = 0

        # Position list, count no. expansions and expand according
        # to pbc. Each pbc adds 3 to the no. of translations. 
        for i in range(3):
            #if pbc[i] == True:
            if counter == 0:
                counter += 3
            elif counter > 0:
                counter *= 3
            charge_n = np.tile(charge_n, 3)
            pc_nc = np.tile(pc_nc, (3,1)) 
                
            cc = counter / 3
            pc_nc[no*cc:no*2*cc,i] -= C[i]
            pc_nc[no*cc*2:no*3*cc,i] += C[i]

        return pc_nc, charge_n
