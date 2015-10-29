import numpy as np
from gpaw.calc_qmmm import *
# NEW #
from gpaw.mm_potentials import PointCharges, DipoleQuad

from ase.io import write
from ase.visualize import view

from gpaw.lcao.tools import world, rank, MASTER

class ase_qmmm:
    def __init__(self, atoms, index, calc_1=None, 
                 calc_2=None, rcut=None, mp=3,
                 LJ_qm = None, LJ_mm = None, cell=None,
                 mm_pbc = True, qm_fixed = False,
                 type='cc'):
        self.atoms = atoms 
        self.index = index # Indexes the beginning of the MM subsystem
        self.positions = atoms.get_positions()
        self.calc_1 = calc_1
        self.calc_2 = calc_2
        self.qm = None
        self.mm = None
        self.rcut = rcut

        self.energy = None
        self.forces = None

        self.comp_char = None
        self.mp = mp # Number of atoms in solvent molecules, default 3 (water)
        self.LJ_qm = LJ_qm
        self.LJ_mm = LJ_mm        
        self.cell = cell
        self.mm_pbc = mm_pbc
        self.qm_fixed = qm_fixed

        # NEW #
        ### For 'cc'/'dc':
        self.type = type # Type of electrostatic interactions
                         # If 'cc' - charge, charge (#nSPC)
                         # If 'dc' - dipole/quadurpole to charge (#SCME)
        self.dipoles = None # Hold on to dipoles if 'dc'
        self.quad    = None # Hold on to quadrupoles if 'dc'
        self.density = None # Hold on to density from QM
        self.efield  = None # Hold on to efield from QM


    def get_qm_subsystem(self):
        """ All atoms before index are considered the qm subsystem. 
            Need to create a minimal cell around the qm subsystem and
            keep track of any displacement to fit it neatly in to the
            cell. """
        index = self.index
        pos = self.atoms[:index].get_positions()
        rcut = self.rcut
        
        """ Can either specify a particular cell and keep it fixed, or
            rcut, where the qm system is then placed in a cell defined 
            by min to max + rcut on both sides. 
        """
        
        if self.cell is None: 
            # Find xmin-xmax, ymin-ymax and zmin-zmax: need rcut pos from origin
            # plus xmax, ymax and zmax + rcut from other border.
            C = np.zeros((3,3))
            xmin = pos[:,0].min(); xmax = pos[:,0].max() 
            C[0,0] += xmax - xmin + 2 * rcut
            ymin = pos[:,1].min(); ymax = pos[:,1].max() 
            C[1,1] += ymax - ymin + 2 * rcut
            zmin = pos[:,2].min(); zmax = pos[:,2].max()
            C[2,2] += zmax - zmin + 2 * rcut        
            origin = np.array([xmin, ymin, zmin]) - rcut #(O)
        else:
            C = self.cell
            throw = self.atoms[:index]
            pos_old = self.atoms[0].position
            throw.set_cell(C)
            throw.center()
            pos_new = throw[0].position
            origin = pos_old - pos_new

        if self.qm_fixed:
            origin = 0.0
            C = self.cell

        qm_subsystem = self.atoms[:index]
        pos -= origin
        qm_subsystem.set_positions(pos)
        qm_subsystem.set_cell(C)
        qm_subsystem.set_pbc((0,0,0))

        # Hold on to the shift from origin and qm_subsystem:
        self.origin = origin # (O)
        self.qm = qm_subsystem


    def get_mm_subsystem(self):
        """ All atoms after index are considered as mm subsystem.
            Need to go through get_qm_subsystem to get the origin. """
        if (self.qm is None):
           self.update_qm()
        # Need more !!!
       
        pbc = np.zeros(3)

        if self.mm_pbc is True:
            pbc += 1

        # Only need to shift the positions of the mm subsystem by (O)
        # then find minimum distance as for the minimum image conv.
        index = self.index
        pos = self.atoms[index:].get_positions()
        pos -= self.origin
        mp = self.mp

        # Minimum image relative to the center of the qm cell!
        n = np.zeros(np.shape(pos))
        c_mid = self.qm.cell.diagonal() * 0.5

        n[::mp] = np.rint((c_mid - pos[::mp]) / self.atoms.cell.diagonal())
         
        # Grab all atoms of this particular molecule
        for i in range(1,mp):
            n[i::mp] += n[::mp]

        pos += n * self.atoms.cell.diagonal() * pbc

        mm_subsystem = self.atoms[index:]
        mm_subsystem.set_positions(pos)
        mm_subsystem.set_pbc((1,1,1))
        self.mm = mm_subsystem


    def calculate_mm(self):
        mm = self.mm

        # NEW #
        if self.type == 'dc':
            # Add effects of QM to MM solver
            self.get_electric_field()
            self.calc_1.set(efield=self.efield)

        mm.set_calculator(calc_1)

        self.mm_energy = 0
        self.mm_forces = np.zeros((len(mm),3))
        
        self.mm_energy += mm.get_potential_energy()
        self.mm_forces += mm.get_forces()

        # NEW #
        if self.type == 'dc':
            # Hold on to the induced dipoles and qpoles
            self.dipoles = self.calc_1.dipoles
            self.quad    = self.calc_1.quad

    def get_electric_field(self):
        # If init. check to see if qm
        # has initial charges set.
        mm = self.mm
        mp = self.mp
        qm = self.qm

        dens = self.dens
        cch  = self.comp_charge

        # Empty efield of size (n_mm,3)
        efield = np.zeros((len(mm)/mp,3))

        if dens is None: # Initialization
            if qm.get_initial_charges().sum() =! 0:
                # Grab efield at CM of each MM mol
        else:
            # Use charge density to make Efield

        self.efield = efield


    def update_mm(self):
        self.get_mm_subsystem()

           
    def calculate_qm(self):
        calc_2 = self.calc_2
        qm = self.qm
        mm = self.mm
 
        self.qm_energy = 0
        self.qm_forces = np.zeros((len(qm),3))
        self.mm_forces = np.zeros((len(mm),3))

        # NEW #
        if self.type == 'cc':
            calc_2.set(external=PointCharges(mm))
        elif self.type == 'dc':
            calc_2.set(external=DipoleQuad(mm,
                                           self.dipoles,
                                           self.quad))
        else:
            raise NotImplementedError

        qm.set_calculator(calc_2)

        self.qm_energy += qm.get_potential_energy()
        self.qm_forces += qm.get_forces()
        
        # Hold on to the pseudo core charge
        self.comp_char = calc_2.get_compensation_charges(self.index)

        # NEW # -- Forces on MM due to dens.
        if self.type == 'cc':
            self.mm_forces =\
                calc_2.get_point_charge_forces(mm_subsystem = mm)
        # In case if 'dc' the forces are in there due to Efield


    def update_qm(self):
        self.get_qm_subsystem()


    def calculate_qmmm(self, atoms):

        if self.comp_char is None:
            self.calculate_qm()

        index = self.index
        mm = self.mm
        qm = self.qm
        comp_char = self.comp_char
        LJ_qm = self.LJ_qm
        LJ_mm = self.LJ_mm 
        self.qmmm_energy = 0
        self.qmmm_forces = np.zeros((len(mm)+len(qm),3))

        # NEW #
        self.qmmm_energy, self.qmmm_forces = calc_qmmm(mm_subsystem = mm,
           qm_subsystem = qm, index = index, LJ_mm = LJ_mm, LJ_qm = LJ_qm,
           comp_charge = comp_char, type=self.type, dipoles=self.dipoles,
           quad=self.quad).get_energy_and_forces(atoms)


    def get_energy_and_forces(self, atoms):
        self.positions = atoms.get_positions()
        self.update_qm()
        self.update_mm()

        index = self.index

        self.energy = 0
        self.forces = np.zeros((len(self.mm)+len(self.qm),3))

        # If type == 'dc':
        # Initialize mm with no external component?
        self.calculate_mm()
        self.energy += self.mm_energy
        self.forces[index:,:] += self.mm_forces

        self.calculate_qm()
        self.energy += self.qm_energy
        self.forces[:index,:] += self.qm_forces

        # 
        if self.type == 'cc':
            self.forces[index:,:] += self.mm_forces

        self.calculate_qmmm(atoms)
        self.energy += self.qmmm_energy
        self.forces += self.qmmm_forces


    def get_potential_energy(self, atoms):
        self.update_all(atoms)
        return self.energy


    def get_forces(self, atoms):
        self.update_all(atoms)
        return self.forces


    def get_energy(self, atoms):
        return self.qm_energy


    def update_all(self, atoms):
        if self.energy is None:
            self.get_energy_and_forces(atoms)
        elif (self.positions != atoms.get_positions()).any():
            self.get_energy_and_forces(atoms)

        
    def get_stress(self, atoms):
        raise NotImplementedError


