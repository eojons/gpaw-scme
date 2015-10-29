import os.path

import numpy as np

from ase.atom import Atom
from ase.atoms import Atoms
from ase.units import Bohr

import _gpaw
from gpaw import debug
from gpaw.external_potential import ElectrostaticPotential

#from gpaw.qmmm_potentials import *
### KEYWORD missing for specific potential

class PointCharges:
    def __init__(self, atoms):
        self.pc_nc = None
        self.charge_n = None
        self.atoms = atoms
        #self.potential = None

    def get_potential(self, gd=None):
        """Create the Coulomb potential on the grid."""
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

        # save grid descriptor and potential for future use ?
        # self.potential = potential
        # self.gd = gd

        return potential

    def get_nuclear_energy(self, nucleus):
        return -1. * nucleus.setup.Z * self.get_value(spos_c = nucleus.spos_c)

    def get_value(self, position=None, spos_c=None):
        """The potential value (as seen by an electron)
        at a certain grid point.

        position [Angstrom]
        spos_c scaled position on the grid"""
        if position is None:
            vr = spos_c * self.gd.h_cv * self.gd.N_c
        else:
            vr = position

        if self.pc_nc is None or self.charge_n is None:
            n = len(self.atoms)
            pc_nc = np.empty((n, 3))
            charge_n = np.empty((n))
            for a, pc in enumerate(self.atoms):
                pc_nc[a] = pc.position / Bohr 
                charge_n[a] = pc.charge
            self.pc_nc = pc_nc
            self.charge_n = charge_n

        v = _gpaw.pc_potential_value(vr, self.pc_nc, self.charge_n)
        return v

    def get_taylor(self, position=None, spos_c=None):
        """Get the Taylor expansion around a point

        position [Angstrom]
        output [Hartree, Hartree/Bohr]
        """
        #if position is None:
        #    gd = self.gd
        #    pos = spos_c * gd.h_cv * gd.N_c
        #else:
        #    pos = position
        #vr = np.diag(pos)

        #nabla = np.zeros((3))
        #for a, pc in enumerate(self.atoms):
        #    dist = vr - pc.position / Bohr
        #    d2 = np.sum(dist**2)
        #    nabla += dist * (pc.charge / (d2 * np.sqrt(d2)) )

        # ADD: If Full = 1, then monopole expansion
        # If not, return [[0]]    
        # print nabla
        #return [[self.get_value(position = vr)], np.array([
        #         nabla[1], nabla[2], nabla[0]])] # No taylor expansion
        return [[0]]

    def __eq__(self, other):
        """
        ASE atoms object does not compare charges. Hence, when calling
        GPAW.set(external=...) two identical PointCharge object with different
        charges won't trigger a reinitialization of the Hamiltionian object.
        """
        try:
            return Atoms.__eq__(self, other) and \
                   np.all(self.get_charges() == other.get_charges())
        except:
            return NotImplemented
                    
    def pc_potential(self, beg_c, end_c, h_cv):
        """ 
        Set the potential up on the grid. Distances in Bohr. Take the whole
        list of charges, and compute all distances. User specific potential.
        """
        # ADD: Keyword for specific potential!!!
        #mm_c = self.charge_n
        #mm_pos = self.pc_nc  # Position, in Bohr

        #h = h_cv # Grid spacing in Bohr

        # Create a grid having dim gd.end_c (finegd sent)
        potential = np.zeros(end_c-beg_c)
        pos = np.zeros(3)

        h = np.array([h_cv[0,0], h_cv[1,1], h_cv[2,2]])

        # Potential is setup via v_ales, defined in qmmm_potentials
        _gpaw.pc_potential(potential,self.pc_nc,self.charge_n,beg_c,end_c,h)
 
        return potential

#    def pc_potential_value(self, vr):
#        """ 
#        Grab the potential value at a certain grid point r (vr).
#        This is for the taylor expansion around a nuclei i.e.
#        """
#        mm_c = self.charge_n
#        mm_pos = self.pc_nc        
#
#        dis = ((vr - mm_pos)**2).sum(axis=1)**0.5 # in Bohr
#        v = ((-1) * v_ales(dis, mm_c, bohr = True)).sum()
#
#        return v


class PointCharge(Atom):
    def __init__(self, position, charge):
        Atom.__init__(self, position=position, charge=charge)

