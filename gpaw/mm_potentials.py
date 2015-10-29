import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.units import Bohr, kcal, mol, Hartree

import _gpaw
from gpaw import debug


class DipoleQuad:

    def __init__(self, mm, dipole, quad, mp):
        self.mm     = mm
        self.dipole = dipole # dipole at CM of mol i
        self.quad   = quad   # quad   at CM of mol i
        self.mp     = mp     # no. atoms per CM
        # CM is center of mass.
        self.cm     = None


    def get_potential(self, gd=None):
        """ Create external potential from dipoles
            and quadrupoles with origin at the 
            center of mass of each classical 
            molecule in the atoms object """
        if hasattr(self, 'potential'):
            if gd == self.gd or gd is None:
                # Nothing changed
                return self.potential

        if gd is None:
            gd = self.gd

        # Values are in atomic-units
        dipole = self.dipole / (332.1 * kcal / mol) / 2.5417462310548435
        quad   = self.quad   / (332.1 * kcal / mol) / 2.5417462310548435**2

        # No. solvent mols
        n = len(self.mm) / self.mp

        # Grab center of mass of each molecule
        # Scale to Bohrs
        self.get_cm(n)

        # Make empty POT
        potential = np.zeros(gd.end_c-gd.beg_c)

        sG = (np.indices(gd.n_c, float).T + \
              gd.beg_c) / gd.N_c

        # Dipole is something with Bohr**2

        # Place external potential due to Dipoles and Quads on grid
        # For a given MM mol, get all distances
        for a in range(n):
            nsG = sG - np.linalg.solve(gd.cell_cv.T, self.cm[a])
            # drX, drY, drZ distances:
            xyz = np.dot(nsG, gd.cell_cv)
            # mUr            
            mUr = np.dot(xyz,dipole[a])
            # rQr
            Q = quad[:,:,a]
            # |r - rcm|
            dis = np.sqrt(((xyz.T)**2).sum(axis=0))
            # Add dipole component
            potential += mUr.T / dis**3
            # Quadrupole components:
            for i in range(3):
                for j in range(3):
                    potential += Q[i,j]*xyz.T[i,:,:,:]*xyz.T[j,:,:,:] / dis**5
                    #if i == j:
                    #    potential -= 1./3 * Q[i,j]*xyz.T[i,:,:,:]**2 / dis**5

        # Hold on to
        self.gd = gd
        self.potential = potential

        return self.potential


    def get_nuclear_energy(self, nucleus):
        return -1. * nucleus.setup.Z * self.get_value(spos_c = nucleus.spos_c)


    def get_value(self, position=None, spos_c=None):
        """ Potential value as seen by an electron at a
            certain gridpoint """

        if position is None:
            vr = spos_c * self.gd.h_cv * self.gd.N_c
        else:
            vr = position

        dipole = self.dipole
        quad   = self.quad   

        n = len(self.atoms / self.mp)
        self.get_cm(n,self.mp)

        v = 0

        # Eval. external potential at vr due to Dipoles and Quads
        for a in range(n):
            dr = vr - self.cm[a]
            dis = np.sqrt((dr**2).sum())

            # mUr
            mUr = np.dot(dis,dipole[a])
            Q = quad[:,:,a]

            v += mUr / dis**3

            for i in range(3):
                for j in range(3):
                    v += Q[i,j]*dr[i]*dr[j] / dis**5
                    #if i == j:
                    #   v -= 1./3 * Q[i,j]*dr[i]**2 / dis**5
        return v


    def get_taylor(self, position=None, spos_c=None):
        return [[0]]


    def get_cm(self, n):
        cm   = np.zeros((n,3))
        atoms = self.mm
        mp = self.mp

        for i in range(n):
            cm[i,:] += atoms[i*mp:(i+1)*mp].get_center_of_mass() / Bohr

        self.cm = cm.copy()


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

        # save grid descriptor and potential for future use
        self.potential = potential
        self.gd = gd

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

        # Create a grid having dim gd.end_c (finegd sent)
        potential = np.zeros(end_c-beg_c)
        pos = np.zeros(3)

        h = np.array([h_cv[0,0], h_cv[1,1], h_cv[2,2]])

        # Potential is setup via v_ales, defined in qmmm_potentials
        _gpaw.pc_potential(potential,self.pc_nc,self.charge_n,beg_c,end_c,h)
 
        return potential


class PointCharge(Atom):
    def __init__(self, position, charge):
        Atom.__init__(self, position=position, charge=charge)

