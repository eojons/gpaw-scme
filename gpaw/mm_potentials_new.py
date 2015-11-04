import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.units import Bohr, kcal, mol, Hartree
from gpaw.utilities.timing import Timer

from math import sqrt, pi

import _gpaw
from gpaw import debug


class DipoleQuad:

    """ Class to handle external potential due to 
        dipole and quadrupole from SCME type 
        calculator object.

    """

    def __init__(self, mm, qm,  mp, calcmm, dyn=False):

        self.mm     = mm     # mm atoms object
        self.mp     = mp     # no. atoms per center of mass (cm)
        self.dyn    = dyn    # dyn. update of potential?
        self.calcmm = calcmm # mm calculator (SCME)
        self.qm     = qm     # qm atoms object
        #
        self.qmidx  = len(qm)# no. QM atoms
        #
        self.nm     = len(self.mm) / self.mp
        self.cm     = self.get_cm(self.nm)
        self.timer  = Timer()

        # initialization
        self.initial = True


    def get_potential(self, gd=None, density=None, wfs=None):
        """ Create external potential from dipoles
            and quadrupoles with origin at the 
            center of mass of each classical 
            molecule in the atoms object """

        if self.initial:
            self.update_potential(gd=gd, density=density, wfs=wfs)
            return self.potential    
        elif self.dyn:
            self.update_potential(gd=gd, density=density, wfs=wfs)
            return self.potential
        else:
            if hasattr(self, 'potential'):
                if gd == self.gd or gd is None:
                # Nothing changed
                return self.potential


    def update_potential(self, gd=None, density=None, wfs=None):

        if gd is None:
            gd = self.gd

        # Grab electric field and derivative values
        self.timer.start('Electric Field and Derivative')
        eF, deF = self.get_field(density, wfs)
        self.timer.stop('Electric Field and Derivative')

        #######################################
        # Pass (new) values to SCME
        self.calcmm.eF  = eF   # <-- ! CHECK
        self.calcmm.deF = deF
        self.mm.set_calculator(self.calcmm)
        self.timer.start('SCME Calculation')
        self.mm.get_potential_energy()
        self.timer.stop('SCME Calculation')
        #######################################

        # Values are in atomic-units
        dipole = self.calcmm.dipoles \
                 / (332.1 * kcal / mol) / 2.541746 # <-- UNITS

        # Qpole here

        # No. solvent mols
        n = len(self.mm) / self.mp

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
            #
            # |r - rcm|
            dis = np.sqrt(((xyz.T)**2).sum(axis=0))
            # Add dipole component
            potential += mUr.T / dis**3

        # Potential updated
        self.initial = False

        # Hold on to
        self.gd = gd
        self.potential = potential
        self.eF  = eF
        self.deF = deF


    def get_efield(self, density, wfs):
        """ Evaluate electric field at each cm
            from total psuedo charge density. 

        """

        eF  = np.zeros((3,self.nm))
        deF = np.zeros((3,3,self.nm))

        # Grab comp. charges
        comp = np.zeros(self.qmidx)
        density.Q_aL = {}
        for a, D_sp in density.D_asp.items():
            Q_L = density.Q_aL[a] = np.dot(D_sp[:wfs.nspins].sum(0),
                                           wfs.setups[a].Delta_pL)
            Q_L[0] += wfs.setups[a].Delta0
            comp[a] += Q_L[0]

        # Collect over gd domains
        #wfs.gd.comm.sum(comp)
        comp *= -1*sqrt(4.*pi)

        # Grab pseudo-density on gd
        if density.nt_sg is None:
            density.interpolate_pseudo_density()
        nt_sg = density.nt_sg
        #
        if density.nspins == 1:
            nt_g = nt_sG[0]
        else:
            nt_g = nt_sg.sum(axis=0)

        assert np.shape(nt_g) == np.shape(self.gd)
        #
        sG = (np.indices(self.gd.n_c, float).T + \
              self.gd.beg_c) / self.gd.N_c
        # Arrays
        # eFdens = np.zeros_like(eF)
        # eFnucl = np.zeros_like(eF)
        # deFdens = np.zeros_like(deF)
        # deFnucl = np.zeros_like(deF)
        #
        for a, pos in enumerate(self.cm):
            # Get all gpt distances relative to molecule a
            nsG = sG - np.linalg.solve(gd.cell_cv.T, self.cm[a])
            # r(xyz) to all gpts
            xyz = np.dot(nsG, gd.cell_cv)
            # distance to all gpts
            dis = np.sqrt(((xyz.T)**2).sum(axis=0))
            # total field on cm due to density
            eFT = (xyz.T)*nt_g / dis**3
            eF[:,a] += [eFT[0].sum(),eFT[1].sum(),eFT[2].sum()]
            # nuclei-to-dipole
            xyz_n = self.qm.get_positions() - pos
            dis_n = np.sqrt((xyz_n**2).sum(axis=1))
            eF[:,a] += (xyz_n.T * comp  / dis_n).T.sum(axis=0)
            # Loop for deF 
            for n in range(3):
                # ...
                nr_ir = xyz.T[n]*xyz.T*nt_g

        # Collect eF and deF here... then gd.comm.sum(eF)? <-- check!!!

        #nt_g = self.gd.collect(nt_g, True)
        # Collect over gd domains


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

        # Eval. external potential at vr
        for a in range(n):
            dr = vr - self.cm[a]
            dis = np.sqrt((dr**2).sum())

            # mUr
            mUr = np.dot(dis,dipole[a])
            Q = quad[:,:,a]

            v += mUr / dis**3

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

