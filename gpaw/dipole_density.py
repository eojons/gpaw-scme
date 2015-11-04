""" Calculate the forces exerted on the center-of-mass
    of SCME water molecules due to dipole (quadrupole) 
    to electronic density interactions """

import sys

import numpy as np
from ase.atoms import Atoms
from ase.units import Bohr, kcal, mol, Hartree

from math import sqrt, pi
import _gpaw

from gpaw import debug
from gpaw.utilities.timing import Timer

k_c = 332.1 * unit.kcal / unit.mol

class DipoleDensityForces:

    def __init__(self, calc, mm, mp, dipoles, finegrid=False): #quadruoles
        # Need QM density (on grid) and MM object for CM
        self.calc = calc # QM calculator
        self.mm   = mm   # SCME atoms object
        self.mp   = mp   # no. atoms per SCME mol
        #
        # Interacting with
        self.dipoles = dipoles
        # self.qpoles = qpoles
        #
        # LOG
        self.timer = Timer()
        self.out   = sys.stdout
        #
        self.cm = None

    def calculate_forces(self):
        self.out.write('Calculating E-dens to MM Forces')
        self.timer.start('E-dens to MM Forces')
        #
        if self.cm is None:
            self.get_cm()
        # Force array
        n = len(self.atoms) / self.mp
        F = np.zeros((3,n))
        #
        calc = self.calc
        # Grab grid descriptor and density
        if self.finegd:
            gd = calc.density.finegd
            # Get dens on fg
            if calc.density.nt_sg is None:
                calc.density.interpolate_pseudo_density()
            nt_sg = calc.density.nt_sg
            #
        else:
            gd = calc.density.gd
            nt_sg = calc.density.nt_sG
        #
        if calc.density.nspins == 1:
            nt_g = nt_sg[0]
        else:
            nt_g = nt_sg.sum(axis=1)
        #
        sg = (np.indices(gd.n_c, float).T + \
                                  gd.beg_c) / gd.N_c
        #
        for a, pos in enumerate(self.cm):
            # Get all scaled gpt distance to cm
            asg = sg - np.linalg.solve(gd.cell_cv.T, pos)
            # r(xyz) - in Ang
            xyz = np.dot(asg, gd.cell_cv) * Bohr
            #
            dis = np.sqrt(((xyz.T)**2).sum(axis=0))
            # n(r)/d**3
            n_d = nt_sg / dis**3
            # p*r
            pr_r = xyz.T*np.dot(xyz, self.dipole[a])
            # whole term (with electrostatic constant)  
            tot = k_c * n_d * (self.dipole[a] - 3*pr_r.T).T
            #
            # new = tot.reshape((3,tot[1]*tot[2]*tot[3] in shape)) 
            #
            F[:,a] = [tot[0].sum(), tot[1].sum(), tot[2].sum()]

        self.timer.stop('E-dens to MM Forces')
        self.out.write('  E-dens/MM Forces took: %.3f sec'
                       %self.timer.timers[('E-dens to MM Forces',)])
        return F

    def get_cm(self):
        """ Get CM for SCME mols in units [Bohr]

        """
        #
        atoms = self.mm
        mp = self.mp
        n = len(self.atoms) / self.mp
        cm = np.zeros((n,3))
        #
        for i in range(n):
            cm[i,:] += atoms[i*mp:(i+1)*mp].get_center_of_mass() / Bohr
        #
        self.cm = cm.copy()
