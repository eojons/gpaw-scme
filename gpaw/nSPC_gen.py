""" A general classical interpotential class. Inputs are the
    Lennard-Jones parameters and index of the no. atoms per
    classical molecule. This calculator is attached to an
    atoms object where the classical charges are set beforehand.
    (atoms.set_initial_charges(#ARRAY)).

"""

import numpy as np
import ase.units as unit
from scipy.special import erf

# Electrostatic constant:
k_c = 332.1 * unit.kcal / unit.mol

# erfc
def erfc(x):
    return 1 - erf(x)

class InterPot:

    def __init__(self, LJ, index, type=None):

        self.energy = None
        self.forces = None
        self.LJ = LJ
        self.index = index
        self.type = type

        """ Form of the LJ parameters should be : array((2,index))
            where (0,:) are the epsilon values and (1,:) the sigma
            values.

            Type refers to the periodic boundary conditions and
            how they are treated.

            None   --> just 1/r for the object as is, with 
                       minimum image convention wherever
                       pbc   != 0
                          xyz
                       OK for cells with l    > 18.0
                                          xyz
                       if dielectric constant is big.

            'Ew'   --> Ewald summation method, OK for cells
                       with l    > ?
                             xyz 

            'Wolf' --> Cheap? Ewald summation method (only a
                       single nearest neighbor cell is req.)
                       OK for cells with l   > 9.0
                                          xyz

        """

    def calculate(self):
        # Calculates the intermolecular energies 
        print self.positions

#class MIC(InterPot):


