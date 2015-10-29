#include "extensions.h"
#include <stdlib.h>

double distance(double *a, double *b);

double pc_pot_value(double *pos,    // position [Bohr]
		    double *pc_pos, // PC positions [Bohr]
		    double *pc_q,   // PC charges [atomic units]
		    int npc,        // # of PCs
		    double v_max)   //
// loop over all point charges and add their potentials
{
  double V = 0.0;
  double top = 0.0;
  double bottom = 0.0;
  for (int a=0; a < npc; a++) {
    double d = distance(pc_pos + a*3, pos);
    double r = 0.7558904*0.25;
    top = r*r*r*r - d*d*d*d;
    bottom = r*r*r*r*r - d*d*d*d*d;
    V -= pc_q[a] * top / bottom;
  }
  return V;
}

double pc_der_value(double *pos,    // grid point position [Ang]
                    double *pc_pos, // PC position [Ang]
                    double *n_q)   // PC charge
// Find the derivative at a particular point
{
  double d = distance(pc_pos + 0, pos);
  double r = 0.4*0.25;
  double top = 4.0*d*d*d*r*r*r*r*r - 4.0*d*d*d*d*d*d*d*d 
               - 5.0*d*d*d*d*r*r*r*r + 5.0*d*d*d*d*d*d*d*d;
  double bottom =  r*r*r*r*r*r*r*r*r*r - 2.0*r*r*r*r*r*d*d*d*d*d 
                   + d*d*d*d*d*d*d*d*d*d;
  double dV = n_q[0] * top / bottom;
  return dV;
}

PyObject *pc_potential_value(PyObject *self, PyObject *args)
{
  PyArrayObject* posi_c;
  PyArrayObject* pci_nc;
  PyArrayObject* qi_n;
  if (!PyArg_ParseTuple(args, "OOO", &posi_c, &pci_nc, &qi_n))
    return NULL;

  double *pos_c = DOUBLEP(posi_c);
  int npc = PyArray_DIMS(pci_nc)[0];
  double *pc_nc = DOUBLEP(pci_nc);
  double *q_n = DOUBLEP(qi_n);

  return Py_BuildValue("d", pc_pot_value(pos_c, pc_nc, q_n, npc, 1.e+99));
}

PyObject *pc_potential(PyObject *self, PyObject *args)
{
  PyArrayObject* poti;
  PyArrayObject* pci_nc;
  PyArrayObject* beg_c;
  PyArrayObject* end_c;
  PyArrayObject* hh_c;
  PyArrayObject* qi_n;
  if (!PyArg_ParseTuple(args, "OOOOOO", &poti, &pci_nc, &qi_n,
			&beg_c, &end_c, &hh_c))
    return NULL;

  double *pot = DOUBLEP(poti);
  int npc = PyArray_DIMS(pci_nc)[0];
  double *pc_nc = DOUBLEP(pci_nc);
  long *beg = LONGP(beg_c);
  long *end = LONGP(end_c);
  double *h_c = DOUBLEP(hh_c);
  double *q_n = DOUBLEP(qi_n);

  // cutoff to avoid singularities
  // = Coulomb integral over a ball of the same volume
  //   as the volume element of the grid
  double dV = h_c[0] * h_c[1] * h_c[2];
  double v_max = 2.417988 / cbrt(dV);
  //  double v_max = .5;

  int n[3], ij;
  double pos[3];
  for (int c = 0; c < 3; c++) { n[c] = end[c] - beg[c]; }
  // loop over all points
  for (int i = 0; i < n[0]; i++) {
    pos[0] = (beg[0] + i) * h_c[0];
    for (int j = 0; j < n[1]; j++) {
      pos[1] = (beg[1] + j) * h_c[1];
      ij = (i*n[1] + j)*n[2];
      for (int k = 0; k < n[2]; k++) {
	pos[2] = (beg[2] + k) * h_c[2];
	pot[ij + k] = pc_pot_value(pos, pc_nc, q_n, npc, v_max);
      }
    }
  }

  Py_RETURN_NONE;
}

PyObject *pc_der_potential(PyObject *self, PyObject *args)
{
  PyArrayObject* forcei;
  PyArrayObject* densi;
  PyArrayObject* pci_nc;
  PyArrayObject* pci_q;
  PyArrayObject* beg_c;
  PyArrayObject* end_c;
  PyArrayObject* hh_c;
  if (!PyArg_ParseTuple(args, "OOOOOOO", &forcei, &densi, &pci_nc,
                        &pci_q, &beg_c, &end_c, &hh_c))
    return NULL;
  double *force = DOUBLEP(forcei);
  double *dens = DOUBLEP(densi);
  double *pc_nc = DOUBLEP(pci_nc);
  double *pc_q = DOUBLEP(pci_q);
  long *beg = LONGP(beg_c);
  long *end = LONGP(end_c);
  double *h_c = DOUBLEP(hh_c);

  int n[3], ij;
  double pos[3], dis[3], fc;

  double distance(double *a, double *b);

  for (int c = 0; c < 3; c++) { n[c] = end[c] - beg[c]; }
  // Loop over all grid points
  for (int i = 0; i < n[0]; i++) {
    pos[0] = (beg[0] + i) * h_c[0];
    dis[0] = pos[0] - pc_nc[0];
    for (int j = 0; j < n[1]; j++) {
      pos[1] = (beg[1] + j) * h_c[1];
      dis[1] = pos[1] - pc_nc[1];
      ij = (i*n[1] + j)*n[2];
      for (int k = 0; k < n[2]; k++) {
        pos[2] = (beg[2] + k) * h_c[2];
        dis[2] = pos[2] - pc_nc[2];
        double d = distance(pc_nc + 0, pos);
        fc = dens[ij + k] * pc_der_value(pos, pc_nc, pc_q) / d;
        for (int m = 0; m < 3; m++) { force[m] += fc * dis[m]; }
      }
    }
  }

  Py_RETURN_NONE;
}


