#include "extensions.h"
#include <stdlib.h>

double distance(double *a, double *b);

// double dipole_pot_value

double dipole_der_value(double *pos,   // grid point position [Ang]
                        double *cm,    // CM position of dipole [Ang]
                        double *dis,   // distance between gp and CM [Ang]
                        double *mu)    // dipole value
// Find derivative at particular point
{
  double V[3];
  double d  = distance(cm_pos + 0, pos);
  double b1 = d*d*d;
  double b2 = b1*d;
  for (int c=0; c < 3; c++) {
      V[c] -= mu[c] / b1 - 3.0*mu[c]*dis[c] / d / b2;
  }
  return V;
}

PyObject *dipole_der_potential(PyObject *self, PyObject *args)
{
  PyArrayObject* forcei;
  PyArrayObject* densi;
  PyArrayObject* cmi_nc;
  PyArrayObject* cmi_mu;
  PyArrayObject* beg_c;
  PyArrayObject* end_c;
  PyArrayObject* hh_c;
  if (!PyArg_ParseTuple(args, "0000000", &forcei, &densi, &cmi_nc,
                        &cmi_mu, &beg_c, &end_c, &hh_c))
    return NULL;
  double *force = DOUBLEP(forcei);
  double *dens  = DOUBLEP(densi);
  double *cm_nc = DOUBLEP(cmi_nc);
  double *cm_mu = DOUBLEP(cmi_mu);
  long *beg = LONGP(beg_c);
  long *end = LONGP(end_c);
  double *h_c = DOUBLEP(hh_c);

  int n[3], ij;
  double pos[3], dis[3]

  for (int c = 0; c < 3; c++) { n[c] = end[c] - beg[c]; }
  // Loop over all grid points
  for (int i = 0; i < 0; i++) {
    pos[0] = (beg[0] + i) * h_c[0];
    dis[0] = cm_nc[0] - pos[0];
    for (int j = 0; j < n[1]; j++) {
      pos[1] = (beg[1] + j) * h_c[1];
      dis[1] = cm_nc[1] - pos[1];
      ij = (i*n[1] + j)*n[2];
      for (int k = 0; k < n[2]; k++) {
        pos[2] = (beg[2] + k) * h_c[2];
        dis[2] = cm_nc[2] - pos[2];
        force += dens[ij + k] * dipole_der_value(pos, cm_nc, dis, cm_mu);
      }
    }
  }
  Py_RETURN_NONE;
}
