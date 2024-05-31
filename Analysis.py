import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import pymultinest

#load the catalogue
from TDcatalogue import *



class loglike_flrw(object):
    def __init__(self, zl, zs, thetaA, sigmaA, thetaB, sigmaB, td, sigmatd):
        self.zl = zl
        self.zs = zs
        self.thetaA = thetaA
        self.sigmaA = sigmaA
        self.thetaB = thetaB
        self.sigmaB = sigmaB
        self.td = td
        self.sigmatd = sigmatd
        self.Om0 = None
        self.c = c

    def __call__(self, cube, ndim, nparams):
        params = {'Om0': cube[0]}
        self.cosmo = FlatLambdaCDM(H0 = 70, **params)
        return self.ln_likelihood_flrw()
    
    def ln_likelihood_flrw(self):
        """
        Returns the log likelihood of the data given the model parameters.
        """
        model_predictions = self.ddt_flrw()
        residuals = self.ddt() - model_predictions
        return -0.5 * np.sum(residuals**2 / self.sigmaddt()**2)

    def ddt_flrw(self):
        R = self.cosmo.angular_diameter_distance(self.zs) * self.cosmo.angular_diameter_distance(self.zl)/ self.cosmo.angular_diameter_distance_z1z2(self.zl, self.zs)
        return (1 + self.zl) * R

    def ddt(self):
        return 2 * self.c * self.td / (self.thetaB**2 - self.thetaA**2)
    
    def sigmaddt(self): ##Check this properly!!!!!!
        return self.ddt() * np.sqrt((self.sigmatd / self.td)**2 + (2 * self.sigmaA / self.thetaA)**2 + (2 * self.sigmaB / self.thetaB)**2)
    
ndim = 1
prior_lims = [(0.0, 0.5)]

loglike = loglike_flrw(zl, zs, thetaA, sigmaA, thetaB, sigmaB, td, sigmatd)

def prior(cube, ndim, nparams):
    for i in range(ndim):
        cube[i] = cube[i] * (prior_lims[i][1] - prior_lims[i][0]) + prior_lims[i][0]
    
def llikenargs(cube, ndim, nparams):
    return loglike(cube, ndim, nparams)

tol = 1e-3
nlive = 800
folder = '/Bayes/'
basename = 'flrw_' + str(nlive) + '_' + str(tol) + '_'

pymultinest.run(llikenargs, prior, ndim,
                outputfiles_basename = folder+basename,
                multimodal = False,
                sampling_efficiency = 'model',
                n_live_points = nlive,
                evidence_tolerance = tol,
                const_efficiency_mode = False,
                n_iter_before_update = 8000,
                verbose = True)