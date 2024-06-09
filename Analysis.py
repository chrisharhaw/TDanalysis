import numpy as np
import os
import logging
import sys
import math
from astropy.cosmology import FlatLambdaCDM
import pymultinest

#load the catalogue
from TDcatalogue import catalogue, c

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger()


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
        self.ddt = ddt
        self.sigmaddt = sigmaddt

    def __call__(self, cube, ndim, nparams):
        params = {'Om0': float(cube[0])}
        self.cosmo = FlatLambdaCDM(H0 = 70, **params)
        return self.ln_likelihood_flrw()
    
    def ln_likelihood_flrw(self):
        """
        Returns the log likelihood of the data given the model parameters.
        """
        model_predictions = self.ddt_flrw()
        residuals = self.ddt - model_predictions
        return -0.5 * np.sum((residuals / self.sigmaddt)**2)

    def ddt_flrw(self):
        R = np.empty(len(self.zs))
        for i in range(len(self.zs)):
            Rs = self.cosmo.angular_diameter_distance(self.zs[i]).value
            Rl = self.cosmo.angular_diameter_distance(self.zl[i]).value 
            Rls = self.cosmo.angular_diameter_distance_z1z2(self.zl[i], self.zs[i]).value
            R[i] = Rs * Rl / Rls
        return (1 + self.zl) * R
    

zl, zs, thetaA, sigmaA, thetaB, sigmaB, td, sigmatd = catalogue()

ddt = abs((2 * c * td / (thetaB**2 - thetaA**2))) * 3.2408e-17 / 1e6
sigmaddt = np.zeros(len(zl))
for i in range(len(zl)):
    sigmaddt[i] = ddt[i] * np.sqrt((sigmatd[i] / td[i])**2 + (2 * sigmaA[i] / thetaA[i])**2 + (2 * sigmaB[i] / thetaB[i])**2)


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
folder = '/home/users/cha227/Documents/TDanalysis-main/Bayes/FLRW/'
basename = 'flrw_' + str(nlive) + '_' + str(tol) + '_'

if not os.path.exists(folder):
    os.makedirs(folder)

try:
    pymultinest.run(llikenargs, prior, ndim,
                    outputfiles_basename = folder+basename,
                    multimodal = False,
                    sampling_efficiency = 'model',
                    n_live_points = nlive,
                    evidence_tolerance = tol,
                    const_efficiency_mode = False,
                    n_iter_before_update = 8000,
                    verbose = True)
    
except Exception as e:
    logger.exception("An error occurred")

