import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

    def __call__(self, cube, ndim, nparams):

        cube_aug = np.zeros(ndim)
        self.ipars = list(range(ndim))

        for icube,i in enumerate(self.ipars):
            cube_aug[i] = cube[icube]

        (self.Q) = cube_aug
        return self.ln_likelihood_flrw()
    
    def ln_likelihood_flrw(self):
        """
        Returns the log likelihood of the data given the model parameters.
        """
        model_predictions = self.ddt_flrw()
       

        return ln_likelihood




def ddt():
    return 2*c*td / (thetaB**2 - thetaA**2)

