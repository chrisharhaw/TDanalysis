import pandas as pd
import numpy as np

def catalogue():
    """
    Reads a CSV file containing gravitational lensing data and returns the extracted data as arrays.

    Returns:
    zl (ndarray): Array of lens redshift values.
    zs (ndarray): Array of source redshift values.
    thetaA (ndarray): Array of Image A positions in radians.
    sigmaA (ndarray): Array of Image A position uncertainties in radians.
    thetaB (ndarray): Array of Image B positions in radians.
    sigmaB (ndarray): Array of Image B position uncertainties in radians.
    td (ndarray): Array of time delays in seconds.
    sigmatd(ndarray): Array of time delay uncertainties in seconds.
    
    """
    df = pd.read_csv("TDcatalogue.csv", header=None)
    arr = df.to_numpy()
    zl = arr[:,1]
    zs = arr[:,2]
    thetaA = arr[:,3] * np.pi / (3600.0 * 180.0)
    sigmaA = arr[:,4] * np.pi / (3600.0 * 180.0)
    thetaB = arr[:,5] * np.pi / (3600.0 * 180.0)
    sigmaB = arr[:,6] * np.pi / (3600.0 * 180.0)
    td = arr[:,7] * 60 * 60 * 24
    sigmatd =  arr[:,8] * 60 * 60 * 24
   
    return zl, zs, thetaA, sigmaA, thetaB, sigmaB, td, sigmatd

zl, zs, thetaA, sigmaA, thetaB, sigmaB, td, sigmatd = catalogue()
c = 299792458 