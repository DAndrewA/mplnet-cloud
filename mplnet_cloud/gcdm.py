'''Author: Andrew Martin
Creation Date: 10/10/23

Script containing the functions to implement the GCDM (Gradient-based Cloud Detection Method) algorithm outlined in 
Overview of MPLNET Version 3 Cloud Detection; Lewis, Campbell et al (2016)
'''

import numpy as np
from scipy.interpolate import lagrange


def GCDM(NRB, dNRB, beta_m, z, kappa):
    '''Function to implement the GCDM threading.
    
    The steps for the method are as follows:
    
    1. Normalise the NRB by the molecular backscatter to get the scattering ratio, CR`
    
    2. Calculate the first differential of CR` using the locally defined quadratic lagrange polynomials of CR`
    
    3. Determine the noise altitude, z_noise
    
    4. Calculate the algorithm thresholds based on the vertically-integrated mean of CR` below z_noise
    
    5. Determine regions of cloud based on the determined thresholds.
    
    INPUTS:
        NRB : np.ndarray
            (n,m) numpy array containing n profiles of m vertical bins of normalised backscatter data from the MPL. This should be given in SI units of (/m/sr).

        dNRB : np.ndarray
            (n,) broadcastable array containing the mean noise for the NRB for each given vertical profile.

        beta_m : np.ndarray
            (m,) numpy array containing the molecular backscatter data to normalise the NRB by. They should be present on the same height scale that the NRB is provided on.

        z : np.ndarray
            (m,) numpy array containing the heights at which the NRB retrievals are taken.

        kappa : float
            parameter used in determining the gradient threshold

    OUTPUTS:
        cloud_mask : np.ndarray
            (n,m) numpy array containing 1s where clouds are present and 0s where clouds are not present.
    '''

    CR = NRB / beta_m

    # need to ensure differentiation can be done once, rather than once per profile. Will greatly speed up process...
    dCR = np.vectorize(differentiate_lagrange)(z, CR)







def differentiate_lagrange(x,f):
    '''Function to perform a differentiation of the quadratic lagrange polynomial of function f(x).
    
    INPUTS:
        f : np.ndarray
            (m,) values of the function to be differentiated according to the locally defined quadratic lagrange polynomial
            
        x : np.ndarray
            (m,) x values for where the differential of the function is to be evaluated
            
    OUPUTS:
        diff_lag : np.ndarray
            (m-2,) numpy array of the differentiated lagrange-polynomial values
    '''
    diff_lag = np.zeros((f.size - 2))
    for j,(fi,xi) in enumerate(zip(f[1:-1], x[1:-1])):
        i = j+1
        poly = lagrange(x[i-1:i+2], f[i-1:i+2])
        diff_lag[j] = poly.deriv()(xi)
    return diff_lag
