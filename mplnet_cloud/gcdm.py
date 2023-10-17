'''Author: Andrew Martin
Creation Date: 10/10/23

Script containing the functions to implement the GCDM (Gradient-based Cloud Detection Method) algorithm outlined in 
Overview of MPLNET Version 3 Cloud Detection; Lewis, Campbell et al (2016)
'''

import numpy as np
from scipy.interpolate import lagrange
from tqdm import tqdm


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
            (n,m) numpy array containing 1s where clouds are present and 0s where clouds are not present. For apparent cloud tops (i.e. at the noise altitude), the value 2 is used
    '''

    CR = NRB / beta_m
    z_noise_ind = calculate_noise_height_index(NRB,dNRB)


    cloud_mask = np.zeros_like(NRB, dtype=int)
    # the following steps can be performed to each vertical profile individually

    dCR = differentiate_lagrange_algebraic(z,CR)
    for pdCR, cm_prof, zni in tqdm(zip(dCR, cloud_mask, z_noise_ind)):
        #dCR = differentiate_lagrange(z, profile)
        # determine the gradient thresholds for cloud presence
        dCR_mean = np.mean(pdCR[:zni - 1]) # zni-1 used because dCR smaller than profile 
        a_max = kappa * dCR_mean
        a_min = dCR_mean - a_max

        # from bottom to top of the profile
        inCloud = False
        cloud_true_top = False
        for i,val in enumerate(pdCR[:zni-1]):
            if not inCloud and val > a_max:
                inCloud = True
                cloud_true_top = False
                cm_prof[i] = 1
            if inCloud:
                cm_prof[i+1] = 1 # accounts for the reduced length of dCR, sets the current bit as 1. Allows for cloud tops to be included...
                if val < a_min:
                    cloud_true_top = True
                if cloud_true_top and val > a_min: 
                    # in this instance we must have reached a cloud top
                    inCloud = False
        # once we've reached the top of the "above-noise" profile if we're still in a cloud it must be an apparent cloud top...
        if inCloud:
            cm_prof[zni-1] = 2
        
    return cloud_mask


def calculate_noise_height_index(NRB, dNRB):
    '''Function to calculate the index of the noise height for each vertical profile. In this case, the noise height is defined as the first vertical bin in which the uncertainty dNRB exceeds half the NRB value.
    
    INPUTS:
        NRB : np.ndarray
            (n,m) the normalised backscatter for which the noise altitude is being found.
            
        dNRB: np.ndarray
            (n,) the uncertainty in the NRB measurements in each of the n vertical profiles.

    OUTPUTS:
        z_noise_ind : np.ndarray
            (n,) indices for the first bin in each vertical profile that is at or exceeds the noise altitude threshold. The actual noise altitude can be obtained by taking z[z_noise_ind].
    '''
    # firstly, calculate where all the pixels where the noise exceeds the threshold
    noise_mask = NRB <= (2*dNRB)
    # by taking the cumulative sum, values below the noise altitude will have 0, whereas those above will be 1 or greater.
    cs_noise_mask = np.cumsum(noise_mask, axis=1)
    z_noise_ind = np.sum( (cs_noise_mask == 0) , axis=1 )
    return z_noise_ind


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


def differentiate_lagrange_algebraic(x,f):
    '''Function to perform a differentiation of the quadratic lagrange polynomial for the function f(x).

    The function is paralelised so it will differentiate w.r.t. the last axis of both x and f
    
    This is derived from the definition of the lagrange polynomial, and is evaluated at the given x values.
    
    INPUTS:
        f : np.ndarray
            (n,m) values of the function to be differentiated according to the locally defined quadratic lagrange polynomial
            
        x : np.ndarray
            (...,m) x values for where the differential of the function is to be evaluated
            
    OUPUTS:
        diff_lag : np.ndarray
            (n,m-2) numpy array of the differentiated lagrange-polynomial values
    '''
    # for the naming convention q0 denotes quantity q at index n-1, q1 denotes quantity q at index n, and q2 denotes quantity q at index n+1
    x0 = x[...,:-2]
    f0 = f[...,:-2]
    x1 = x[...,1:-1]
    f1 = f[...,1:-1]
    x2 = x[...,2:]
    f2 = f[...,2:]

    alpha0 = 1 / (x0 - x1) / (x0 - x2)
    alpha1 = 1 / (x1 - x0) / (x1 - x2)
    alpha2 = 1 / (x2 - x0) / (x2 - x1)

    diff_lag = f0*alpha0*(x1-x2) + f1*alpha1*(2*x1 - (x0+x2)) + f2*alpha2*(x1-x0)
    return diff_lag