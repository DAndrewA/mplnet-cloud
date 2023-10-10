'''Author: Andrew Martin
Creation date: 10/10/23

Script containing functions to handle the molecular backscatter and transmission functions required in the MPLNET-cloud algorithm.
'''

import numpy as np

def calculate_rayleigh_backscatter(P, T, wavelength=532e-9):
    '''Function to calculate the rayleigh backscatter coefficient, as given in Resonance Fluorescence Lidar for Measurements of the Middle and Upper Atmosphere (Chu and Papen), eq5.10
    
    INPUTS:
        P : np.ndarray (units=mbar)
            (m,) numpy array containing pressure values at the m specified vertical levels. Given in units of mbar.

        T : np.ndarray (units=K)
            (m,) numpy array containing the temperature values corresponding to the pressure values in P. Given in units of K.

        wavelength : float (units=m)
            Value of the laser wavelength at which the backscatter coefficient is to be calculated. Given in units of meters, the default value is 532 nm.

    OUTPUTS:
        beta_m : np.ndarray (units= /m /sr)
            (m,) numpy array containing the rayleigh backscatter coefficients for the values of T and P provided. In units of per-meter per-steradian. 
    '''
    beta_m = (2.938e-32) / np.power(wavelength, 4.0117) * (P/T)
    return beta_m


def calculate_rayleigh_transmission_squared(P,T, z, wavelength=532e-9):
    '''Function to caluclate the rayleigh two-way transmission by approximating the extinction coefficient as the total rayleigh volume scattering coefficient (i.e. not considering atmospheric absorption).
    
    This is given in Chu and Papen eq.5.9, originally cited in Cerny and Sechrist (1980).
    
    INPUTS:
        P : np.ndarray (units=mbar)
            (m,) numpy array with pressure values provided for m vertical levels. The units of pressure is mbar.
            
        T : np.ndarray (units=K)
            (m,) numpy array containing the temperature values corresponding to the pressure levels. Units are kelvin.

        z : np.ndarray (units=m)
            (m,) numpy array containing the height values at which the pressures and temperatures occur.
            
        wavelength : float (units=m)
            Wavelength of the laser light that is being transmitted. Given in units of meters, the default value is 532 nm.
            
    OUTPUTS:
        trans2 : np.ndarray
            (m,) numpy array containing the transmission values for the height values.
    '''
    # start by calculating the rayleigh "extinction" (absorption) coefficient.
    alpha = (9.807e-23 * 273 / 1013 / np.power(100,4.0117)) / np.power(wavelength, 4.0117) * (P/T)

    print(np.power(wavelength, 4.0117))
    print(P)
    print(T)

    # the next step is to integrate the extinction coefficient up to the desired altitudes, giving us the optical depth, tau. We will designate tau(z=z0)=0
    dz = np.diff([z[0],*z])
    tau = np.cumsum(dz*alpha)

    # use the optical depth to calculate the two-way transmission
    trans2 = np.exp(-2*tau)
    return trans2