'''Author: Andrew Martin
Creation Date: 11/10/23

Scripts to implement the Uncertainty-based Cloud Detection Method of the MPLNET-cloud algorithms, outlined in 
Overview of MPLNET Version 3 Cloud Detection; Lewis, Campbell et al (2016)
'''

import numpy as np

def UCDM(NRB, dNRB, beta_m, dbeta_m, epsilon, z_n, dz_ni, phi, kappa):
    '''Function to implement the threading of the UCDM. This is outlined in the MPLNET paper, which cites Campbell et al. (2008). (CO8)
    
    The algorithm steps will be applied to each vertical profile individually -- much of the logic for the process can't be parallelised.

    The algorithm steps are as follows:
    
    1. Attempt to find a region of clear air within which the profile can be normalised by C*, r_n

    2. Use this C* normalisation to calculate the objective threshold function, alpha

    3. Determine if any cloud bases are detected above r_n according to alpha

    4. If so, recalculate alpha based on the cloud transmission, to estimate the cloud top.

    5. Repeat steps 3 and 4 if further cloud bases are detected.

    6. Remove any false positives.

    INPUTS:
        NRB : np.ndarray
            (n,m)

        dNRB : np.ndarray
            (n,m)

        beta_m : np.ndarray
            (m,)

        dbeta_m : np.ndarray
            (m,)

        epsilon : float

        z_n : np.ndarray
            (n,) array of indices for thw noise altitude

        dz_ni : int
            Number of bins to be checked in the window for the search for the noise altitude

        phi : float
            Value used in determining the threshold criteria for a cloud base.
        
        kappa : float
            Value used to discriminate between valid cloud bases and non-cloud-bases.

    OUTPUTS:

    '''
    (n,m) = NRB.shape
    # according to CO8, the NRB and dNRB values need to be averaged in time.
    # As I will already be passing in time-averaged values, I won't be doing this YET.

    # next step is to normalise the profile to get C* (C_star), and obtain its uncertainty
    C_star = NRB / beta_m
    dC_star = C_star * np.power( np.power(dNRB/NRB, 2) + np.power(dbeta_m/beta_m, 2) , 0.5 )

    # the next step is to determine the range bins where the clear sky calibration can be performed.
    # firstly, for each height index i, calculate N_i, the number of bins above i which should be included in the clean-air search
    N_i = np.ceil(np.power(epsilon,2) * np.power(dC_star/C_star, 2))

    # C_fstar needs to be found profile by profile
    C_fstar = np.zeros((n))
    dC_fstar = np.zeros((n))
    for j, (pC_star, pdC_star, pN_i, pz_ni) in enumerate(zip(C_star, dC_star, N_i, z_n)):
        # start with the check from pz_ni to find C*f, dC*f and z_n
        for tmp in range(3): #perform 3 gradually lower window checks
            pC_fstar, pdC_fstar, pz_n = calculate_C_fstar(pC_star, pdC_star, pN_i, pz_ni, dz_ni)
            
            if pC_fstar is not None:
                C_fstar[j] = pC_fstar
                dC_fstar[j] = pdC_fstar
                z_n[j] = pz_n
                break
            pz_ni += -dz_ni
        
        # if we reach here and no clean-air region has been found, set C_fstar to nan, indicating that no clean air region is present and the GCDM should be used
        if pC_fstar is None:
            C_fstar[j] = np.nan
            dC_fstar[j] = np.nan

    # now that C_fstar values have been found, we can calculate the Pseudo-Attenuated Backscatter, PAB, and then the objective threshold function, alpha
    PAB = NRB / C_fstar
    dPAB = PAB* np.power( np.power(dNRB/NRB, 2) + np.power(dC_fstar/C_fstar, 2) , 0.5 )

    alpha = beta_m * (1 + np.power( np.power( dNRB / beta_m / C_fstar , 2) + np.power(dC_fstar/C_fstar, 2) , 0.5 ))

    # calculate the number of bins that require checking to determine if a bin is a valid cloud base
    Y_i = np.ceil(np.power(phi,2) * np.power(dPAB/PAB, 2))
    # determine possible bins that satisfy the valid-base-candidate condition: above the noise altitude index AND PAB sufficiently above the objective threshold.
    possible_bases = np.logical_and( (PAB-dPAB > alpha), np.arange(m).reshape(1,m) > z_n)

    cloud_mask = np.zeros_like(NRB, dtype=bool)
    # for each profile, perform the itterative process of determining base heights, and cloud top heights
    for j, (bases, pYi, pPAB, pdPAB) in enumerate(zip(possible_bases, pYi, PAB, dPAB)):
        cloud_mask[j,:] = determine_cloud_boundaries(bases, pYi, pPAB, pdPAB, phi, kappa)




def calculate_C_fstar(C_star, dC_star, N_i, z_ni, dz_ni):
    '''Function to calculate the value of C_fstar and dC_fstar given the values for C_star, dC_star, N_i.
    
    The implementation of this function allows for a 2km window to be scanned through from an initial z_n of 5km, then for that to be reduced to 3km and 1km if those loops fail. This is in accordance with the MPLNETV3 code.

    INPUTS:
        C_star : np.ndarray
            (m,) profile of the C_star values

        dC_star : np.ndarray
            (m,) profile of the uncertainty associated with C_star

        N_i : np.ndarray
            (m,) numpy array for the profile of how many additional bins will be checked in the clean-air search

        z_ni : int
            Index for the initial height to check the clear-air regions at.

        dz_ni : int
            Number of bins corresponding to 2km, the number of bins above z_ni within which normalisations will be attempeted.

    OUTPUTS:
        C_fstar : float, None
            Value of C*f to use in calculating the PAB. If no clean-air region is found, this will return None.
            
        dC_fstar : float, None
            Uncertainty of C*f for use in calculating dPAB.

        z_n : int, None
            index of the start of the clean-air calibration zone used
    '''
    # range to be checked
    range = slice(z_ni, z_ni + dz_ni + 1)
    # initialise the outputs as None, to simplify process if no suitable regions are found.
    C_fstar = None
    dC_fstar = None
    z_n = None
    
    for z_i, val_C_star, val_dC_star, val_N_i in zip(z_ni[range], C_star[range], dC_star[range], N_i[range]):
        range_clean_air = slice(z_i, z_i+val_N_i+1)
        
        con1 = np.all( (C_star+dC_star)[range_clean_air] >= (val_C_star-val_dC_star) )
        
        con2 = np.all( (C_star-dC_star)[range_clean_air] <= (val_C_star+val_dC_star) )

        if con1 and con2:
            # once both conditions have been met for the first time, set the outputs and break from the loop
            C_fstar = np.sum(C_star[range_clean_air]) / (val_N_i+1)
            dC_fstar = np.power( np.sum(np.power(dC_star[range_clean_air], 2)) , 0.5) / (val_N_i+1)
            z_n = z_i
            break
    # if no valid clean-air region was found, the variables will return None
    return C_fstar, dC_fstar, z_n

    
def determine_cloud_boundaries(bases, Yi, PAB, dPAB, phi, kappa):
    '''Funciton to determine the cloud boundaries and extent for a single vertical profile.
    
    The steps are as follows: for each valid base in the vertical column:
        + Calculate the phi criterion for Yi bins that satisfy the valid cloud base criteria
        + Calculate the kappa criterion for Yi bins that don't satisfy the valid cloud criteria.
        
        If a bin meets both of the criteria, it is a valid cloud base. Continue vertically until either the kappa or phi criteria fail. That is the cloud top, continue from here upwards.

    NOTE: no attempt is currently made to correct for the attenuation within the cloud...
    
    INPUTS:
        bases : np.ndarray (dtype=bool)
            (m,) numpy array containing 1s where valid cloud base candidates exist.

        Yi : np.ndarray (dtype=int)
            (m,) numpy array containing the number of bins above the current index that need to be checked to satisfy the phi and kappa criteria

        PAB : np.ndarray
            (m,) numpy array containing the Pseudo Attenuated Backscatter data that is used to determine cloud presence

        dPAB : np.ndarray
            (m,) numpy array containing the uncertainties on PAB values.

        phi : float
            Threshold parameter used in determining if a cloud base candidate is an actual cloud base.

        kappa : float
            Parameter used to discriminate against cloud-base candidates that are near too much clear sky.
    
    OUTPUTS:
        cm : np.ndarray (dtype=bool)
            (m,) boolean numpy array containing 1s where detected clouds are present and 0s where clouds are not present.
    '''
    # from bottom to top, run up the profile and apply the relevant criteria
    m = PAB.size
    j = 0
    cm = np.zeros((m))

    while j < m:
        b = bases[j]
        # if we're not checking a valid base, continue up the profile
        if not b:
            j += 1
            continue
        
        # initialise loop to iterate through possible clouds, allowing full extent to be calculated
        inCloud = True
        y = Yi[j] # initial y, which will be incremented on each pass of the loop that succeeds.
        while inCloud and j+y+1 <= m:
            inCloud = False

            range = slice(j, np.min(j+y+1, m))
            mask = bases[range]
            pab = PAB[range]
            dpab = dPAB[range]

            # using the reduced pab and dpab data, calculate the conditions required for a valid cloud base. In this instance, (con_phi and con_kappa) == 1 results in a valid cloud base.
            con_phi = np.sum(pab[mask]) / np.power(np.sum(np.power(dpab[mask], 2)), 0.5) >= phi

            con_kappa = np.sum(pab[~mask]) / np.power(np.sum(np.power(dpab[~mask], 2)), 0.5) < kappa

            if con_kappa and con_phi:
                inCloud = True
                cm[range] = 1
            y+=1

        j += y
    return cm

