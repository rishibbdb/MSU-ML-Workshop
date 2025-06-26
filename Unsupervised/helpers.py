from helper import *
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from astropy.coordinates import Angle, SkyCoord
import astropy.units as u

from sklearn.cluster import DBSCAN, OPTICS,  cluster_optics_dbscan
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import multivariate_normal, uniform
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000



def load_events(array):
    ra = array[0]
    dec = array[1]
    ra_array = []
    dec_array = []
    for i in range(len(ra)):
        ra_array.append(np.rad2deg(ra[i]))

    for i in range(len(dec)):
        dec_array.append(np.rad2deg(dec[i]))
    return ra_array, dec_array

def load_events_w_energy_pandas(input_list):
    ra_list = []
    dec_list = []
    energy_list = []
    sigma_list = []
    xra_list = []
    xdec_list = []
    log_energy_list = []
    for array in input_list:
        ra = array[0]
        dec = array[1]
        logenergy = array[2]
        sigma = array[3]
        ra_array = []
        dec_array = []
        energy_array = []
        sigma_array = []
        xra_array = []
        xdec_array = []
        log_energy_array = []
        for i in range(len(ra)):
            ra_array.append(np.rad2deg(ra[i]))
            dec_array.append(np.rad2deg(dec[i]))
            sigma_array.append(sigma[i])
            sigma_deg = np.degrees(sigma[i])
            xra = sigma_deg / np.cos(np.radians(np.rad2deg(dec[i])))
            xdec = sigma_deg
            xra_array.append(xra)
            xdec_array.append(xdec)
            energy = 10**logenergy[i]
            energy_array.append(energy)
            log_energy_array.append((logenergy[i]))
        ra_list.append(ra_array)
        dec_list.append(dec_array)
        energy_list.append(energy_array)
        sigma_list.append(sigma_array)
        xra_list.append(xra_array)
        xdec_list.append(xdec_array)
        log_energy_list.append(log_energy_array)
        print("Number of events loaded = {}".format(len(ra_array)))
    dataset = pd.DataFrame({'ra': ra_list, 'dec': dec_list, 'energy':energy_list, 'logenergy': log_energy_list, 'sigma': sigma_list, 'xra': xra_list, 'xdec': xdec_list})
    return dataset


def load_events_wo_energy_pandas(input_list):
    ra_list = []
    dec_list = []
    xra_list = []
    xdec_list = []
    for array in input_list:
        ra = array[0]
        dec = array[1]
        ra_array = []
        dec_array = []
        xra_array = []
        xdec_array = []
        for i in range(len(ra)):
            ra_array.append(np.rad2deg(ra[i]))
            dec_array.append(np.rad2deg(dec[i]))
        ra_list.append(ra_array)
        dec_list.append(dec_array)
        print("Number of events loaded = {}".format(len(ra_array)))
    dataset = pd.DataFrame({'ra': ra_list, 'dec': dec_list})
    return dataset
# def load_events_w_energy(input_list):
#     ra_list = []
#     dec_list = []
#     energy_list = []
#     sigma_list = []
#     xra_list = []
#     xdec_list = []
#     for array in input_list:
#         ra = array[0]
#         dec = array[1]
#         energy = array[2]
#         sigma = array[3]
#         ra_array = []
#         dec_array = []
#         energy_array = []
#         sigma_array = []
#         xra_array = []
#         xdec_array = []
#         for i in range(len(ra)):
#             ra_array.append(np.rad2deg(ra[i]))
#             dec_array.append(np.rad2deg(dec[i]))
#             sigma_array.append(sigma[i])
#             energy_array.append(energy[i])
#             sigma_deg = np.degrees(sigma[i])
#             xra = sigma_deg / np.cos(np.radians(np.rad2deg(dec[i])))
#             xdec = sigma_deg
#             xra_array.append(xra)
#             xdec_array.append(xdec)
#         ra_list.append(ra_array)
#         dec_list.append(dec_array)
#         energy_list.append(energy_array)
#         sigma_list.append(sigma_array)
#         xra_list.append(xra_array)
#         xdec_list.append(xdec_array)

#         print("Number of events loaded = {}".format(len(ra_array)))
#     dataset = pd.DataFrame({'ra': ra_list, 'dec': dec_list, 'energy': energy_list, 'sigma': sigma_list, 'xra': xra_list, 'xdec': xdec_list})
#     return dataset

def ra_deg_to_rad(array):
    ra_rad = []
    for i in range(len(array)):
        if array[i] < 180:
            ra_rad.append(np.deg2rad(array[i]))
        else:
            ra_rad.append(np.pi-np.deg2rad(array[i]))
    return ra_rad


def dec_deg_to_rad(array):
    dec_rad=[]
    for i in range(len(array)):
        dec_rad.append(np.deg2rad(array[i]))
    return dec_rad

def get_roi_corners(origin, coord_sys, convert_to_cel):

    if convert_to_cel:
        if coord_sys == 'G':
            r_low_cel_coord = SkyCoord(l= (origin[0]-origin[2]) * u.deg, b= (origin[1]-origin[3])* u.deg, unit='deg', frame='galactic')
            r_low = [r_low_cel_coord.icrs.ra.value, r_low_cel_coord.icrs.dec.value]

            l_low_cel_coord = SkyCoord(l=(origin[0]+origin[2]) * u.deg, b=(origin[1]-origin[3])* u.deg, unit='deg', frame='galactic')
            l_low = [l_low_cel_coord.icrs.ra.value, l_low_cel_coord.icrs.dec.value]

            r_high_cel_coord = SkyCoord(l=(origin[0]-origin[2]) * u.deg, b=(origin[1]+origin[3])* u.deg, unit='deg', frame='galactic')
            r_high = [r_high_cel_coord.icrs.ra.value, r_high_cel_coord.icrs.dec.value]

            l_high_cel_coord = SkyCoord(l=(origin[0]+origin[2]) * u.deg, b=(origin[1]+origin[3])* u.deg, unit='deg', frame='galactic')
            l_high = [l_high_cel_coord.icrs.ra.value, l_high_cel_coord.icrs.dec.value]
            
        else:
            r_low_cel_coord = SkyCoord(ra=(origin[0]-origin[2]) * u.deg, dec=(origin[1]-origin[3])* u.deg, unit='deg', frame='icrs')
            r_low = [r_low_cel_coord.icrs.ra.value, r_low_cel_coord.icrs.dec.value]

            l_low_cel_coord = SkyCoord(ra=(origin[0]+origin[2]) * u.deg, dec=(origin[1]-origin[3])* u.deg, unit='deg', frame='icrs')
            l_low = [l_low_cel_coord.icrs.ra.value, l_low_cel_coord.icrs.dec.value]

            r_high_cel_coord = SkyCoord(ra=(origin[0]-origin[2]) * u.deg, dec=(origin[1]+origin[3])* u.deg, unit='deg', frame='icrs')
            r_high = [r_high_cel_coord.icrs.ra.value, r_high_cel_coord.icrs.dec.value]

            l_high_cel_coord = SkyCoord(ra=(origin[0]+origin[2]) * u.deg, dec=(origin[1]+origin[3])* u.deg, unit='deg', frame='icrs')
            l_high = [l_high_cel_coord.icrs.ra.value, l_high_cel_coord.icrs.dec.value]
    
    return r_low, l_low, r_high, l_high

# def select_event_roi(dataframe, origin, coord_sys, convert_to_cel):
#     r_low, l_low, r_high, l_high = get_roi_corners(origin, coord_sys, convert_to_cel)
#     ra_list = []
#     dec_list = []
#     for row in dataframe.itertuples():
#         ra_array = []
#         dec_array = []
#         for ra, dec in zip(row.ra, row.dec):
#             if (r_high[0]<ra<l_low[0]):
#                 if (r_low[1]< dec<l_high[1]):
#                     ra_array.append(ra)
#                     dec_array.append(dec)
#         ra_list.append(ra_array)
#         dec_list.append(dec_array)
#     dataset = pd.DataFrame({'ra': ra_list, 'dec': dec_list})
#     return dataset

def select_event_roi(ra_array, dec_array, origin, coord_sys, convert_to_cel):
    # Get ROI corners: assuming returns (ra_min, dec_min, ra_max, dec_max)
    r_low, l_low, r_high, l_high = get_roi_corners(origin, coord_sys, convert_to_cel)

    ra_min = min(r_low[0], l_low[0], r_high[0], l_high[0])
    ra_max = max(r_low[0], l_low[0], r_high[0], l_high[0])
    dec_min = min(r_low[1], l_low[1], r_high[1], l_high[1])
    dec_max = max(r_low[1], l_low[1], r_high[1], l_high[1])

    selected_ra = []
    selected_dec = []

    for ra, dec in zip(ra_array, dec_array):
        if ra_min <= ra <= ra_max and dec_min <= dec <= dec_max:
            selected_ra.append(ra)
            selected_dec.append(dec)

    return selected_ra, selected_dec



# def select_event_roi_wenergy(ra_array, dec_array, energy_array, origin, coord_sys, convert_to_cel):
#     new_ra=[]
#     new_dec=[]
#     new_energy=[]
#     r_low, l_low, r_high, l_high = get_roi_corners(origin, coord_sys, convert_to_cel)
#     # print(r_low, l_low, r_high, l_high)
#     for ra, dec, energy in zip(ra_array, dec_array, energy_array):
#         if (r_high[0]<ra<l_low[0]):
#             if (r_low[1]< dec<l_high[1]):
#                 new_ra.append(ra)
#                 new_dec.append(dec)
#                 new_energy.append(energy)
#     # print("Len", len(new_ra))
#     ra_rad = []
#     for i in range(len(new_ra)):
#         if np.deg2rad(new_ra[i]) < np.pi:
#             ra_rad.append(np.deg2rad(new_ra[i])) #Radian
#         else:
#             ra_rad.append(np.pi-np.deg2rad((new_ra[i])))

#     dec_rad=[]
#     for i in range(len(new_dec)):
#         dec_rad.append(np.deg2rad(new_dec[i]))
#     return ra_rad, dec_rad, new_energy

# def select_event_roi_wenergy(ra_array, dec_array, energy_array, origin, coord_sys, convert_to_cel):
def select_event_roi_wenergy(dataframe, origin, coord_sys, convert_to_cel):
    r_low, l_low, r_high, l_high = get_roi_corners(origin, coord_sys, convert_to_cel)
    ra_list = []
    dec_list = []
    energy_list = []
    sigma_list = []
    for row in dataframe.itertuples():
        ra_array = []
        dec_array = []
        energy_array = []
        sigma_array = []
        for ra, dec, energy, sigma in zip(row.ra, row.dec, row.energy, row.sigma):
            if (r_high[0]<ra<l_low[0]):
                if (r_low[1]< dec<l_high[1]):
                    ra_array.append(ra)
                    dec_array.append(dec)
                    energy_array.append(energy)
                    sigma_array.append(sigma)
        ra_list.append(ra_array)
        dec_list.append(dec_array)
        energy_list.append(energy_array)
        sigma_list.append(sigma_array)
    dataset = pd.DataFrame({'ra': ra_list, 'dec': dec_list, 'energy': energy_list, 'sigma': sigma_list})
    return dataset


def select_event_roi_wenergy_wsigma(dataframe, origin, coord_sys, convert_to_cel):
    r_low, l_low, r_high, l_high = get_roi_corners(origin, coord_sys, convert_to_cel)
    ra_list = []
    dec_list = []
    energy_list = []
    sigma_list = []
    xra_list = []
    xdec_list = []
    logenergy_list = []
    for row in dataframe.itertuples():
        ra_array = []
        dec_array = []
        energy_array = []
        sigma_array = []
        xra_array = []
        xdec_array = []
        logenergy_array = []
        for ra, dec, energy, sigma, xra, xdec, logenergy in zip(row.ra, row.dec, row.energy, row.sigma, row.xra, row.xdec, row.logenergy):
            if (r_high[0]<ra<l_low[0]):
                if (r_low[1]< dec<l_high[1]):
                    ra_array.append(ra)
                    dec_array.append(dec)
                    energy_array.append(energy)
                    sigma_array.append(sigma)
                    xra_array.append(xra)
                    xdec_array.append(xdec)
                    logenergy_array.append(logenergy)
        ra_list.append(ra_array)
        dec_list.append(dec_array)
        energy_list.append(energy_array)
        sigma_list.append(sigma_array)
        xra_list.append(xra_array)
        xdec_list.append(xdec_array)
        logenergy_list.append(logenergy_array)
    dataset = pd.DataFrame({'ra': ra_list, 'dec': dec_list, 'energy': energy_list, 'sigma': sigma_list, 'xra': xra_list, 'xdec': xdec_list, 'logenergy': logenergy_list})
    return dataset


def load_files(directory):
    filelist = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filelist.append(file)
    return [filelist]


def energy_smear_map(ra, dec, energy, xra, xdec, bins=500, padding_sigma=1):
    """Create energy map with Gaussian smearing and edge padding."""
    max_xra = np.max(xra)
    max_xdec = np.max(xdec)

    ra_min = np.min(ra) - padding_sigma * max_xra
    ra_max = np.max(ra) + padding_sigma * max_xra
    dec_min = np.min(dec) - padding_sigma * max_xdec
    dec_max = np.max(dec) + padding_sigma * max_xdec

    ra_edges = np.linspace(ra_min, ra_max, bins)
    dec_edges = np.linspace(dec_min, dec_max, bins)
    ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    RA_grid, DEC_grid = np.meshgrid(ra_centers, dec_centers)


    heatmap = np.zeros_like(RA_grid)
    for i in range(len(ra)):
        if xra[i] == 0 or xdec[i] == 0:
            continue
        mean = [ra[i], dec[i]]
        cov = [[xra[i]**2, 0], [0, xdec[i]**2]]
        rv = multivariate_normal(mean, cov)
        kernel = rv.pdf(np.dstack((RA_grid, DEC_grid)))
        heatmap += energy[i] * kernel

    ra_mask = (ra_centers >= np.min(ra)) & (ra_centers <= np.max(ra))
    dec_mask = (dec_centers >= np.min(dec)) & (dec_centers <= np.max(dec))
    heatmap_cropped = heatmap[np.ix_(dec_mask, ra_mask)]
    RA_crop, DEC_crop = RA_grid[np.ix_(dec_mask, ra_mask)], DEC_grid[np.ix_(dec_mask, ra_mask)]

    return RA_crop, DEC_crop, heatmap_cropped, ra_min, ra_max, dec_min, dec_max

def blob_filter_intensity(blobs, image, min_intensity):
    hfiltered_blobs = []
    hfiltered_coords = []
    hfiltered_radius = []
    for blob in blobs:
        y, x, r = blob
        y_min, y_max = int(max(y - r, 0)), int(min(y + r, np.array(image).shape[0]))
        x_min, x_max = int(max(x - r, 0)), int(min(x + r, np.array(image).shape[1]))
        y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
        distance_from_center = np.sqrt((y_grid - y)**2 + (x_grid -x)**2)
        circular_mask = distance_from_center <= r
        mean_intensity = np.array(image)[y_min:y_max, x_min:x_max][circular_mask].mean()
        if min_intensity <= mean_intensity :
            coord = [x, y]
            hfiltered_blobs.append(blob)
            hfiltered_coords.append(coord)
            hfiltered_radius.append(r*(1/360))
            print(r"Blob Intensity {}, Coords ({}, {}), Radius {}, Pixel Radius {}".format(mean_intensity, coord[0], coord[1], r, r*(1/360)))
    return hfiltered_blobs, hfiltered_coords, hfiltered_radius



def setupMilagroColormap(amin, amax, threshold, ncolors):
    thresh = (threshold - amin) / (amax - amin)
    if threshold <= amin or threshold >= amax:
        thresh = 0.
    dthresh = 1 - thresh
    threshDict = { "blue"  : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0, 0),
                              (thresh+0.462*dthresh, 0, 0),
                              (thresh+0.615*dthresh, 1, 1),
                              (thresh+0.692*dthresh, 1, 1),
                              (thresh+0.769*dthresh, 0.6, 0.6),
                              (thresh+0.846*dthresh, 0.5, 0.5),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)),
                   "green" : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0, 0),
                              (thresh+0.231*dthresh, 0, 0),
                              (thresh+0.308*dthresh, 1, 1),
                              (thresh+0.385*dthresh, 0.8, 0.8),
                              (thresh+0.462*dthresh, 1, 1),
                              (thresh+0.615*dthresh, 0.8, 0.8),
                              (thresh+0.692*dthresh, 0, 0),
                              (thresh+0.846*dthresh, 0, 0),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)),
                   "red"   : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0.5, 0.5),
                              (thresh+0.231*dthresh, 1, 1),
                              (thresh+0.385*dthresh, 1, 1),
                              (thresh+0.462*dthresh, 0, 0),
                              (thresh+0.692*dthresh, 0, 0),
                              (thresh+0.769*dthresh, 0.6, 0.6),
                              (thresh+0.846*dthresh, 0.5, 0.5),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)) }

    newcm = mpl.colors.LinearSegmentedColormap("thresholdColormap",
                                               threshDict,
                                               ncolors)
    newcm.set_over(newcm(1.0))
    newcm.set_under("w")
    newcm.set_bad("gray")
    textcolor = "#000000"

    return textcolor, newcm
