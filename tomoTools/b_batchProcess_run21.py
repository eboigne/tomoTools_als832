from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import ipywidgets
ipywidgets.Widget.close_all()

import SimpleITK
import datetime
import gc # Garbage collected
import numpy as np
import sys, os
import time
import tifffile
import matplotlib.pyplot as plt
import scipy
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory
import torch
import copy

this_path = os.getcwd()
print(this_path)


import h5py
# import File
from scripts import *
import wrapper_ASTRA

# path_save = '/Documents/nersc_als/2022_wood/'
path_save = '/global/homes/e/eboigne/cfs_als/2022_wood/'

path_in = path_save

torch.cuda.set_device(1)

#%% Run 21 (Oak, low heat)

scans = []
scans_queue = []
tiles_queue = []

if True:
    scan = {}
    scan['path_proj'] = path_in+'/20220501_100425_21_SampleO7_2xLens_01_pre_2625.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -8.8
    scan['N_angles_per_half_circle'] = 2625
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_100935_21_SampleO7_2xLens_02_pre_2625.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -8.8
    scan['N_angles_per_half_circle'] = 2625
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_101446_21_SampleO7_2xLens_03_pre_2625.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -8.9
    scan['N_angles_per_half_circle'] = 2625
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_102625_21_SampleO7_2xLens_04_dry_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.0
    scan['N_angles_per_half_circle'] = 1313
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_102927_21_SampleO7_2xLens_05_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.2
    scan['N_angles_per_half_circle'] = 1313
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_103229_21_SampleO7_2xLens_06_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.32
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_103532_21_SampleO7_2xLens_07_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.32
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_103832_21_SampleO7_2xLens_08_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.53
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_104143_21_SampleO7_2xLens_09_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.58
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_104454_21_SampleO7_2xLens_10_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.51
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_104802_21_SampleO7_2xLens_11_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.58
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_105112_21_SampleO7_2xLens_12_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.58
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_105420_21_SampleO7_2xLens_13_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.58
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_105723_21_SampleO7_2xLens_14_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.72
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_110021_21_SampleO7_2xLens_15_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -9.93
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_110325_21_SampleO7_2xLens_16_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -10.1
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_110625_21_SampleO7_2xLens_17_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -10.1
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_110926_21_SampleO7_2xLens_18_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -10.34
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_111226_21_SampleO7_2xLens_19_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -10.26
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_111544_21_SampleO7_2xLens_20_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -11.8
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_111845_21_SampleO7_2xLens_21_cooling_1313.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -11.15
    scan['N_angles_per_half_circle'] = 1313
    scan['split_half_circles'] = True
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_112154_21_SampleO7_2xLens_22_post_2625.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -11.45
    scan['N_angles_per_half_circle'] = 2625
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_112705_21_SampleO7_2xLens_23_post_2625.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -11.6
    scan['N_angles_per_half_circle'] = 2625
    scans.append(scan)
    scans_queue.append(scan)

    scan = {}
    scan['path_proj'] = path_in+'/20220501_113214_21_SampleO7_2xLens_24_post_2625.h5'
    scan['use_dark_from_scan'] = False
    scan['path_dark'] = path_in+'/20220501_094201_21_dark_2560x1396.h5'
    scan['use_flat_from_scan'] = True
    scan['COR'] = -11.85
    scan['N_angles_per_half_circle'] = 2625
    scans.append(scan)
    scans_queue.append(scan)


tiles_queue.append(('run21_oak_lowHeat', scans))
path_mask = path_in+'mask_allWhite.tif'

#%% Check which tiles are used

for tile in tiles_queue:
    print(tile[0])
    for e in tile[1]:
        print(e)

#%% Modify tile list to split half circles if needed

new_tiles_queue = []
for tile in tiles_queue:

    name_tile = tile[0]
    new_scan_list = []

    for ind_scan in range(len(tile[1])):
        this_scan = tile[1][ind_scan]
        list_FBPs = sorted([e for e in os.listdir(this_scan['path_proj'][:-3]) if 'FBP' in e])

        this_scan['ind'] = this_scan['path_proj'].split('/')[-1].split('_')[5]

        if len(list_FBPs) > 1:
            this_scan['FBP_folder'] = list_FBPs[0]
            this_scan['ind'] = this_scan['ind']+'a'
            new_scan_list.append(this_scan)

            new_scan = copy.deepcopy(this_scan)
            new_scan['FBP_folder'] = list_FBPs[1]
            new_scan['ind'] = new_scan['ind'][:-1]+'b'
            new_scan_list.append(new_scan)
        else:
            this_scan['FBP_folder'] = list_FBPs[0]
            new_scan_list.append(this_scan)

    new_tile = (name_tile, new_scan_list)
    new_tiles_queue.append(new_tile)

# print(new_tiles_queue[0][1][23])
# new_tiles_queue[0] = (new_tiles_queue[0][0], new_tiles_queue[0][1][23:])


for new_tile in new_tiles_queue:
    print(new_tile[0])
    for e in new_tile[1]:
        print(e['ind'], e)

#%% Align scans to each other

# Test options
# nb_it = 1
# smoothingSigmas = [1]
# shrinkFactors = [8]
# ind_z_start = 20

# # Real
nb_it = 10
smoothingSigmas = [4, 2, 2, 2, 2, 1, 1]
shrinkFactors =   [8, 8, 4, 4, 2, 2, 1]
ind_z_start = 20

path_tile = path_save
for tile in new_tiles_queue:

    name_tile = tile[0]
    print('\n\n============================== NEW TILE ==============================\n')
    print('Processing tile: '+name_tile)

    print('\n\n============ NEW SCAN ============\n')
    print('Processing scan: '+tile[1][0]['path_proj'].split('/')[-1])
    ind = 0

    static = import_data(path_tile, tile[1][0])
    File(path_tile+name_tile+'/'+tile[1][0]['ind']+'/b_movingRegisteredToStatic/').saveTiffStack(static)

    # static = File(path_tile+name_tile+'/'+tile[1][0]['ind']+'/b_movingRegisteredToStatic/').readAll()

    print(static.shape)
    moving_reg = static

    for ind_scan in range(1, len(tile[1])):
        scan = tile[1][ind_scan]

        print('\n\n============ NEW SCAN ============\n')
        print('Processing scan: '+scan['path_proj'].split('/')[-1])

        static = moving_reg
        moving = import_data(path_tile, scan)

        folder_out_name = path_tile+name_tile+'/'+scan['ind']+'/'
        moving_reg = run_registration(folder_out_name, static, moving, path_mask, save_str='b', nb_it = nb_it, smoothingSigmas = smoothingSigmas, shrinkFactors = shrinkFactors, ind_z_start=ind_z_start)
        print(folder_out_name)

    print('\n')
    print('\n\n')

#%%

print(time.time())
