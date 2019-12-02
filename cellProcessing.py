import json
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import timeit
import random
from itertools import groupby, product
from scipy import optimize,stats
import shapely
from shapely.geometry import Polygon, Point
from utils import *

dsfolder = './datasets/'

cellfile = 'retina_LE_V.json'
cellfile = 'retina_LE_E.json'
cellfile = 'retina_RCS_E.json'
#filename = 'RCS_spikeTimes_cell391.json'
#filename = 'LE_E_spikeTimes_cell301.json'
with open(dsfolder + cellfile,"r") as datafile:
  cell = json.load(datafile)

# counter = 0
# cell = dict()
# for key, val in cell1.items():
#     counter += 1
#     if key == '5311' or key == '5566': cell[key] = val

#print(cell)
dictLists2npArray(cell)

#  === EDIT THIS BLOCK ===
n_output = 2
lum = (0.5, 1)
tol = 0.1
horizBar = (7, 2) #(start, width)
vertBar = (7, 2)
startFrame = 13
# === ===

# 2. 
plotDuration = timeConversion(2, source = 'real', target = 'mea', WNType = 'N')
binWidth = timeConversion(0.01, source = 'real', target = 'mea', WNType = 'N')
nbins = int(plotDuration / binWidth)

popRaster = dict()
spikingPattern = dict()
spikingPattern['info'] = dict()
spikingPattern['info']['stimulusType'] = 'crossbar'


# 3.
stims = dict()

WNType = cell[random.choice(list(cell.keys()))]['experiment']['info']['WNType']

if WNType == 'N': dims = (24,64,32)
else: dims = (24,20,20)

params = dict()
outStim = dict()
for i in range(4):
  params[i] = dict()
  params[i]['center'] = (17,15)#(32, 12) #(17,15)
  params[i]['width'] = 8
  params[i]['angle'] = np.pi/2 * i
  params[i]['startFrame'] = 20
  params[i]['dims'] = dims
  if WNType == 'N':
    outStim[i] = buildBarStimulus2(params[i], visualize = True)
  else:
    #outStim[i] = buildBarStimulus(params[i], visualize = True)
    outStim[i] = buildLandoltCStimulus(params[i], visualize = False)

for j in ('dims','lum','plotDuration','binWidth','nbins', 'tol'):
  spikingPattern['info'][j] = locals()[j]

lC_stimulus_file = "/datasets/landoltC_stimulus_dict.json"
with open(lC_stimulus_file, 'r') as f_in:
    landoltC_stimulus_dict = json.load(f_in)

#  === EDIT THIS BLOCK ===
center = (14,14)
width = 16

# === ===

# 2. 
plotDuration = timeConversion(2, source = 'real', target = 'mea', WNType = 'N')
binWidth = timeConversion(0.01, source = 'real', target = 'mea', WNType = 'N')
nbins = int(plotDuration / binWidth)

popRaster = dict()
spikingPattern = dict()
spikingPattern['info'] = dict()
spikingPattern['info']['stimulusType'] = 'landoltC'


# 3.
stims = dict()

WNType = cell[random.choice(list(cell.keys()))]['experiment']['info']['WNType']

if WNType == 'N': dims = (24,64,32)
else: dims = (24,20,20)

params = dict()
outStim = dict()
for i in range(4):
    keystring = '(('+ str(center[0]) + ', ' + str(center[1]) + '), ' + str(width) + ', ' + str(i)+ ')'
    outStim[i] = np.array(landoltC_stimulus_dict[keystring]['stimulus'])
    params[i] = np.array(landoltC_stimulus_dict[keystring]['params'])

cx = np.arange(10) + 13
cy = np.arange(10) + 10
centerList = np.array([[x, y] for x in cx for y in cy])

print(centerList)

widthList = 2 * np.arange(8) + 4
#widthList = np.array([10])
angleList = np.arange(4)

params = dict()
outStim = dict()
gen = dict()


for center in centerList:
    for width in widthList:
        for angle in angleList:
            key = composeKeyString(center, width, angle)
            outStim[key] = np.array(landoltC_stimulus_dict[key]['stimulus'])

   tol = 0.1

plotDuration = timeConversion(2, source = 'real', target = 'mea', WNType = 'N')
binWidth = timeConversion(0.01, source = 'real', target = 'mea', WNType = 'N')
nbins = int(plotDuration / binWidth)

populationRaster = dict()
spikingPattern = dict()
spikingPattern['info'] = dict()
spikingPattern['info']['stimulusType'] = 'landoltC'
counter = 1
flist = os.listdir(dsfolder)

for key, c in cell.items():
    wfn = 'RCS_E_spikingPattern_cell_' + str(key) + '.json'
    print("\n =============================")
    print(f"Computing spiking pattern for cell #{key} ({counter}/{len(cell)})")
    counter += 1
    start_time = time.time()
    if wfn in flist:
        print(f"{wfn} already exists! Skipping...")
        continue
    
    STA = np.array(c['STA'])

    MEAN = np.mean(STA)
    MAX = np.max(STA)
    MIN = np.min(STA)
    kernel = (STA - MEAN) / (MAX - MIN)

    gen = dict()
    populationRaster[key] = dict()
    generator = c['generator']
    experiment = c['experiment']
    raster = dict()
    spikingPattern = dict()
    
    for stimKey, stim in outStim.items():
        gen[stimKey] = np.sum(np.multiply(kernel,stim), axis = None)
        #print(f"Generator for {stimKey}: {gen[stimKey]}")

        populationRaster[key][stimKey], idx_h = rasterFromGenerator(experiment, generator, (gen[stimKey], tol), plotDuration)
        raster[stimKey] = np.array([k for i,k in populationRaster[key][stimKey].items()])
        norm = np.average([len(aa) for aa in raster[stimKey]])/nbins
        raster[stimKey] = timeConversion(np.concatenate(raster[stimKey]).ravel(), source = "mea", target = "real", WNType = experiment['info']['WNType'])

        spikingPattern[stimKey] = dict()
        spikingPattern[stimKey], _, _ = stats.binned_statistic(raster[stimKey],
                                                           np.ones_like(raster[stimKey]) * norm / len(raster[stimKey]),
                                                           statistic = 'sum',
                                                           bins = nbins)
        spikingPattern[stimKey] = spikingPattern[stimKey].tolist()

    with open(dsfolder + wfn,'w') as f_out:
        json.dump(spikingPattern, f_out)

    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time} secs')