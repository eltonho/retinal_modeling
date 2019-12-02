import json
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
import random
from itertools import groupby, product, permutations
from scipy import optimize, stats
import shapely
from shapely.geometry import Polygon, Point
import pickle
import time
from utils import timeConversion, dictLists2npArray, makeRectangle, makeLandoltC, buildLandoltCStimulus_natural

start_time = time.time()
print(f"Start generating Landolt Cs at {time.ctime()}")
dsfolder = "./datasets/"
cellfile = 'retina_LE_V.json'

with open(dsfolder + cellfile,"r") as datafile:
  cell = json.load(datafile)

dictLists2npArray(cell)

#  === EDIT THIS BLOCK ===
n_output = 2
lum = (0.5, 1)
tol = 0.1
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

cx = np.arange(10) + 25
cy = np.arange(10) + 8

centerList = [[x, y] for x in cx for y in cy]

widthList = 2 * np.arange(8) + 2
angleList = np.arange(4)

# 3.
stims = dict()

WNType = cell[random.choice(list(cell.keys()))]['experiment']['info']['WNType']

if WNType == 'N': dims = (24,64,32)
else: dims = (24,20,20)

params = dict()
outStim = dict()
for center in centerList:
	print(f"Center: {center}")
	for width in widthList:
		for angle in angleList:
			key = ((int(center[0]),int(center[1])), int(width), int(angle))
			params[key] = dict()
			params[key]['center'] = center
			params[key]['width'] = width
			params[key]['angle'] = angle * np.pi/2
			params[key]['startFrame'] = 21
			params[key]['dims'] = dims
			
			if np.random.random() > 1:
				outStim[key] = buildLandoltCStimulus_natural(params[key], visualize = True)
			else:
				outStim[key] = buildLandoltCStimulus_natural(params[key], visualize = False)
	print(f"Elapsed time - {time.time() - start_time}")

lc_map = dict()
for key, val in params.items():
	lc_map[str(key)] = dict()
	lc_map[str(key)]['params'] = dict()
	lc_map[str(key)]['params']['center'] = [int(i) for i in params[key]['center']]
	lc_map[str(key)]['params']['width'] = int(params[key]['width'])
	lc_map[str(key)]['params']['angle'] = int(params[key]['angle'] / np.pi * 2)
	lc_map[str(key)]['params']['dims'] = params[key]['dims']
	lc_map[str(key)]['stimulus'] = outStim[key].astype(float).tolist()

dbfolder = "./datasets/"
outfile = "landoltC_stimulus_dict.json"
fn = dbfolder + outfile
print(fn)
with open(fn, 'w') as fout:
	json.dump(lc_map, fout)
