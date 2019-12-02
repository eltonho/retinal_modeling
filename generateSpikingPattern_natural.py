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
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch.nn.functional as F
import torch.nn.init
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable

dsfolder = "./datasets/"
cellfile = 'retina_LE_V.json'
with open(dsfolder + cellfile,"r") as datafile:
  cell = json.load(datafile)

dictLists2npArray(cell)

lC_stimulus_file = "./datasets/landoltC_stimulus_dict.json"
with open(lC_stimulus_file, 'r') as f_in:
	landoltC_stimulus_dict = json.load(f_in)

cx = np.arange(10) + 25
cy = np.arange(10) + 8

centerList = [[x, y] for x in cx for y in cy]

widthList = 2 * np.arange(8) + 2
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
	wfn = 'LE_V_spikingPattern_cell_' + str(key) + '.json'
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