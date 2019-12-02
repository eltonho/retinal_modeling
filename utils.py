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

# convert times among three systems: real, mea, wn

def timeConversion(t, source = "mea", target = "wn", WNType = "N"):
  
  meaRepRate = 20000
  if WNType == "N":
    stimRepRate = 1 / (33.27082098251457/1000)
  else:
    stimRepRate = 1 / (49.90624667553192/1000)
  
  if source == "mea":
    t = np.divide(t, meaRepRate)
  elif source == "wn":
    t = np.divide(t, stimRepRate)
  elif source == "ms":
    t = np.divide(t, 1000)
  
  if target == "mea":
    t = np.multiply(t, meaRepRate)
  elif target == "wn":
    t = np.multiply(t, stimRepRate)
  elif target == "ms":
    t = np.multiply(t, 1000)
  
  return t


# =====================
# compute STA

def getSTA(experiment,  whiteNoise, staLength = 25, verbose = "on", normalize = False):

  startTime = experiment['info']['startTime']
  skipped = 0
  STA = np.zeros((staLength * 2, experiment['info']['WNShape'][0], experiment['info']['WNShape'][1]))

  st_wn = timeConversion(np.array(experiment["spikeTimes"]) - startTime, source = "mea", target = "wn", WNType = experiment['info']['WNType'])
  st_frame = np.ceil(st_wn).astype(int)
  st_frame = st_frame[st_frame >= (staLength * 2 - 1)]

  #for i, st in enumerate(experiment["spikeTimes"]):
  for i, st in enumerate(st_frame):
    # if st < timeConversion(staLength * 2 - 1, source = "wn", target = "mea", WNType = experiment['info']['WNType']) + startTime:
    #   skipped += 1
    #   continue
    if i % 2000 == 0 and verbose == "on":
      print(f"Computing spike {i} out of {len(experiment['spikeTimes'])}; using frame {st_frame}")

    # st_wn = timeConversion(st - startTime, source = "mea", target = "wn")
    # st_frame = int(np.ceil(st_wn))
    shift = 0
    try:
      STA += whiteNoise[st-staLength * 2 + 1 + shift:st + 1 + shift, :, :]
    except:
      print(f"Terminated on spike #{i}")
      print(f"Spike time at {st}. White noise length: {whiteNoise.shape[0]}")
      break

  STA /= (i - 1 - skipped)
  if normalize == True: STA = (STA - np.mean(STA)) / (STA.max(axis = 0) - STA.min(axis = 0))
  
  return STA


# =====================
# Bin generator signals

def getGenerator(experiment, whiteNoise, STA):
  print(f"STA dimensions: {STA.shape}")
  print(f"WN dimensions: {whiteNoise.shape}")
  MAX = STA.max(axis = None)
  MIN = STA.min(axis = None)
  MEAN = np.mean(STA, axis = None)
  
  stimulus = np.zeros(STA.shape)
  numStim = whiteNoise.shape[0] - STA.shape[0] + 1
  generator = np.zeros((numStim,))
  #kernel = np.flip(STA, axis = 0)
  #kernel = (STA[:,:,::-1] - MEAN) / (MAX - MIN)
  kernel = (STA - MEAN) / (MAX - MIN)
  
  for i in range(numStim):
    generator[i] = np.sum(np.multiply(whiteNoise[i:i + STA.shape[0],:,:], kernel), axis = None)
  
  return np.array(generator)


# =====================
# Get responses

def getResponse(experiment):
  refreshPeriod = timeConversion(experiment['info']['refreshPeriod'], source = "ms", target = "mea", WNType = experiment['info']['WNType'])
  spikeTimes = np.array(experiment['spikeTimes']) - experiment['info']['startTime'] - (STA.shape[0]) * refreshPeriod
#   spikeTimes = np.floor(spikeTimes)
  spikeTimes = np.floor(spikeTimes[spikeTimes > 0] / refreshPeriod)
  response = [len(list(group)) for key, group in groupby(spikeTimes)]
  
#   respLen = int(experiment['info']['wn']['duration'])
#   response = np.zeros((respLen,))
  
#   for i in range(respLen):
#     for j in range(len(spikeTimes)):
#       if np.floor(spikeTimes[j] / refreshPeriod) == i:
#         response[i] += 1
#       else:
#         if j == 0: break
#         spikeTimes = np.delete(spikeTimes, np.arange(j))
#         break

  return np.array(response)


# =====================
# Compute nonlinear function

def getNL(generator, response, shift, nbins = 800):
  MAX = np.max(generator)
  MIN = np.min(generator)
  INT = (MAX - MIN) / nbins
  
  counts = np.zeros((nbins, 2))
  length = np.min([generator.shape[0] , response.shape[0]])
  
  for i in range(length):
    if i+shift > len(response)-1: break
    binNum = int(np.ceil((generator[i] - MIN) / INT) + 1)
    counts[binNum, 0] += response[i+shift]
    counts[binNum, 1] += 1
  
  return np.divide(counts[:,0], counts[:,1])


# =====================
# Get average stimulus given a generator

def stimulationFromGenerator(whiteNoise, STA, generator, signal, width, shift, plot = "on"):
  idx = np.where(((np.absolute(generator - signal) < width / 2)) == 1)[0]
  idx = np.array(idx).astype(dtype = int)
  avgStim = np.zeros(STA.shape)
  for _,i in enumerate(idx):
    avgStim += whiteNoise[i + shift:i + shift + STA.shape[0], :, :]
  
  avgStim /= len(idx)
  
  if plot == "on":
    plt.figure(num = None, figsize = (np.ceil(avgStim.shape[0]/2),2))
    for i in range(avgStim.shape[0]):
      plt.subplot(2, np.ceil(avgStim.shape[0]/2), i+1)
      plt.imshow(avgStim[i,:,:])
      plt.clim(avgStim.min(axis = None), avgStim.max(axis = None))
      plt.axis('off')
    
  return avgStim, idx


# =====================
# Get raster data with a given generator range

def rasterFromGenerator(experiment, generator, genRange, t):
  
  refreshPeriod = timeConversion(experiment['info']['refreshPeriod'], source = 'ms', target = 'mea', WNType = experiment['info']['WNType'])
  #print(refreshPeriod)
  generator = np.array(generator)
  staLength = experiment['info']['STALength']
  idx = np.where(((np.absolute(generator - genRange[0]) < genRange[1] / 2)) == 1)[0]
  trialTimes = timeConversion(idx, 
                              source = "wn", 
                              target = "mea", 
                              WNType = experiment['info']['WNType'])
  trialTimes += experiment['info']['startTime'] + (staLength-4) * refreshPeriod
  spikeTimes = np.array(experiment['spikeTimes'])
  
  raster = dict()
  for i in range(len(trialTimes)):
    spikes = np.where(np.logical_and(spikeTimes > trialTimes[i], spikeTimes < trialTimes[i] + t))[0]
    trial = np.array([spikeTimes[j] for j in spikes]) - trialTimes[i] #- (staLength-1) * refreshPeriod
    #spikeTimes.remove(spikeTimes[(trialTimes[i] - spikeTimes).any()])
    raster[i] = trial
    
  return raster, idx


# =====================
# Plot raster

def plotRaster(raster, WNType):
  for i, trial in raster.items():
    if len(trial) == 0: continue
    trial = timeConversion(trial, source = "mea", target = "real", WNType = WNType)
    plt.scatter(trial, i * np.ones(len(trial)), c = 'tab:blue', s = 1)
  return None


# =====================
# Lists in dictionary to Numpy arrays

def dictLists2npArray(d):
  for k, v in d.items():
    if isinstance(v, dict):
      dictLists2npArray(v)
    elif isinstance(v, list):
      d[k] = np.array(v)


# =====================
# Make geometries

def makeHexagon(l, s = 1):
  a1 = [s * np.cos(np.pi/6), s * np.sin(np.pi/6)]
  corners = [np.array(l)]
  corners.append(l + np.array([a1[0], -a1[1]]))
  corners.append(l + np.array([2 * a1[0], 0]))
  corners.append(l + np.array([2 * a1[0], s]))
  corners.append(l + np.array([a1[0], s + a1[1]]))
  corners.append(l + np.array([0, s]))
  return corners

def makeRectangle(l = (0,0), dim = (1,1)):
  corners = [np.array(l)]
  corners.append(l + np.array([dim[0], 0]))
  corners.append(l + np.array([dim[0], dim[1]]))
  corners.append(l + np.array([0, dim[1]]))
  return corners

def makeBar(center, width, angle):
  R = np.array([[np.cos(angle), -np.sin(angle)], 
                [np.sin(angle), np.cos(angle)]])
  center = np.array(center)
  corners = np.array([[-40, -width/2],
             [-40, width/2],
             [40, width/2],
             [40, -width/2]])
  newCorners = np.matmul(R,corners.T)
  newCorners = newCorners.T + center
  return newCorners

def makeLandoltC(center, diameter, angle):
  R = np.array([[np.cos(angle), -np.sin(angle)], 
                [np.sin(angle), np.cos(angle)]])
  center = np.array(center)
  outer = Polygon(Point(*center).buffer(diameter/2))
  inner = Polygon(Point(*center).buffer (diameter/2 - diameter/5))
  ring = outer.difference(inner)
  opening = np.array([[0, -diameter/10],
                     [diameter/2, -diameter/10],
                     [diameter/2, diameter/10],
                     [0, diameter/10]])
  opening = Polygon(np.matmul(R,opening.T).T + center)
  landoltC = ring.difference(ring.intersection(opening))
  return landoltC
  

# =====================
# Make bar for hexgonal lattice

def buildBarStimulus(params, visualize = False):

  center = params['center']
  width = params['width']
  angle = params['angle']
  startFrame = params['startFrame']
  dims = params['dims']

  origin = [0, 0]
  hexCoord = makeHexagon(origin, 1)
  hx = Polygon(hexCoord)

  xlist = np.arange(dims[1])
  ylist = np.arange(dims[2])

  xylist = list(product(xlist, ylist))
  hx_dict = dict()
  hx_coord = []

  bar = Polygon(makeBar(center, width, angle))
  inter = dict()

  for x, y in xylist:
    xx = np.cos(np.pi/6) * (2*x+y-2*np.floor(y/2))
    yy = 1.5 * y
    hx_coord.append([xx, yy])
    hx_dict[(x,y)] = Polygon(makeHexagon([xx,yy], 1))
    inter[(x,y)] = hx_dict[(x,y)].intersection(bar).area / hx_dict[(x,y)].area #/ 2 + 0.5

  outStim = np.zeros((dims[0],dims[1],dims[2])) #* 0.5
  for x in range(dims[1]):
    for y in range(dims[2]):
      outStim[startFrame:, x, y] = inter[(x,y)]

  if visualize == True:
    plt.subplot(1,2,1)

    for (x,y), hx in hx_dict.items():
      xl, yl = hx.exterior.xy
      plt.plot(xl,yl)
      plt.scatter(*hx.centroid.xy, alpha = inter[(x,y)])

    bx, by = bar.exterior.xy
    plt.plot(bx, by)
    plt.xlim(-1,35)
    plt.ylim(-1,31)

    plt.subplot(1,2,2)

    plt.imshow(outStim[-1,:,:].T)
    plt.show()
  return outStim

# =====================
# Make bar for square lattice

def buildBarStimulus2(params, visualize = False):

  center = params['center']
  width = params['width']
  angle = params['angle']
  startFrame = params['startFrame']
  dims = params['dims']

  origin = [0, 0]
  sqCoord = makeRectangle()
  sq = Polygon(sqCoord)

  xlist = np.arange(dims[1])
  ylist = np.arange(dims[2])

  xylist = list(product(xlist, ylist))
  sq_dict = dict()
  sq_coord = []

  bar = Polygon(makeBar(center, width, angle))
  inter = dict()

  for x, y in xylist:
    sq_coord.append([x, y])
    sq_dict[(x,y)] = Polygon(makeRectangle([x,y]))
    inter[(x,y)] = sq_dict[(x,y)].intersection(bar).area / sq_dict[(x,y)].area #/ 2 + 0.5

  outStim = np.ones((dims[0],dims[1],dims[2])) #* 0.5
  for x in range(dims[1]):
    for y in range(dims[2]):
      outStim[startFrame:, x, y] = inter[(x,y)]

  if visualize == True:
    plt.subplot(1,2,1)

    for (x,y), sq in sq_dict.items():
      xl, yl = sq.exterior.xy
      plt.plot(xl,yl)
      plt.scatter(*sq.centroid.xy, alpha = inter[(x,y)])

    bx, by = bar.exterior.xy
    plt.plot(bx, by)
    plt.xlim(-1,65)
    plt.ylim(-1,33)

    plt.subplot(1,2,2)

    plt.imshow(outStim[-1,:,:].T)
    plt.show()
  return outStim


# =====================
# Make Landolt C on hexgonal lattice

def buildLandoltCStimulus(params, visualize = False):

  center = params['center']
  width = params['width']
  angle = params['angle']
  startFrame = params['startFrame']
  dims = params['dims']

  origin = [0, 0]
  #hexCoord = makeHexagon(origin, 1)
  #hx = Polygon(hexCoord)

  xlist = np.arange(dims[1])
  ylist = np.arange(dims[2])

  xylist = list(product(xlist, ylist))
  c_dict = dict()
  c_coord = []

  print(f"{center} {width} {angle}")
  C = makeLandoltC(center, width, angle)
  inter = dict()

  for x, y in xylist:
    xx = np.cos(np.pi/6) * (2*x+y-2*np.floor(y/2))
    yy = 1.5 * y
    c_coord.append([xx, yy])
    c_dict[(x,y)] = Polygon(makeHexagon([xx,yy], 1))
    inter[(x,y)] = c_dict[(x,y)].intersection(C).area / c_dict[(x,y)].area #/ 2 + 0.5

  outStim = np.zeros((dims[0],dims[1],dims[2])) #* 0.5
  for x in range(dims[1]):
    for y in range(dims[2]):
      outStim[startFrame:, x, y] = inter[(x,y)]

  if visualize == True:
    plt.subplot(1,2,1)

    for (x,y), c in c_dict.items():
      xl, yl = c.exterior.xy
      plt.plot(xl,yl)
      plt.scatter(*c.centroid.xy, alpha = inter[(x,y)])

    bx, by = C.exterior.xy
    plt.plot(bx, by)
    plt.xlim(-1,35)
    plt.ylim(-1,31)

    plt.subplot(1,2,2)

    plt.imshow(outStim[-1,:,:].T)
    plt.show()
  return outStim

# =====================
# Make Landolt C on hexgonal lattice

def buildLandoltCStimulus_natural(params, visualize = False):

  center = params['center']
  width = params['width']
  angle = params['angle']
  startFrame = params['startFrame']
  dims = params['dims']

  origin = [0, 0]

  xlist = np.arange(dims[1])
  ylist = np.arange(dims[2])

  xylist = list(product(xlist, ylist))
  c_dict = dict()
  c_coord = []

  print(f"{center} {width} {angle}")
  C = makeLandoltC(center, width, angle)
  
  outStim = np.zeros((dims[0],dims[1],dims[2])) #* 0.5

  for x, y in xylist:
    c_dict[(x,y)] = Polygon(makeRectangle([x,y], (1, 1)))
    outStim[startFrame:, x, y] = c_dict[(x,y)].intersection(C).area / c_dict[(x,y)].area #/ 2 + 0.5

  
#   for x in range(dims[1]):
#     for y in range(dims[2]):
#       outStim[startFrame:, x, y] = inter[(x,y)]

  if visualize == True:
    plt.subplot(1,2,1)

    for (x,y), c in c_dict.items():
      xl, yl = c.exterior.xy
      plt.plot(xl,yl)
      plt.scatter(*c.centroid.xy, alpha = outStim[-1,x,y])

    bx, by = C.exterior.xy
    plt.plot(bx, by)
    plt.xlim(-1,35)
    plt.ylim(-1,31)

    plt.subplot(1,2,2)

    plt.imshow(outStim[-1,:,:].T)
    plt.show()
  return outStim

def composeKeyString(center, width, angle):
    return '(('+ str(center[0]) + ', ' + str(center[1]) + '), ' + str(width) + ', ' + str(angle)+ ')'

def decomposeKeyString(keystring):
    array = np.fromstring(keystring)
    return array[0], array[1], array[2]