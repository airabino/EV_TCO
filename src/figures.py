import os
os.environ['USE_PYGEOS'] = '0'
import sys
import time
import json
import requests
import warnings
import numpy as np
import numpy.random as rand
import pandas as pd
import geopandas as gpd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from shapely.ops import cascaded_union
from itertools import combinations
from shapely.geometry import Point,Polygon,MultiPolygon
from scipy.stats import t
from scipy.stats._continuous_distns import _distn_names
from cycler import cycler

#Defining some 4 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_4_0=["#8de4d3", "#0e503e", "#43e26d", "#2da0a1"]
color_scheme_4_1=["#069668", "#49edc9", "#2d595a", "#8dd2d8"]

#Defining some 3 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_3_0=["#72e5ef", "#1c5b5a", "#2da0a1"]
color_scheme_3_1=["#256676", "#72b6bc", "#1eefc9"]
color_scheme_3_2=['#40655e', '#a2e0dd', '#31d0a5']

#Defining some 2 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_2_0=["#21f0b6", "#2a6866"]
color_scheme_2_1=["#72e5ef", "#3a427d"]
color_scheme_2_2=["#6f309f", "#dfccfa"]

#Distributions to try (scipy.stats continuous distributions)
dist_names=['alpha','beta','gamma','logistic','norm','lognorm']
dist_labels=['Alpha','Beta','Gamma','Logistic','Normal','Log Normal']

#Named color schemes from https://www.canva.com/learn/100-color-combinations/

colors={
	'day_night':["#e6df44","#f0810f","#063852","#011a27"],
	'beach_house':["#d5c9b1","#e05858","#bfdccf","#5f968e"],
	'autumn':["#db9501","#c05805","#6e6702","#2e2300"],
	'ocean':["#003b46","#07575b","#66a5ad","#c4dfe6"],
	'forest':["#7d4427","#a2c523","#486b00","#2e4600"],
	'aqua':["#004d47","#128277","#52958b","#b9c4c9"],
	'field':["#5a5f37","#fffae1","#524a3a","#919636"],
	'misty':["#04202c","#304040","#5b7065","#c9d1c8"],
	'greens':["#265c00","#68a225","#b3de81","#fdffff"],
	'citroen':["#b38540","#563e20","#7e7b15","#ebdf00"],
	'blues':["#1e1f26","#283655","#4d648d","#d0e1f9"],
	'dusk':["#363237","#2d4262","#73605b","#d09683"],
	'ice':["#1995ad","#a1d6e2","#bcbabe","#f1f1f2"],
	'ibm':['#648fff','#785ef0','#dc267f','#fe6100','#ffb000'],
	'2_0':["#21f0b6","#2a6866"],
	'2_1':["#72e5ef","#3a427d"],
	'2_2':["#6f309f","#dfccfa"],
	'3_0':["#72e5ef","#1c5b5a","#2da0a1"],
	'3_1':["#256676","#72b6bc","#1eefc9"],
	'3_2':['#40655e','#a2e0dd','#31d0a5'],
	'4_0':["#8de4d3","#0e503e","#43e26d","#2da0a1"],
	'4_1':["#069668","#49edc9","#2d595a","#8dd2d8"],
	'5_0':["#e7b7a5","#da9b83","#b1cdda","#71909e","#325666"],
	'7_0':["#58b5e1","#316387","#40ceae","#285d28","#ade64f","#63a122","#2ce462"],
}

hatches_default=['/','\\','|','-','+','x','o','O','.','*']

continental_us=([1,4,5,6,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,
	27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,
	51,53,54,55,56])

alaska=2
hawaii=15

def ReturnColorMap(colors):

	if type(colors)==str:
		cmap=matplotlib.cm.get_cmap(colors)
	else:
		cmap=LinearSegmentedColormap.from_list('custom',colors,N=256)

	return cmap

def PlotStackedBar(data_dict,figsize=(8,8),cmap=ReturnColorMap('viridis'),ax=None,
	bar_kwargs={},axes_kwargs={},legend_kwargs={},grid_kwargs={},legend_pad=0):

	keys=list(data_dict.keys())
	legend_keys=[key.ljust(legend_pad) for key in keys]
	
	positions=np.arange(0,len(data_dict[keys[0]]))

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	bottom=np.zeros(len(positions))

	handles=[]
	for idx in range(len(keys)):
		values=np.array(data_dict[keys[idx]]).flatten()
		b=ax.bar(positions,values,
			bottom=bottom,color=cmap(idx/(len(keys)+.001)),**bar_kwargs)
		bottom+=values
		handles.append(b)

	ax.legend(handles[::-1],legend_keys[::-1],**legend_kwargs)

	ax.set(**axes_kwargs)

	if return_fig:
		return fig

def SeriesPlot(x,y,labels=[],figsize=(8,8),colors=color_scheme_2_1,ax=None,
	line_kwargs={},axes_kwargs={}):

	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	if labels:
		for idx in range(len(y)):
			color=cmap(np.interp(idx,[0,len(y)],[0,.99]))
			ax.plot(x,y[idx],color=color,label=labels[idx],**line_kwargs)
			ax.legend()
	else:
		for idx in range(len(y)):
			color=cmap(np.interp(idx,[0,len(y)],[0,.99]))
			ax.plot(x,y[idx],color=color,**line_kwargs)

	ax.set(**axes_kwargs)
	ax.grid(ls='--')

	if return_fig:
		return fig

def PlotLine(x,y,figsize=(8,8),colors=color_scheme_2_1,ax=None,
	poly_kwargs={},line_kwargs={},axes_kwargs={}):

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	ax.plot(x,y,**line_kwargs)

	ax.set(**axes_kwargs)

	if return_fig:
		return fig

def PlotContour(x_grid,y_grid,c_values,figsize=(8,8),ax=None,cmap=ReturnColorMap('viridis'),
	contourf_kwargs={},contour_kwargs={},axes_kwargs={},colorbar_kwargs={},grid_kwargs={}):

	return_fig=False
	if ax is None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True
	
	cs=ax.contourf(x_grid,y_grid,c_values,cmap=cmap,**contourf_kwargs)
	if contour_kwargs:
		ax.contour(x_grid,y_grid,c_values,**contour_kwargs)

	vmin=c_values.min()
	vmax=c_values.max()
	sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=vmin, vmax=vmax))
	plt.colorbar(sm,ax=ax,**colorbar_kwargs)

	ax.set(**axes_kwargs)

	if grid_kwargs:
		ax.grid(**grid_kwargs)
	

	if return_fig:
		return fig
	