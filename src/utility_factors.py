import os
import sys
import time
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.interpolate import interp1d

from .utilities import ProgressBar,FullFact

def DownSelect(itineraries,conditions):

	keep=[True]*len(itineraries)

	for idx,itinerary in enumerate(itineraries):

		for condition in conditions:

			lhs=itinerary['veh'][condition[0]]
			conditional=condition[1]
			rhs=condition[2]

			keep[idx]=eval(f'{lhs}{conditional}{rhs}')

	return itineraries[keep]

def RemoveNone(itineraries):

	return [itinerary for itinerary in itineraries if itinerary is not None]

def KeepRandom(itineraries,n):

	return np.random.choice(itineraries,size=n,replace=False)

def UtilityFactor(
	itineraries,
	veh_range,
	charge_events=[],
	conditions=[],
	max_itineraries=np.inf,
	freq=1e3,
	disp=False):
	
	'''
	Calculating aggregate utility factors for a population of vehicles with defined
	itineraries
	'''

	itineraries=RemoveNone(itineraries)
	
	# Down-selecting itineraries based on conditions
	if conditions:
		itineraries=DownSelect(np.array(itineraries),conditions)

	if len(itineraries)>max_itineraries:
		itineraries=KeepRandom(itineraries,max_itineraries)

	# Initializing loop sums for total distance driven and total electric distance driven
	sum_dist=0
	sum_dist_aer=np.zeros(len(veh_range))

	# Looping in itineraries
	for idx in ProgressBar(range(len(itineraries)),freq=freq,disp=disp):

		# Converting itinerary dataframe columns to ndarrays
		distances=itineraries[idx]['trips']['TRPMILES'].to_numpy()*1.609
		types=itineraries[idx]['trips']['WHYFROM'].to_numpy()

		# Initializing the trip chain distance
		trip_chain_dist=0

		# Looping on itinerary trips
		for idx,trip in enumerate(distances):

			# Determining if the origin location for the current trip is a charge location
			# if so a new trip chain is started
			if np.isin(types[idx],charge_events):

				# Adding the previous trip chain information to the outer loop sums
				sum_dist+=trip_chain_dist
				sum_dist_aer+=np.min(np.vstack(
					(veh_range,np.ones_like(veh_range)*trip_chain_dist)),axis=0)

				# Starting the new trip chain with the current trip distance
				trip_chain_dist=trip

			else:

				# Adding current trip distance to trip chain
				trip_chain_dist+=trip

		# Adding the trip information for the last itinerary trip chain to the outer
		# loop sums
		sum_dist+=trip_chain_dist
		sum_dist_aer+=np.min(np.vstack(
			(veh_range,np.ones_like(veh_range)*trip_chain_dist)),axis=0)

	return sum_dist_aer/sum_dist