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

'''
All units are SI unless spcificed otherwise
'''

default_inputs={
	'year':2025, #[year]
	'chassis_type':'sedan', #['sedan','cuv','suv']
	'powertrain_type':'bev', #['icev','hev','phev','bev']
	'bev_long_itinerary_assumption':'charge', #['charge','replace'']
	'electric_range':200*1e3, #[m]
	'annual_distance':13250*1609, #[m]
	'ownership_duration':10, #[years]
	'override_utility_factor':0, #[-]
	'override_battery_capacity':0, #[J]
	'override_powertrain_rated_power':0, #[J]
	'override_engine_power':0, #[J]
	'override_motor_power':0, #[J]
}

default_params={
	'year':2025, #[year]
	'chassis_type':'sedan', #['sedan','cuv','suv']
	'powertrain_type':'bev', #['icev','hev','phev','bev']
	'bev_long_itinerary_assumption':'charge', #['charge','replace'']
	'electric_range':200*1e3, #[m]
	'annual_distance':13250*1609, #[m]
	'ownership_duration':10, #[years]
	'override_utility_factor':0, #[-]
	'override_battery_capacity':0, #[J]
	'override_powertrain_rated_power':0, #[J]
	'override_engine_power':0, #[J]
	'override_motor_power':0, #[J]
	'reference_battery_cost':273/3.6e6, #[$]
	'reference_battery_year':2016, #[year]
	'reference_battery_capacity':45*3.6e6, #[J]
	'battery_cost_annual_decline_factor':0.936, #[-]
	'battery_cost_capacity_scaling_factor':0.0227, #[-]
	'battery_cost_phev_multiplier':1.182, #[-]
	'ac_level_2_charger_efficiency':0.85, #[-]
	'dc_level_1_charger_efficiency':0.80, #[-]
	'dc_level_2_charger_efficiency':0.75, #[-]
	'hev_battery_swing':0.76, #[-]
	'bev_battery_swing':0.96, #[-]
	'hev_engine_power_portion':0.9, #[-]
	'hev_motor_power_portion':0.3, #[-]
	'phev_engine_power_portion':0.7, #[-]
	'phev_motor_power_portion':0.7, #[-]
	'engine_c_0_2018':845, #[$]
	'engine_c_1_2018':21.25/1e3, #[$/W]
	'transmission_cost_factor':0.69, #[-]
	'motor_c_0_2018':300, #[$]
	'motor_c_1_2018':22.3/1e3, #[$/W]
	'speed_reducer_c_0_2018':668.7, #[$]
	'speed_reducer_c_1_2018':3.85/1e3, #[$/W]
	'charger_and_cable_c_0_2018':423, #[$]
	'charger_and_cable_c_1_2018':0/1e3, #[$/W]
	'conventional_component_cost_annual_decline_factor':1, #[-]
	'motor_component_cost_annual_decline_factor':0.99, #[-]
	'electronics_component_cost_annual_decline_factor':0.99, #[-]
	'powertrain_rated_power_sedan':150*1e3, #[W]
	'powertrain_rated_power_cuv':150*1e3, #[W]
	'powertrain_rated_power_suv':220*1e3, #[W]
	'reference_direct_cost':12700, #[$]
	'sedan_direct_cost_multiplier':1.06, #[-]
	'cuv_direct_cost_multiplier':1.05, #[-]
	'suv_direct_cost_multiplier':1.21, #[-]
	'icev_indirect_cost_multiplier':0.205, #[-]
	'ev_indirect_cost_multiplier':0.4, #[-]
	'oem_sedan_profit_margin':0.05, #[-]
	'oem_cuv_profit_margin':0.1, #[-]
	'oem_suv_profit_margin':0.15, #[-]
	'dealer_markup':0.15, #[-]
	'sales_tax':0.085, #[-]
	'home_charger_cost':1854, #[$]
	'on_peak_electricity_cost':0.55/3.6e6, #[$/J]
	'off_peak_electricity_cost':0.15/3.6e6, #[$/J]
	'gasoline_cost':3.5/33.7/3.6e6, #[$/J]
	'low_rate_charging_premium':0.1, #[-]
	'high_rate_charging_premium':0.75, #[-]

	'icev_sedan_maintenance_cost':0.085/1609, #[$/m]
	'icev_cuv_maintenance_cost':0.091/1609, #[$/m]
	'icev_suv_maintenance_cost':0.096/1609, #[$/m]

	'bev_sedan_maintenance_cost':0.066/1609, #[$/m]
	'bev_cuv_maintenance_cost':0.070/1609, #[$/m]
	'bev_suv_maintenance_cost':0.074/1609, #[$/m]

	'bev_sedan_replacement_cost':0.580/1609, #[$/m]
	'bev_cuv_replacement_cost':0.618/1609, #[$/m]
	'bev_suv_replacement_cost':0.652/1609, #[$/m]

	'icev_annual_value_retention':0.8739, #[-]
	'hev_annual_value_retention':0.8731, #[-]
	'phev_annual_value_retention':0.8731, #[-]
	'bev_annual_value_retention':0.8305, #[-]

	'icev_sedan_consumption_2018':1/(30/33.7*1.609)*3600, #[J/m]
	'icev_sedan_consumption_2030':1/(37/33.7*1.609)*3600, #[J/m]
	'icev_cuv_consumption_2018':1/(26/33.7*1.609)*3600, #[J/m]
	'icev_cuv_consumption_2030':1/(33/33.7*1.609)*3600, #[J/m]
	'icev_suv_consumption_2018':1/(20/33.7*1.609)*3600, #[J/m]
	'icev_suv_consumption_2030':1/(25/33.7*1.609)*3600, #[J/m]

	'hev_sedan_consumption_2018':1/(47/33.7*1.609)*3600, #[J/m]
	'hev_sedan_consumption_2030':1/(56/33.7*1.609)*3600, #[J/m]
	'hev_cuv_consumption_2018':1/(41/33.7*1.609)*3600, #[J/m]
	'hev_cuv_consumption_2030':1/(49/33.7*1.609)*3600, #[J/m]
	'hev_suv_consumption_2018':1/(27/33.7*1.609)*3600, #[J/m]
	'hev_suv_consumption_2030':1/(32/33.7*1.609)*3600, #[J/m]

	'phev_sedan_consumption_cs_2018':1/(56/33.7*1.609)*3600, #[J/m]
	'phev_sedan_consumption_cs_2030':1/(65/33.7*1.609)*3600, #[J/m]
	'phev_cuv_consumption_cs_2018':1/(51/33.7*1.609)*3600, #[J/m]
	'phev_cuv_consumption_cs_2030':1/(60/33.7*1.609)*3600, #[J/m]
	'phev_suv_consumption_cs_2018':1/(45/33.7*1.609)*3600, #[J/m]
	'phev_suv_consumption_cs_2030':1/(54/33.7*1.609)*3600, #[J/m]

	'phev_sedan_consumption_cd_2018':0.31*3.6e3/1.609, #[J/m]
	'phev_sedan_consumption_cd_2030':0.29*3.6e3/1.609, #[J/m]
	'phev_cuv_consumption_cd_2018':0.37*3.6e3/1.609, #[J/m]
	'phev_cuv_consumption_cd_2030':0.34*3.6e3/1.609, #[J/m]
	'phev_suv_consumption_cd_2018':0.53*3.6e3/1.609, #[J/m]
	'phev_suv_consumption_cd_2030':0.49*3.6e3/1.609, #[J/m]

	'bev_sedan_consumption_2018':0.3*3.6e3/1.609, #[J/m]
	'bev_sedan_consumption_2030':0.28*3.6e3/1.609, #[J/m]
	'bev_cuv_consumption_2018':0.36*3.6e3/1.609, #[J/m]
	'bev_cuv_consumption_2030':0.33*3.6e3/1.609, #[J/m]
	'bev_suv_consumption_2018':0.51*3.6e3/1.609, #[J/m]
	'bev_suv_consumption_2030':0.47*3.6e3/1.609, #[J/m]

	'annual_registration':285, #[$]
	'annual_insurance':887, #[$]

	'default_utility_factor_interpolation_aer':np.linspace(0,500,51)*1e3, #[m]
	'default_utility_factor_interpolation_uf':np.array([
		1.62585037e-05,  1.88380155e-01,  3.29034543e-01,  4.33126716e-01,
        5.10764933e-01,  5.69675365e-01,  6.15252789e-01,  6.51642251e-01,
        6.81295321e-01,  7.05857152e-01,  7.26565521e-01,  7.44346224e-01,
        7.59819247e-01,  7.73386535e-01,  7.85496096e-01,  7.96390907e-01,
        8.06223731e-01,  8.15186436e-01,  8.23403313e-01,  8.31016730e-01,
        8.38039472e-01,  8.44545766e-01,  8.50579257e-01,  8.56166818e-01,
        8.61402271e-01,  8.66320952e-01,  8.70946184e-01,  8.75312433e-01,
        8.79432354e-01,  8.83323436e-01,  8.87017408e-01,  8.90523434e-01,
        8.93893887e-01,  8.97105504e-01,  9.00160929e-01,  9.03081889e-01,
        9.05850682e-01,  9.08506527e-01,  9.11036722e-01,  9.13445122e-01,
        9.15773278e-01,  9.18014140e-01,  9.20173861e-01,  9.22246663e-01,
        9.24229987e-01,  9.26138617e-01,  9.27966688e-01,  9.29722524e-01,
        9.31409944e-01,  9.33026536e-01,  9.34553399e-01]),
}

class VehicleTCO():

	def __init__(self,inputs,params):

		self.params=params

		for key in inputs.keys():
			self.params[key]=inputs[key]

		self.Compute()

	def ParameterSweep(self,params,value_arrays,output='total'):

		output_array=np.empty([arr.size for arr in value_arrays])

		indices=FullFact([arr.size for arr in value_arrays]).astype(int)

		levels=[]
		for idx in range(len(params)):
			levels.append(value_arrays[idx][indices[:,idx]])

		levels=np.vstack((levels)).T

		for idx in range(levels.shape[0]):
			for idx1 in range(len(params)):

				self.params[params[idx1]]=levels[idx,idx1]

			self.Compute()

			output_array[tuple(indices[idx])]=self.costs[output]

		return output_array

	def Compute(self):

		self.ComputeIntermediates()
		self.CapitalCosts()
		self.OperationalCosts()
		# print(self.intermediates['utility_factor'],self.intermediates['consumption'],
		# 	self.costs['gasoline_annual'],self.costs['electricity_annual'])

		self.costs['total']=(
			self.costs['purchase']+
			self.costs['resale']+
			self.costs['operation']
			)

	def OperationalCosts(self):

		self.EnergyCosts()
		self.MaintenanceCosts()
		self.ReplacementCosts()
		self.RegistrationCosts()
		self.InsuranceCosts()

		self.costs['operation_annual']=(
			self.costs['energy_annual']+
			self.costs['maintenance_annual']+
			self.costs['replacement_annual']+
			self.costs['registration_annual']+
			self.costs['insurance_annual']
			)

		self.costs['operation']=(
			self.costs['energy']+
			self.costs['maintenance']+
			self.costs['replacement']+
			self.costs['registration']+
			self.costs['insurance']
			)

	def InsuranceCosts(self):

		self.costs['insurance_annual']=self.params['annual_insurance']
		self.costs['insurance']=self.costs['insurance_annual']*self.params['ownership_duration']

	def RegistrationCosts(self):

		self.costs['registration_annual']=self.params['annual_registration']
		self.costs['registration']=self.costs['registration_annual']*self.params['ownership_duration']

	def ReplacementCosts(self):

		self.costs['replacement_annual']=0

		if self.params['powertrain_type']=='bev':
			if self.params['bev_long_itinerary_assumption']=='replace':

				if self.params['chassis_type']=='sedan':
					self.costs['replacement_annual']=(
						self.params['annual_distance']*(1-self.intermediates['utility_factor'])*
						self.params['bev_sedan_replacement_cost'])

				elif self.params['chassis_type']=='cuv':
					self.costs['replacement_annual']=(
						self.params['annual_distance']*(1-self.intermediates['utility_factor'])*
						self.params['bev_cuv_replacement_cost'])

				elif self.params['chassis_type']=='suv':
					self.costs['replacement_annual']=(
						self.params['annual_distance']*(1-self.intermediates['utility_factor'])*
						self.params['bev_suv_replacement_cost'])

		self.costs['replacement']=self.costs['replacement_annual']*self.params['ownership_duration']

	def MaintenanceCosts(self):

		if self.params['powertrain_type']=='icev':
			if self.params['chassis_type']=='sedan':
				self.costs['maintenance_annual']=(
					self.params['annual_distance']*self.params['icev_sedan_maintenance_cost'])
			elif self.params['chassis_type']=='cuv':
				self.costs['maintenance_annual']=(
					self.params['annual_distance']*self.params['icev_cuv_maintenance_cost'])
			elif self.params['chassis_type']=='suv':
				self.costs['maintenance_annual']=(
					self.params['annual_distance']*self.params['icev_suv_maintenance_cost'])

		elif self.params['powertrain_type']=='hev':
			if self.params['chassis_type']=='sedan':
				self.costs['maintenance_annual']=((1-self.intermediates['utility_factor'])*
					self.params['annual_distance']*self.params['icev_sedan_maintenance_cost']+
					self.intermediates['utility_factor']*
					self.params['annual_distance']*self.params['bev_sedan_maintenance_cost'])
			if self.params['chassis_type']=='cuv':
				self.costs['maintenance_annual']=((1-self.intermediates['utility_factor'])*
					self.params['annual_distance']*self.params['icev_cuv_maintenance_cost']+
					self.intermediates['utility_factor']*
					self.params['annual_distance']*self.params['bev_cuv_maintenance_cost'])
			if self.params['chassis_type']=='suv':
				self.costs['maintenance_annual']=((1-self.intermediates['utility_factor'])*
					self.params['annual_distance']*self.params['icev_suv_maintenance_cost']+
					self.intermediates['utility_factor']*
					self.params['annual_distance']*self.params['bev_suv_maintenance_cost'])

		elif self.params['powertrain_type']=='phev':
			if self.params['chassis_type']=='sedan':
				self.costs['maintenance_annual']=((1-self.intermediates['utility_factor'])*
					self.params['annual_distance']*self.params['icev_sedan_maintenance_cost']+
					self.intermediates['utility_factor']*
					self.params['annual_distance']*self.params['bev_sedan_maintenance_cost'])
			if self.params['chassis_type']=='cuv':
				self.costs['maintenance_annual']=((1-self.intermediates['utility_factor'])*
					self.params['annual_distance']*self.params['icev_cuv_maintenance_cost']+
					self.intermediates['utility_factor']*
					self.params['annual_distance']*self.params['bev_cuv_maintenance_cost'])
			if self.params['chassis_type']=='suv':
				self.costs['maintenance_annual']=((1-self.intermediates['utility_factor'])*
					self.params['annual_distance']*self.params['icev_suv_maintenance_cost']+
					self.intermediates['utility_factor']*
					self.params['annual_distance']*self.params['bev_suv_maintenance_cost'])

		elif self.params['powertrain_type']=='bev':
			if self.params['chassis_type']=='sedan':
				self.costs['maintenance_annual']=(
					self.params['annual_distance']*self.params['bev_sedan_maintenance_cost'])
			elif self.params['chassis_type']=='cuv':
				self.costs['maintenance_annual']=(
					self.params['annual_distance']*self.params['bev_cuv_maintenance_cost'])
			elif self.params['chassis_type']=='suv':
				self.costs['maintenance_annual']=(
					self.params['annual_distance']*self.params['bev_suv_maintenance_cost'])

		self.costs['maintenance']=self.costs['maintenance_annual']*self.params['ownership_duration']

	def EnergyCosts(self):

		if self.params['powertrain_type']=='icev':
			self.costs['gasoline_annual']=(self.params['annual_distance']*
				self.intermediates['consumption']*self.params['gasoline_cost'])
			self.costs['electricity_annual']=0

		elif self.params['powertrain_type']=='hev':
			self.costs['gasoline_annual']=(
				self.params['annual_distance']*(1-self.intermediates['utility_factor'])*
				self.intermediates['consumption']*self.params['gasoline_cost'])
			self.costs['electricity_annual']=(
				self.params['annual_distance']*self.intermediates['utility_factor']*
				self.intermediates['consumption']*self.params['off_peak_electricity_cost']*
				(1+self.params['low_rate_charging_premium']))

		elif self.params['powertrain_type']=='phev':
			self.costs['gasoline_annual']=(
				self.params['annual_distance']*(1-self.intermediates['utility_factor'])*
				self.intermediates['consumption_cs']*self.params['gasoline_cost'])
			self.costs['electricity_annual']=(
				self.params['annual_distance']*self.intermediates['utility_factor']*
				self.intermediates['consumption']*self.params['off_peak_electricity_cost']*
				(1+self.params['low_rate_charging_premium']))

		elif self.params['powertrain_type']=='bev':
			if self.params['bev_long_itinerary_assumption']=='charge':
				self.costs['gasoline_annual']=0
				self.costs['electricity_annual']=(
					self.params['annual_distance']*self.intermediates['utility_factor']*
					self.intermediates['consumption']*self.params['off_peak_electricity_cost']*
					(1+self.params['low_rate_charging_premium'])+
					self.params['annual_distance']*(1-self.intermediates['utility_factor'])*
					self.intermediates['consumption']*self.params['on_peak_electricity_cost']*
					(1+self.params['high_rate_charging_premium']))
			else:
				self.costs['gasoline_annual']=(
				self.params['annual_distance']*(1-self.intermediates['utility_factor'])*
				self.intermediates['consumption_rep']*self.params['gasoline_cost'])
				self.costs['electricity_annual']=(
					self.params['annual_distance']*self.intermediates['utility_factor']*
					self.intermediates['consumption']*self.params['off_peak_electricity_cost']*
					(1+self.params['low_rate_charging_premium']))

		self.costs['energy_annual']=self.costs['gasoline_annual']+self.costs['electricity_annual']
		self.costs['energy']=self.costs['energy_annual']*self.params['ownership_duration']

	def AssignConsumption(self):

		if self.params['chassis_type']=='sedan':

			if self.params['powertrain_type']=='icev':
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['icev_sedan_consumption_2018'],
					self.params['icev_sedan_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))

			elif self.params['powertrain_type']=='hev':
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['hev_sedan_consumption_2018'],
					self.params['hev_sedan_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))

			elif self.params['powertrain_type']=='phev':
				self.intermediates['consumption_cs']=float(interp1d([2018,2030],
					[self.params['phev_sedan_consumption_cs_2018'],
					self.params['phev_sedan_consumption_cs_2030']],
					fill_value='extrapolate')(self.params['year']))
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['phev_sedan_consumption_cd_2018'],
					self.params['phev_sedan_consumption_cd_2030']],
					fill_value='extrapolate')(self.params['year']))

			elif self.params['powertrain_type']=='bev':
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['bev_sedan_consumption_2018'],
					self.params['bev_sedan_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))
				self.intermediates['consumption_rep']=float(interp1d([2018,2030],
					[self.params['icev_sedan_consumption_2018'],
					self.params['icev_sedan_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))

		if self.params['chassis_type']=='cuv':

			if self.params['powertrain_type']=='icev':
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['icev_cuv_consumption_2018'],
					self.params['icev_cuv_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))

			elif self.params['powertrain_type']=='hev':
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['hev_cuv_consumption_2018'],
					self.params['hev_cuv_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))

			elif self.params['powertrain_type']=='phev':
				self.intermediates['consumption_cs']=float(interp1d([2018,2030],
					[self.params['phev_cuv_consumption_cs_2018'],
					self.params['phev_cuv_consumption_cs_2030']],
					fill_value='extrapolate')(self.params['year']))
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['phev_cuv_consumption_cd_2018'],
					self.params['phev_cuv_consumption_cd_2030']],
					fill_value='extrapolate')(self.params['year']))

			elif self.params['powertrain_type']=='bev':
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['bev_cuv_consumption_2018'],
					self.params['bev_cuv_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))
				self.intermediates['consumption_rep']=float(interp1d([2018,2030],
					[self.params['icev_cuv_consumption_2018'],
					self.params['icev_cuv_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))

		if self.params['chassis_type']=='suv':

			if self.params['powertrain_type']=='icev':
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['icev_suv_consumption_2018'],
					self.params['icev_suv_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))

			elif self.params['powertrain_type']=='hev':
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['hev_suv_consumption_2018'],
					self.params['hev_suv_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))

			elif self.params['powertrain_type']=='phev':
				self.intermediates['consumption_cs']=float(interp1d([2018,2030],
					[self.params['phev_suv_consumption_cs_2018'],
					self.params['phev_suv_consumption_cs_2030']],
					fill_value='extrapolate')(self.params['year']))
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['phev_suv_consumption_cd_2018'],
					self.params['phev_suv_consumption_cd_2030']],
					fill_value='extrapolate')(self.params['year']))

			elif self.params['powertrain_type']=='bev':
				self.intermediates['consumption']=float(interp1d([2018,2030],
					[self.params['bev_suv_consumption_2018'],
					self.params['bev_suv_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))
				self.intermediates['consumption_rep']=float(interp1d([2018,2030],
					[self.params['icev_suv_consumption_2018'],
					self.params['icev_suv_consumption_2030']],
					fill_value='extrapolate')(self.params['year']))

	def ComputeIntermediates(self):

		self.intermediates={}

		#Powertrain sizing
		#Powertrain rated power [W]
		if ~self.params['override_powertrain_rated_power']:
			if self.params['chassis_type']=='sedan':
				self.intermediates['powertrain_rated_power']=self.params['powertrain_rated_power_sedan']
			elif self.params['chassis_type']=='suv':
				self.intermediates['powertrain_rated_power']=self.params['powertrain_rated_power_suv']

		#Engine power [W]
		if self.params['override_engine_power']:
			self.intermediates['engine_power']=self.params['override_engine_power']
		else:
			if self.params['powertrain_type']=='icev':
				engine_portion=1
			elif self.params['powertrain_type']=='hev':
				engine_portion=self.params['hev_engine_power_portion']
			elif self.params['powertrain_type']=='phev':
				engine_portion=self.params['phev_engine_power_portion']
			else:
				engine_portion=0

			self.intermediates['engine_power']=(
				self.intermediates['powertrain_rated_power']*engine_portion)

		#Motor power [W]
		if self.params['override_motor_power']:
			self.intermediates['motor_power']=self.params['override_motor_power']
		else:
			if self.params['powertrain_type']=='icev':
				motor_portion=0
			elif self.params['powertrain_type']=='hev':
				motor_portion=self.params['hev_motor_power_portion']
			elif self.params['powertrain_type']=='phev':
				motor_portion=self.params['phev_motor_power_portion']
			else:
				motor_portion=1

			self.intermediates['motor_power']=(
				self.intermediates['powertrain_rated_power']*motor_portion)

		#Battery capacity [J]
		if self.params['powertrain_type']=='icev':
				self.params['electric_range']=0
		if self.params['override_battery_capacity']:
			self.intermediates['battery_capacity']=self.params['override_battery_capacity']
		else:
			self.AssignConsumption()

			if self.params['powertrain_type']=='bev':
				battery_swing=self.params['bev_battery_swing']
			else:
				battery_swing=self.params['hev_battery_swing']

			self.intermediates['usable_battery_capacity']=(
				self.params['electric_range']*self.intermediates['consumption'])
			self.intermediates['battery_capacity']=(
				self.intermediates['usable_battery_capacity']/battery_swing)

		#Powertrain component costs
		years_exponent=self.params['year']-self.params['reference_battery_year']

		self.intermediates['engine_c_0']=self.params['engine_c_0_2018']*(
			self.params['conventional_component_cost_annual_decline_factor']**years_exponent)
		self.intermediates['engine_c_1']=self.params['engine_c_1_2018']*(
			self.params['conventional_component_cost_annual_decline_factor']**years_exponent)

		self.intermediates['motor_c_0']=self.params['motor_c_0_2018']*(
			self.params['motor_component_cost_annual_decline_factor']**years_exponent)
		self.intermediates['motor_c_1']=self.params['motor_c_1_2018']*(
			self.params['motor_component_cost_annual_decline_factor']**years_exponent)

		self.intermediates['speed_reducer_c_0']=self.params['speed_reducer_c_0_2018']*(
			self.params['electronics_component_cost_annual_decline_factor']**years_exponent)
		self.intermediates['speed_reducer_c_1']=self.params['speed_reducer_c_1_2018']*(
			self.params['electronics_component_cost_annual_decline_factor']**years_exponent)

		self.intermediates['charger_and_cable_c_0']=self.params['charger_and_cable_c_0_2018']*(
			self.params['electronics_component_cost_annual_decline_factor']**years_exponent)
		self.intermediates['charger_and_cable_c_1']=self.params['charger_and_cable_c_1_2018']*(
			self.params['electronics_component_cost_annual_decline_factor']**years_exponent)

		#Utility factor
		if self.params['override_utility_factor']:
			self.intermediates['utility_factor']=self.params['override_utility_factor']
		else:
			self.intermediates['utility_factor']=interp1d(
				self.params['default_utility_factor_interpolation_aer'],
				self.params['default_utility_factor_interpolation_uf'],
				fill_value='extrapolate')(self.params['electric_range'])

	def CapitalCosts(self):

		self.costs={}
		self.ComponentCosts()
		self.ResaleValue()

	def ResaleValue(self):

		if self.params['powertrain_type']=='icev':
			self.costs['resale']=-(self.costs['purchase']*
				self.params['icev_annual_value_retention']**self.params['ownership_duration'])

		elif self.params['powertrain_type']=='hev':
			self.costs['resale']=-(self.costs['purchase']*
				self.params['hev_annual_value_retention']**self.params['ownership_duration'])

		elif self.params['powertrain_type']=='phev':
			self.costs['resale']=-(self.costs['purchase']*
				self.params['phev_annual_value_retention']**self.params['ownership_duration'])

		elif self.params['powertrain_type']=='bev':
			self.costs['resale']=-(self.costs['purchase']*
				self.params['bev_annual_value_retention']**self.params['ownership_duration'])

	def ComponentCosts(self):

		self.BatteryPackCost()
		self.EngineCost()
		self.MotorCost()
		self.TransmissionCost()
		self.SpeedReducerCost()
		self.ChargerAndCableCost()
		self.DirectCost()
		self.IndirectCost()
		self.OtherCosts()

	def OtherCosts(self):

		self.costs['components']=(
			self.costs['battery_pack']+
			self.costs['powertrain']+
			self.costs['direct']+
			self.costs['indirect'])

		if self.params['chassis_type']=='sedan':
			self.costs['profit']=self.costs['components']*self.params['oem_sedan_profit_margin']
		elif self.params['chassis_type']=='cuv':
			self.costs['profit']=self.costs['components']*self.params['oem_cuv_profit_margin']
		elif self.params['chassis_type']=='suv':
			self.costs['profit']=self.costs['components']*self.params['oem_suv_profit_margin']

		self.costs['markup']=(
			self.costs['components']+self.costs['profit'])*self.params['dealer_markup']

		self.costs['msrp']=self.costs['components']+self.costs['profit']+self.costs['markup']

		self.costs['tax']=self.costs['msrp']*self.params['sales_tax']

		if self.params['powertrain_type']=='icev':
			self.costs['home_charger']=0
		else:
			self.costs['home_charger']=self.params['home_charger_cost']

		self.costs['purchase']=self.costs['msrp']+self.costs['tax']+self.costs['home_charger']

	def DirectCost(self):

		if self.params['chassis_type']=='sedan':
			self.costs['direct']=(
				self.params['reference_direct_cost']*self.params['sedan_direct_cost_multiplier'])
		elif self.params['chassis_type']=='cuv':
			self.costs['direct']=(
				self.params['reference_direct_cost']*self.params['cuv_direct_cost_multiplier'])
		elif self.params['chassis_type']=='suv':
			self.costs['direct']=(
				self.params['reference_direct_cost']*self.params['suv_direct_cost_multiplier'])
	
	def IndirectCost(self):

		self.costs['powertrain']=(
			self.costs['engine']+
			self.costs['transmission']+
			self.costs['motor']+
			self.costs['speed_reducer']+
			self.costs['charger_and_cable'])

		if self.params['powertrain_type']=='icev':
			self.costs['indirect']=(
				(self.costs['battery_pack']+self.costs['powertrain']+self.costs['direct'])*
				self.params['icev_indirect_cost_multiplier'])
		else:
			self.costs['indirect']=(
				(self.costs['battery_pack']+self.costs['powertrain']+self.costs['direct'])*
				self.params['ev_indirect_cost_multiplier'])

	def EngineCost(self):

		if self.intermediates['engine_power']:
			self.costs['engine']=self.intermediates['engine_c_0']+(
				self.intermediates['engine_c_1']*self.intermediates['engine_power'])
		else:
			self.costs['engine']=0

	def TransmissionCost(self):

		self.costs['transmission']=(
			self.params['transmission_cost_factor']*self.costs['engine'])

	def MotorCost(self):

		if self.intermediates['motor_power']:
			self.costs['motor']=self.intermediates['motor_c_0']+(
				self.intermediates['motor_c_1']*self.intermediates['motor_power'])
		else:
			self.costs['motor']=0

	def SpeedReducerCost(self):

		if self.intermediates['motor_power']:
			self.costs['speed_reducer']=self.intermediates['speed_reducer_c_0']+(
				self.intermediates['speed_reducer_c_1']*self.intermediates['motor_power'])
		else:
			self.costs['speed_reducer']=0

	def ChargerAndCableCost(self):
		
		if self.intermediates['motor_power']:
			self.costs['charger_and_cable']=self.intermediates['charger_and_cable_c_0']+(
				self.intermediates['charger_and_cable_c_1']*self.intermediates['motor_power'])
		else:
			self.costs['charger_and_cable']=0

	def BatteryPackCost(self):

		cost_multiplier_exponent=self.params['year']-self.params['reference_battery_year']
		cost_multiplier=self.params['battery_cost_annual_decline_factor']**cost_multiplier_exponent
		reference_cost=self.params['reference_battery_cost']*cost_multiplier

		relative_capacity=(self.params['reference_battery_capacity']-
			self.intermediates['battery_capacity'])/self.params['reference_battery_capacity']
		capacity_cost=(relative_capacity*self.params['battery_cost_capacity_scaling_factor']*
			reference_cost)

		cost_per_kwh=reference_cost+capacity_cost
		
		self.costs['battery_pack']=self.intermediates['battery_capacity']*cost_per_kwh

