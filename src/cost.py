import os
import sys
import time
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.interpolate import interp1d

from .utilities import ProgressBar,FullFact

class Vehicle():
	'''
	Stores vehicle parameters and contains functions to compute TCO and other costs
	'''

	def __init__(self,params,inputs={}):

		self.params=params

		for key,val in inputs.items():

			self.params[key]=val

		# print(self.params['all_electric_range'])

		self.Populate()

	def Populate(self):

		self.Intermediates()
		self.CapitalCosts()
		self.OperationalCosts()

		self.costs['total']=(
			self.costs['purchase']+
			self.costs['resale']+
			self.costs['operation'])

	def OperationalCosts(self):

		self.Energy()
		self.Maintenance()
		self.Replacement()
		self.Registration()
		self.Insurance()

		self.costs['other']=(
			self.costs['registration']+
			self.costs['insurance']+
			self.costs['replacement'])

		self.costs['operation']=(
			self.costs['energy']+
			self.costs['maintenance']+
			self.costs['replacement']+
			self.costs['registration']+
			self.costs['insurance'])

	def Insurance(self):

		self.costs['insurance']=(
			self.params['insurance_annual_cost']*
			self.params['ownership_duration'])

	def Registration(self):

		self.costs['registration']=(
			self.params['registration_annual_cost']*
			self.params['ownership_duration'])

	def Replacement(self):

		if self.params['range_extension']=='replace':

			self.costs['replacement']=(
				self.params['annual_distance']*
				self.params['ownership_duration']*
				self.params['replacement_cost']*
				(1-self.params['utility_factor']))

		else:

			self.costs['replacement']=0

	def Maintenance(self):

		self.costs['maintenance']=(
			self.params['annual_distance']*
			self.params['maintenance_cost_combustion']*
			self.params['ownership_duration']*
			(1-self.params['utility_factor'])+
			self.params['annual_distance']*
			self.params['maintenance_cost_electric']*
			self.params['ownership_duration']*
			self.params['utility_factor'])

	def Energy(self):

		if self.params['range_extension']=='refuel':

			self.costs['fuel']=(
				self.params['annual_distance']*
				self.params['consumption_combustion']*
				self.params['fuel_cost']*
				self.params['ownership_duration']*
				(1-self.params['utility_factor']))

			self.costs['electricity']=(
				self.params['annual_distance']*
				self.params['consumption_electric']*
				self.params['electricity_cost']*
				self.params['ownership_duration']*
				self.params['utility_factor'])

		else:

			self.costs['fuel']=0

			self.costs['electricity']=(
				self.params['annual_distance']*
				self.params['consumption_electric']*
				self.params['electricity_cost']*
				self.params['ownership_duration']*
				self.params['utility_factor'])

		if self.params['range_extension']=='recharge':

			self.costs['electricity']+=(
				self.params['annual_distance']*
				self.params['consumption_electric']*
				self.params['electricity_cost_high_rate']*
				self.params['ownership_duration']*
				(1-self.params['utility_factor']))

		self.costs['energy']=(
			self.costs['fuel']+
			self.costs['electricity'])

	def CapitalCosts(self):

		self.costs={}
		self.Components()
		self.Resale()

	def Resale(self):

		self.costs['resale']=-(
			self.costs['purchase']*
			self.params['annual_value_retention']**
			self.params['ownership_duration'])

	def Components(self):

		self.Battery()
		self.Engine()
		self.Motor()
		self.Gearset()
		self.Wiring()
		self.Chassis()
		self.Auxiliary()
		self.Other()

	def Other(self):

		self.costs['components']=(
			self.costs['battery']+
			self.costs['powertrain']+
			self.costs['chassis']+
			self.costs['auxiliary'])

		self.costs['oem_profit']=(
			self.costs['components']*
			self.params['oem_profit_margin'])

		self.costs['dealer_profit']=(
			self.costs['components']*
			self.params['dealer_profit_margin'])

		self.costs['margin']=(
			self.costs['oem_profit']+
			self.costs['dealer_profit'])

		self.costs['msrp']=(
			self.costs['components']+
			self.costs['oem_profit']+
			self.costs['dealer_profit'])

		self.costs['tax']=(
			self.costs['msrp']*
			self.params['sales_tax'])

		self.costs['home_charger']=self.params['home_charger_cost']

		self.costs['purchase']=(
			self.costs['msrp']+
			self.costs['tax']+
			self.costs['home_charger'])

	def Auxiliary(self):

		self.costs['powertrain']=(
			self.costs['engine']+
			self.costs['transmission']+
			self.costs['motor']+
			self.costs['gearset']+
			self.costs['wiring'])

		self.costs['auxiliary']=(
			(self.costs['chassis']+
				self.costs['powertrain']+
				self.costs['battery'])*
			self.params['auxiliary_cost_multiplier'])

	def Chassis(self):

		self.costs['chassis']=self.params['chassis_cost']

	def Wiring(self):

		if self.params['motor_power_portion']>0:

			self.costs['wiring']=(
				self.params['c_0']['wiring']+
				self.params['c_1']['wiring']*
				self.params['powertrain_rated_power']*
				self.params['motor_power_portion'])

		else:

			self.costs['wiring']=0

	def Gearset(self):

		if self.params['motor_power_portion']>0:

			self.costs['gearset']=(
				self.params['c_0']['gearset']+
				self.params['c_1']['gearset']*
				self.params['powertrain_rated_power']*
				self.params['motor_power_portion'])

		else:

			self.costs['gearset']=0

	def Motor(self):

		if self.params['motor_power_portion']>0:

			self.costs['motor']=(
				self.params['c_0']['motor']+
				self.params['c_1']['motor']*
				self.params['powertrain_rated_power']*
				self.params['motor_power_portion'])

		else:

			self.costs['motor']=0

	def Engine(self):

		if self.params['engine_power_portion']>0:

			self.costs['engine']=(
				self.params['c_0']['engine']+
				self.params['c_1']['engine']*
				self.params['powertrain_rated_power']*
				self.params['engine_power_portion'])

			self.costs['transmission']=(
				self.costs['engine']*
				self.params['engine']['transmission_cost_factor'])

		else:

			self.costs['engine']=0

			self.costs['transmission']=0

	def Battery(self):

		reference_cost=(
			self.params['battery']['reference_cost']*
			self.params['battery']['annual_decline_factor']**
			(self.params['model_year']-
				self.params['battery']['reference_year']))

		relative_capacity=(
			(self.params['battery']['reference_capacity']-
				self.params['battery_capacity'])/
			self.params['battery']['reference_capacity'])

		capacity_cost=(
			relative_capacity*
			reference_cost*
			self.params['battery']['capacity_scaling_factor'])

		self.costs['battery']=(
			self.params['battery_capacity']*(
				reference_cost+capacity_cost))


	def Intermediates(self):
		'''
		Compute values for energy consumption,battery capacity, utility factor,
		and component cost constants
		'''

		# Energy Consmuption


		# Battery capacity
		self.params['usable_battery_capacity']=(
			self.params['all_electric_range']*
			self.params['consumption_electric'])

		self.params['battery_capacity']=(
			self.params['usable_battery_capacity']/
			self.params['battery_swing_efficiency'])

		# print(self.params['usable_battery_capacity'],self.params['battery_capacity'])

		#Utility factor
		self.params['utility_factor']=interp1d(
			self.params['interpolation_all_electric_range'],
			self.params['interpolation_utility_factor'],
			fill_value='extrapolate')(
			self.params['all_electric_range'])

		# print(self.params['utility_factor'])

		# Component cost constants
		self.params['c_0']={
			'engine':self.params['engine']['c_0'],
			'motor':self.params['motor']['c_0'],
			'gearset':self.params['gearset']['c_0'],
			'wiring':self.params['wiring']['c_0'],
		}

		self.params['c_1']={
			'engine':(
				self.params['engine']['c_1']*
				self.params['engine']['annual_decline_factor']**
				(self.params['model_year']-self.params['engine']['reference_year'])
				),
			'motor':(
				self.params['motor']['c_1']*
				self.params['motor']['annual_decline_factor']**
				(self.params['model_year']-self.params['motor']['reference_year'])
				),
			'gearset':(
				self.params['gearset']['c_1']*
				self.params['gearset']['annual_decline_factor']**
				(self.params['model_year']-self.params['gearset']['reference_year'])
				),
			'wiring':(
				self.params['wiring']['c_1']*
				self.params['wiring']['annual_decline_factor']**
				(self.params['model_year']-self.params['wiring']['reference_year'])
				),
		}
