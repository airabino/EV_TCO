'''
Default model parameters for TCO calculations

All units are SI base units (mgs) unless otherwise specified
'''

import numpy as np

inputs={
	'model_year':2025, #[model_year]
	'long_itinerary_choice':'charge', #['charge','replace']
	'electric_range':200*1e3, #[m]
	'annual_distance':13250*1609, #[m]
	'ownership_duration':10, #[years]
	'override_utility_factor':0, #[-]
	'override_battery_capacity':0, #[J]
	'override_powertrain_rated_power':0, #[J]
	'override_engine_power':0, #[J]
	'override_motor_power':0, #[J]
}

# Default AER and UF as a function of AER
all_electric_range=np.linspace(0,500,51)*1e3
utility_factor=np.array([
	0,               1.88380155e-01,  3.29034543e-01,  4.33126716e-01,
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
    9.31409944e-01,  9.33026536e-01,  9.34553399e-01,
])

# Battery cost parameters
parameters_battery={
	'reference_cost':120/3.6e6, # [$]
	'reference_year':2024, # [model_year]
	'reference_capacity':45*3.6e6, # [J]
	'annual_decline_factor':0.936, # [-]
	'capacity_scaling_factor':0.0227, # [-]
	'ac_level_2_efficiency':0.85, # [-]
	'dc_level_1_efficiency':0.80, # [-]
	'dc_level_2_efficiency':0.75, # [-]
}

# Engine cost parameters
parameters_engine={
	'reference_year':2018, # [-]
	'c_0':845., # [$]
	'c_1':21.25/1e3, # [$/W]
	'annual_decline_factor':1, # [-]
	'transmission_cost_factor':0.69, # [-]
}

# Motor cost parameters
parameters_motor={
	'reference_year':2018, # [-]
	'c_0':300., # [$]
	'c_1':22.3/1e3, # [$/W]
	'annual_decline_factor':0.99, # [-]
}

# Gearset cost parameters
parameters_gearset={
	'reference_year':2018, # [-]
	'c_0':668.7, # [$]
	'c_1':3.85/1e3, # [$/W]
	'annual_decline_factor':0.99, # [-]
}

# Wiring cost parameters
parameters_wiring={
	'reference_year':2018, # [-]
	'c_0':423., # [$]
	'c_1':0./1e3, # [$/W]
	'annual_decline_factor':0.99, # [-]
}

#Vehicle default parameters

parameters_icev_sedan={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'refuel', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':1, # [-]
	'engine_power_portion':1, # [-]
	'motor_power_portion':0, # [-]
	'powertrain_rated_power':150.*1e3, # [W]
	'consumption_combustion':1/(33/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.29*3.6e3/1.609, # [J/m]
	'all_electric_range':0, # [m]
	'chassis_cost':12700*1.06, # [$]
	'auxiliary_cost_multiplier':0.205, # [-]
	'oem_profit_margin':0.05, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 0., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8739, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.085/1609, # [$/m]
	'maintenance_cost_electric':0.066/1609, # [$/m]
	'replacement_cost': 0.580/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_icev_cuv={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'refuel', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':1, # [-]
	'engine_power_portion':1, # [-]
	'motor_power_portion':0, # [-]
	'powertrain_rated_power':150.*1e3, # [W]
	'consumption_combustion':1/(29/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.345*3.6e3/1.609, # [J/m]
	'all_electric_range':0, # [m]
	'chassis_cost':12700*1.05, # [$]
	'auxiliary_cost_multiplier':0.205, # [-]
	'oem_profit_margin':0.1, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 0., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8739, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.091/1609, # [$/m]
	'maintenance_cost_electric':0.070/1609, # [$/m]
	'replacement_cost': 0.618/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_icev_suv={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'refuel', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':1, # [-]
	'engine_power_portion':1, # [-]
	'motor_power_portion':0, # [-]
	'powertrain_rated_power':220.*1e3, # [W]
	'consumption_combustion':1/(23/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.49*3.6e3/1.609, # [J/m]
	'all_electric_range':0, # [m]
	'chassis_cost':12700*1.21, # [$]
	'auxiliary_cost_multiplier':0.205, # [-]
	'oem_profit_margin':0.15, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 0., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8739, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.096/1609, # [$/m]
	'maintenance_cost_electric':0.074/1609, # [$/m]
	'replacement_cost': 0.652/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_hev_sedan={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'refuel', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':0.76, # [-]
	'engine_power_portion':.9, # [-]
	'motor_power_portion':.3, # [-]
	'powertrain_rated_power':150.*1e3, # [W]
	'consumption_combustion':1/(50/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.29*3.6e3/1.609, # [J/m]
	'all_electric_range':6e3, # [m]
	'chassis_cost':12700*1.06, # [$]
	'auxiliary_cost_multiplier':0.4, # [-]
	'oem_profit_margin':0.05, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 0., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8731, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.085/1609, # [$/m]
	'maintenance_cost_electric':0.066/1609, # [$/m]
	'replacement_cost': 0.580/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_hev_cuv={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'refuel', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':0.76, # [-]
	'engine_power_portion':.9, # [-]
	'motor_power_portion':.3, # [-]
	'powertrain_rated_power':150.*1e3, # [W]
	'consumption_combustion':1/(45/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.345*3.6e3/1.609, # [J/m]
	'all_electric_range':6e3, # [m]
	'chassis_cost':12700*1.05, # [$]
	'auxiliary_cost_multiplier':0.4, # [-]
	'oem_profit_margin':0.1, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 0., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8731, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.091/1609, # [$/m]
	'maintenance_cost_electric':0.070/1609, # [$/m]
	'replacement_cost': 0.618/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_hev_suv={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'refuel', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':0.76, # [-]
	'engine_power_portion':.9, # [-]
	'motor_power_portion':.3, # [-]
	'powertrain_rated_power':220.*1e3, # [W]
	'consumption_combustion':1/(29/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.49*3.6e3/1.609, # [J/m]
	'all_electric_range':6e3, # [m]
	'chassis_cost':12700*1.21, # [$]
	'auxiliary_cost_multiplier':0.4, # [-]
	'oem_profit_margin':0.15, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 0., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8731, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.096/1609, # [$/m]
	'maintenance_cost_electric':0.074/1609, # [$/m]
	'replacement_cost': 0.652/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_phev_sedan={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'refuel', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':0.76, # [-]
	'engine_power_portion':.7, # [-]
	'motor_power_portion':.7, # [-]
	'powertrain_rated_power':150.*1e3, # [W]
	'consumption_combustion':1/(60/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.30*3.6e3/1.609, # [J/m]
	'all_electric_range':40e3, # [m]
	'chassis_cost':12700*1.06, # [$]
	'auxiliary_cost_multiplier':0.4, # [-]
	'oem_profit_margin':0.05, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 0., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8731, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.085/1609, # [$/m]
	'maintenance_cost_electric':0.066/1609, # [$/m]
	'replacement_cost': 0.580/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_phev_cuv={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'refuel', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':0.76, # [-]
	'engine_power_portion':.7, # [-]
	'motor_power_portion':.7, # [-]
	'powertrain_rated_power':150.*1e3, # [W]
	'consumption_combustion':1/(55/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.355*3.6e3/1.609, # [J/m]
	'all_electric_range':40e3, # [m]
	'chassis_cost':12700*1.05, # [$]
	'auxiliary_cost_multiplier':0.4, # [-]
	'oem_profit_margin':0.1, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 0., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8731, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.091/1609, # [$/m]
	'maintenance_cost_electric':0.070/1609, # [$/m]
	'replacement_cost': 0.618/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_phev_suv={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'refuel', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':0.76, # [-]
	'engine_power_portion':.7, # [-]
	'motor_power_portion':.7, # [-]
	'powertrain_rated_power':220.*1e3, # [W]
	'consumption_combustion':1/(50/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.51*3.6e3/1.609, # [J/m]
	'all_electric_range':40e3, # [m]
	'chassis_cost':12700*1.21, # [$]
	'auxiliary_cost_multiplier':0.4, # [-]
	'oem_profit_margin':0.15, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 0., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8731, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.096/1609, # [$/m]
	'maintenance_cost_electric':0.074/1609, # [$/m]
	'replacement_cost': 0.652/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_bev_sedan={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'recharge', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':0.96, # [-]
	'engine_power_portion':0., # [-]
	'motor_power_portion':1., # [-]
	'powertrain_rated_power':150.*1e3, # [W]
	'consumption_combustion':1/(60/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.29*3.6e3/1.609, # [J/m]
	'all_electric_range':300e3, # [m]
	'chassis_cost':12700*1.06, # [$]
	'auxiliary_cost_multiplier':0.4, # [-]
	'oem_profit_margin':0.05, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 1854., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8305, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.085/1609, # [$/m]
	'maintenance_cost_electric':0.066/1609, # [$/m]
	'replacement_cost': 0.580/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_bev_cuv={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'recharge', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':0.96, # [-]
	'engine_power_portion':0., # [-]
	'motor_power_portion':1., # [-]
	'powertrain_rated_power':150.*1e3, # [W]
	'consumption_combustion':1/(55/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.345*3.6e3/1.609, # [J/m]
	'all_electric_range':300e3, # [m]
	'chassis_cost':12700*1.05, # [$]
	'auxiliary_cost_multiplier':0.4, # [-]
	'oem_profit_margin':0.1, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 1854., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8305, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.091/1609, # [$/m]
	'maintenance_cost_electric':0.070/1609, # [$/m]
	'replacement_cost': 0.618/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}

parameters_bev_suv={
	'battery':parameters_battery,
	'engine':parameters_engine,
	'motor':parameters_motor,
	'gearset':parameters_gearset,
	'wiring':parameters_wiring,
	'range_extension':'recharge', # ['refuel','recharge','replace']
	'model_year':2024,
	'ownership_duration':10, # [year]
	'annual_distance':13250*1609, # [m]
	'battery_swing_efficiency':0.96, # [-]
	'engine_power_portion':0., # [-]
	'motor_power_portion':1., # [-]
	'powertrain_rated_power':220.*1e3, # [W]
	'consumption_combustion':1/(50/33.7*1.609)*3600, # [J/m]
	'consumption_electric':0.49*3.6e3/1.609, # [J/m]
	'all_electric_range':300e3, # [m]
	'chassis_cost':12700*1.21, # [$]
	'auxiliary_cost_multiplier':0.4, # [-]
	'oem_profit_margin':0.15, # [-]
	'dealer_profit_margin':0.15, # [-]
	'sales_tax':0.085, # [-]
	'home_charger_cost': 1854., # [$]
	'interpolation_all_electric_range':all_electric_range,
	'interpolation_utility_factor':utility_factor,
	'annual_value_retention':0.8305, # [-]
	'fuel_cost':3.5/33.7/3.6e6, # [$/J]
	'electricity_cost':0.15/3.6e6, # [$/J]
	'electricity_cost_high_rate':0.55/3.6e6, # [$/J]
	'maintenance_cost_combustion':0.096/1609, # [$/m]
	'maintenance_cost_electric':0.074/1609, # [$/m]
	'replacement_cost': 0.652/1609, # [$/m]
	'insurance_annual_cost':887, # [$]
	'registration_annual_cost':285, # [$]
}