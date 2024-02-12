#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------
"""
Utility to extract parameters and results from a SCORES dataset stored
in an Excel spreadsheet. Physical units are converted and data reformated
so that data is ready for use with the 1D2P solver.


Note:
- this utility is constructed to be used in a jupyter notebook and makes use
of IPython to display summary of data (function "print_summary").
- this utility is build to parse the four benchmark cases as provided
in this directory. It may not be sufficient to read other SCORES data files. 

"""

import os
import numpy as np
import pandas as pd


class dictn(dict):
    """A dictionary making use of attributes to acces data."""
    def __getattr__(self,k):
        return self[k]
    def __setattr__(self,k,v):
        self[k] = v

def row_as_float(row):
    def try_float(v):
        try:
           f = float(v)
        except:
           f = np.nan
        return f
    return np.array([try_float(v) for v in row])

def read_table(df,i,col_names):
    num_cols = len(col_names)
    data = []
    i += 1
    l = row_as_float(df.values[i][:num_cols])
    while(i<len(df) and np.isreal(l[0]) and not np.isnan(l[0])):
        data.append( l[:num_cols].astype(float) )
        i += 1
        if i < len(df):
            l = row_as_float(df.values[i][:num_cols])
    return i-1, pd.DataFrame( data, columns=col_names )

def scomp(v, c):
    return str(v).strip() == c


def read_scores_data(directory='', excel_file_name=''):

    file_name = os.path.join(directory, excel_file_name)

    print(f"Read scores data file {excel_file_name} ...")

    if file_name.endswith('.csv'):
       df = pd.read_csv(file_name)
    else:
       df = pd.read_excel(file_name)
    
    scores = dictn()

    scores.excel_file_name = excel_file_name

    i = 0
    n = len(df)
    while(i<n):
        l = df.values[i]
        if scomp(l[0], 'Gridblocks'):
            scores.Gridblocks = int(l[1])
            scores.VertHor = l[4].strip()
            if scores.VertHor == 'H':
               scores.gravity_multiplier = 0.0
            else:
               scores.gravity_multiplier = 1.0
            scores.LevJ = l[7].strip() # ON or OFF, not used
        if scomp(l[0], 'Core_length'):
            scores.Core_length = float(l[1]) # cm
            scores.Core_diam = float(l[4]) # cm
            scores.Core_area = (scores.Core_diam/2.0)**2 * np.pi # cm2
        if scomp(l[0], 'Porosity'):
            scores.Porosity = float(l[1]) # v/v
            scores.Permeability = float(l[4]) # mdarcy
        if scomp(l[0], 'Flooding'):
            scores.Flooding = l[1]
        if scomp(l[0], 'Swi'):
            scores.Swi = float(l[1]) # v/v
        if scomp(l[0], 'Water_density'):
            scores.Water_density = float(l[1])*1000.0 # kg/m3
            scores.Oil_density = float(l[4])*1000.0 # kg/m3 
        if scomp(l[0], 'Water_visc'):
            scores.Water_visc = float(l[1]) # cP
            scores.Oil_visc = float(l[4]) # cP
        if scomp(l[0], 'IFT'):
            scores.IFT = float(l[1]) # N/M, not used
        if scomp(l[0], 'time(HOUR)') and scomp(l[1], 'Flowrate(CM3/HOUR)'):

            i, scores_schedule = read_table(df,i,['time', 'Flowrate'])

            schedule = pd.DataFrame()
            schedule['StartTime'] = scores_schedule.time # hour
            schedule['InjRate'] = scores_schedule.Flowrate / 60.0 # from cm3/hour to cm3/min
            schedule['FracFlow'] = 1.0 # 100% water
            scores.schedule = schedule

        if scomp(l[0], 'time(HOUR)') and scomp(l[1], 'water flowrate(CM3/HOUR)'):

            i, scores_schedule = read_table(df,i,['time', 
                                                  'water flowrate(CM3/HOUR)',
                                                  'oil flowrate(CM3/HOUR)'])
            water_rate = scores_schedule['water flowrate(CM3/HOUR)'] / 60.0 # from cm3/hour to cm3/min
            oil_rate = scores_schedule['oil flowrate(CM3/HOUR)'] / 60.0 # from cm3/hour to cm3/min

            schedule = pd.DataFrame()
            schedule['StartTime'] = scores_schedule.time # hour
            schedule['InjRate'] = water_rate + oil_rate # cm3/min
            schedule['FracFlow'] = water_rate / (water_rate + oil_rate) # v/v
            scores.schedule = schedule

        if scomp(l[0], 'T_end'):
            scores.T_end_HOUR = float(l[1]) # hour
        if scomp(l[0], 'Start_timestep'):
            scores.Start_timestep_SEC = float(l[1]) # seconds
            scores.Max_timestep_MIN = float(l[4]) # minutes
        if scomp(l[0], 'Max_sat_change'):
            scores.Max_sat_change = float(l[1]) # v/v
            scores.Max_incr = float(l[4]) # dimless
        if scomp(l[0], 'Sw') and scomp(l[1], 'krw'):
            i,scores.sw_krw = read_table(df,i,['Sw', 'krw'])
        if scomp(l[0], 'Sw') and scomp(l[1], 'kro'):
            i, scores.sw_kro = read_table(df,i,['Sw', 'kro']) 
        if scomp(l[0], 'Sw') and scomp(l[1], 'Pc'):
            i, scores.sw_pc = read_table(df,i,['Sw', 'Pc'])  # v/v, bar
        if scomp(l[1], 'Time'):
            scores.result_headers = ['INDEX'] + [v.strip() for v in l[1:]]
            i += 1
            l = df.values[i]
            scores.result_units = ['int'] + [v.strip() for v in l[1:]]

            i, scores.result_data = read_table(df,i,scores.result_headers)

            j = 0
            for  u, c in zip(scores.result_units, scores.result_data.columns):
                if u == '(cm3/s)':
                   scores.result_data[c] *= 60 # from cm3/s to cm3/min 
                   scores.result_units[j] = '(cm3/min)'
                if u == '(s)':
                   scores.result_data[c] /= 3600.0 # from s to hour
                   scores.result_units[j] = '(hour)'
                j+=1

            break
        i += 1

    if scores.LevJ != 'OFF':
       raise ValueError("read_scores_data cannot handle LevJ = ON")

    print("   Unit conversions applied on data to change from scores to scallib conventions:")
    print("    - all fluid density data to kg/m3")
    print("    - all flow rate data to cm3/min")
    print("    - all time axis data to hour")
    print("Data converted.")

    def print_summary():
        import IPython

        print('Summary of scores data:')
        print()
        print("Excel file:", scores.excel_file_name)
        print()
    
        for k,v in scores.items():
            if type(v) is int or type(v) is float or type(v) is str:
                print(k,":",v)
        
        print()
        
        for k,v in scores.items():
            if type(v) is pd.core.frame.DataFrame:
                print("Dataframe",k,":")
                with pd.option_context('display.max_rows', 10):
                    IPython.display.display(v)

    scores.print_summary = print_summary
            
    return scores
