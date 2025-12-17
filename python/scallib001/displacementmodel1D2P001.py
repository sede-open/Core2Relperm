#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------
#
# 1D2P solver
#   - 2 phase incompressible flow with capillary pressure and gravity
#   - uni-directional flow ONLY, i.e., NO counter-current flow
#   - CPU intensive components compiled with numba
#
#   - Assumptions for units:
#   * pressure in bar
#   * volumetric rate in cm3/minute, volume cm3
#   * density kg/m3
#   * viscosity cP
#   * permeability in mDarcy
#   * length in cm, area in cm2
#   * time in hour
#
# 03.04.2020 HD
# - completed set of columns in tss_table
# - included calculation of total pressure drop
#   i.e., assume linear relperms in block before inlet
# - include PVinj as column in tss_table
# - information about columns in tss_table
#
#--------------------------------------------------------------------------


# Constants of Nature:
gravity_constant = 9.8066  # m/s**2

# Conversion constants to convert to SI units in calculations:

METER = 1.0 # m
CENTIMETER = 0.01 # m
CENTIPOISE = 0.001 # Pa.s
MINUTE = 60.0 # s
HOUR = 3600.0 # s
KILOGRAM = 1.0 # kg
BAR = 1e5 # Pa
DARCY = 0.9869233e-12 # m2
MILLIDARCY = DARCY/1000

import numpy as np
import pandas as pd
import numba
import warnings
import time
from .solver_rate_specified import solve_1D2P_rate_specified
from .solver_centrifuge import solve_1D2P_centrifuge
from .utils import dictn

class DisplacementModel1D2P(object):
    
    def __init__(self,
                 core_length=None, 
                 core_area=None,
                 permeability=None,
                 porosity=None,
                 sw_initial=None,
                 viscosity_w=None,
                 viscosity_n=None,
                 density_w=None,
                 density_n=None,
                 gravity_multiplier=1.0,
                 cpr_multiplier=1.0,
                 time_end=None,
                 rlp_model=None,
                 cpr_model=None,
                 flow_direction="imbibition",
                 centrifuge_rmid=None,
                 rate_schedule=None,
                 acceleration_schedule=None,
                 time_rampup=0.0,
                 ref_acceleration=0.0,
                 movie_schedule=None,
                 NX=50,
                 verbose=0,
                 max_step_size=1.0,
                 min_step_size=1.e-100,
                 max_dsw=0.1,
                 start_step_size=0.001,
                 refine_grid=True,
                 max_nr_iter=25,
                 max_num_step=1000000,
                 step_increment_factor=2.0,
                 step_reduction_factor=2.0,
                 nr_tolerance=None,
                 follow_stops=True,
                ):
        
        self.core_length        = core_length
        self.core_area          = core_area
        self.permeability       = permeability
        self.porosity           = porosity
        self.sw_initial         = sw_initial
        self.viscosity_w        = viscosity_w
        self.viscosity_n        = viscosity_n
        self.density_w          = density_w
        self.density_n          = density_n
        
        self.gravity_multiplier = gravity_multiplier
        self.cpr_multiplier     = cpr_multiplier
        
        self.rlp_model          = rlp_model
        self.cpr_model          = cpr_model
        
        self.NX                 = NX
        self.verbose            = verbose
        self.max_step_size      = max_step_size
        self.min_step_size      = min_step_size
        self.max_dsw            = max_dsw
        self.start_step_size    = start_step_size
        self.refine_grid        = refine_grid
        self.max_nr_iter        = max_nr_iter
        self.max_num_step       = max_num_step
        self.step_increment_factor = step_increment_factor
        self.step_reduction_factor = step_reduction_factor
        self.nr_tolerance       = nr_tolerance
        self.follow_stops       = follow_stops

        self.flow_direction     = flow_direction
        
        if acceleration_schedule is not None:
           self.experiment_type = "centrifuge"
           self.acceleration_schedule = acceleration_schedule
           self.centrifuge_rmid = centrifuge_rmid
           self.time_rampup = time_rampup
           self.ref_acceleration = ref_acceleration
           self.nr_tolerance = 1e-5 if nr_tolerance is None else nr_tolerance
        elif rate_schedule is not None:
           self.experiment_type = "rate_specified"
           self.rate_schedule = rate_schedule
           self.nr_tolerance = 1e-10 if nr_tolerance is None else nr_tolerance
        else:
            raise ValueError('FATAL need to specify rate_schedule or acceleration_schedule.' )

        self.time_end           = time_end
        self.movie_schedule     = movie_schedule 
        
        self.check_input_valid()
        
        self.prepare()
        
    def check_input_valid(self):
        
        # Check that all parameters have value
        invalid = []
        for k,v in self.__dict__.items():
            if v is None:
                invalid.append(k)
        if len(invalid):
            raise ValueError('FATAL these input parameters have no value: \n'+ '\n'.join(invalid) )
    
        if self.experiment_type == "rate_specified":
            self.validate_rate_schedule()
        elif self.experiment_type == "centrifuge":
            self.validate_acceleration_schedule()
            assert self.centrifuge_rmid > 0.0, "centrifuge_rmid must be specified"
            assert self.time_rampup >= 0.0, "time_rampup must be non-negative"
            assert self.ref_acceleration >= 0.0, "ref_acceleration must be non-negative"
            assert self.density_w > self.density_n, "density_w must be larger than density_n"

        assert self.flow_direction in ["imbibition", "drainage"], \
             "flow_direction: expect 'imbibition' or 'drainage' but got "+str(self.flow_direction)
        
    def prepare(self):
        
        self.pore_volume = self.core_length * self.core_area * self.porosity
        
        sim_schedule = self.prepare_sim_schedule( self.movie_schedule )

        self.sim_schedule = sim_schedule

    def validate_rate_schedule(self):
        assert 'StartTime' in self.rate_schedule.columns
        assert 'InjRate'   in self.rate_schedule.columns
        assert np.all( self.rate_schedule.InjRate.values>0 ), 'Injection rate must be non-zero'

    def validate_acceleration_schedule(self):
        assert 'StartTime'    in self.acceleration_schedule.columns
        assert 'Acceleration' in self.acceleration_schedule.columns
        assert np.all( np.abs(self.acceleration_schedule.Acceleration.values)>0 ), 'Acceleration must be non-zero'

    def prepare_sim_schedule( self, time_stops):

        if self.experiment_type == "rate_specified":
           schedule = self.rate_schedule
        else:
           schedule = self.acceleration_schedule
        
        K   = self.permeability * MILLIDARCY
        L   = self.core_length * CENTIMETER
        A   = self.core_area * CENTIMETER**2
        por = self.porosity

        tD_conv = HOUR * CENTIMETER**3/MINUTE/ (L*A*por)

        pres_conv = CENTIMETER**3/MINUTE / A * CENTIPOISE * L / K / BAR

        period_time = schedule.StartTime.values

        sel = period_time < self.time_end
        period_time = np.hstack( [period_time[sel], [self.time_end]] )

        time_stops = np.array( time_stops )

        time_stops = time_stops[ (time_stops>period_time.min()) & (time_stops<period_time.max()) ]

        times = np.unique( np.hstack( [period_time, time_stops] ) )

        df = pd.DataFrame( )

        df['TIME' ] = times
        df['dTIME'] = np.diff( times, append=times[-1])

        b = np.digitize( times, period_time, right=False )

        df['Period'] = b

        if self.experiment_type=="rate_specified":

           period_rate = schedule.InjRate.values
           period_rate = np.hstack( [period_rate[sel], [period_rate[sel][-1]]] )

           df['Rate' ] = period_rate[b-1]

           if 'FracFlow' in schedule.columns:
               n = len(schedule)
               df['FracFlow'] = schedule.FracFlow.values[ np.minimum(b-1,n-1)]
               df['FracFlow'].values[-1] = 1
           else:
               df['FracFlow'] = 1.0

           dtD = df.dTIME.values * df.Rate.values * tD_conv

           df['P_conv'] = 1.0/(df.Rate.values * pres_conv)

        else:

            # Note that we ignore the sign of Acceleration, use flow_direction to 
            # specify drainage or imbibition experiment
            n = len(schedule)
            df['Acceleration'] = np.abs(schedule.Acceleration.values)[np.minimum(b-1,n-1)]

            dtD = df.dTIME.values * tD_conv

            df['P_conv'] = 1.0/pres_conv

        tD  = np.hstack( [[0],dtD[:-1].cumsum()] )

        df['tD' ] = tD
        df['dtD'] = dtD

        return df

    def solve(self):

        if self.experiment_type == "rate_specified":

           self.solve_rate_specified()

        elif self.experiment_type == "centrifuge":

           self.solve_centrifuge()

        else:

            raise ValueError("solver not implemented for experiment_type"+str(self.experiment_type))

        return self

        
    def solve_rate_specified(self):
        
        cpu_solve = time.time()

        sim_schedule = self.sim_schedule
       
        gvh_conv = (gravity_constant * KILOGRAM/METER**3 * CENTIMETER) / BAR

        gvhw_bar = self.density_w * self.core_length * gvh_conv
        gvhn_bar = self.density_n * self.core_length * gvh_conv
        
        solver_result = solve_1D2P_rate_specified( 
            
                N               = int(self.NX), 
                 
                verbose         = int(self.verbose),
                
                follow_stops    = bool(self.follow_stops),
                t_stops         = sim_schedule.tD.values,
                s_stops         = sim_schedule.P_conv.values,
                f_stops         = sim_schedule.FracFlow.values,
                tD_end          = float(sim_schedule.tD.max()),              
                
                start_step_size = float(self.start_step_size),
                max_step_size   = float(self.max_step_size),
                nr_tolerance    = float(self.nr_tolerance),
                max_nr_iter     = int(self.max_nr_iter),
                max_num_step    = int(self.max_num_step),
                step_increment_factor = float(self.step_increment_factor),
                step_reduction_factor = float(self.step_reduction_factor),
                max_dsw         = float(self.max_dsw),

                sw_initial      = float(self.sw_initial),
                rlp_model       = self.rlp_model,
                cpr_model       = self.cpr_model,
                viscosity_w     = float(self.viscosity_w),
                viscosity_n     = float(self.viscosity_n),
                gvhw            = float(gvhw_bar * self.gravity_multiplier),
                gvhn            = float(gvhn_bar * self.gravity_multiplier),
                cpr_multiplier  = float(self.cpr_multiplier),
            
                refine_grid     = bool(self.refine_grid),
        )

        self.results = self.process_solver_result( solver_result )

        self.results.cpu_solve = time.time() - cpu_solve

        return self

    def solve_centrifuge(self):
        
        cpu_solve = time.time()

        is_drainage = True if self.flow_direction == "drainage" else False

        sim_schedule = self.sim_schedule
       
        gvh_conv = (gravity_constant * KILOGRAM/METER**3 * CENTIMETER) / BAR
       
        gvhw_bar = self.density_w * self.core_length * gvh_conv 
        gvhn_bar = self.density_n * self.core_length * gvh_conv 
        
        acceleration = sim_schedule.Acceleration.values / gravity_constant
        
        tD_end = sim_schedule.tD.max()

        time_conv = tD_end / sim_schedule.TIME.max()
        
        solver_result = solve_1D2P_centrifuge( 
            
                N               = int(self.NX), 
                 
                verbose         = int(self.verbose),
                
                follow_stops    = bool(self.follow_stops),
                t_stops         = sim_schedule.tD.values,
                s_stops         = sim_schedule.P_conv.values,
                f_stops         = acceleration,
                tD_end          = float(tD_end),              
                
                start_step_size = float(self.start_step_size),
                max_step_size   = float(self.max_step_size),
                min_step_size   = float(self.min_step_size),
                nr_tolerance    = float(self.nr_tolerance),
                max_nr_iter     = int(self.max_nr_iter),
                max_num_step    = int(self.max_num_step),
                step_increment_factor = float(self.step_increment_factor),
                step_reduction_factor = float(self.step_reduction_factor),
                max_dsw         = float(self.max_dsw),
            
                sw_initial      = float(self.sw_initial),
                rlp_model       = self.rlp_model,
                cpr_model       = self.cpr_model,
                viscosity_w     = float(self.viscosity_w),
                viscosity_n     = float(self.viscosity_n),
                gvhw            = float(gvhw_bar * self.gravity_multiplier),
                gvhn            = float(gvhn_bar * self.gravity_multiplier),
                cpr_multiplier  = float(self.cpr_multiplier),
            
                centrifuge_rmid = float(self.centrifuge_rmid / self.core_length),
                is_drainage     = int(is_drainage),
                t_rampup        = float(self.time_rampup*time_conv),
                ref_acceleration= float(self.ref_acceleration/gravity_constant),

                refine_grid     = bool(self.refine_grid),
        )

        self.results = self.process_solver_result( solver_result )

        self.results.cpu_solve = time.time() - cpu_solve

        return self

    def process_solver_result( self, r ):

        # Unpack solver result
        if self.experiment_type == "centrifuge":
            xD, delxD, movie_tD, movie_dtD, movie_sw, movie_pw, \
            movie_pn, movie_pcw, movie_delp, movie_flxw, movie_flxn, \
            movie_acc, movie_nr, failure_reason = r
        else:
            xD, delxD, movie_tD, movie_dtD, movie_sw, movie_pw, \
            movie_pn, movie_pcw, movie_delp, movie_flxw, movie_nr = r
            failure_reason = ''

        del r

        if len(failure_reason):
             warnings.warn(failure_reason, UserWarning)

        x = xD * self.core_length

        if self.experiment_type == "centrifuge":
           if self.flow_direction == "drainage":
               radius = self.centrifuge_rmid - self.core_length*0.5 + x
           else:
               radius = self.centrifuge_rmid - self.core_length*0.5 + (1-xD)*self.core_length
        else: 
           radius = x

        movie_tD   = np.array ( movie_tD   )
        movie_dtD  = np.array ( movie_dtD  )
        movie_sw   = np.vstack( movie_sw   )
        movie_pw   = np.vstack( movie_pw   )
        movie_pn   = np.vstack( movie_pn   )
        movie_pcw  = np.vstack( movie_pcw  )
        movie_delp = np.vstack( movie_delp )
        movie_flxw = np.vstack( movie_flxw )
        movie_nr   = np.array ( movie_nr   )

        if self.experiment_type == "rate_specified":
           movie_flxn = 1 - movie_flxw
        else:
           movie_flxn = np.vstack( movie_flxn )
           movie_acc  = np.array ( movie_acc  ) * gravity_constant

        movie_swavg = movie_sw @ delxD

        sim_schedule = self.sim_schedule.groupby('Period').first()

        EPS = 1e-9 # make sure value is taken on left when exactly on edge

        movie_period = np.digitize( movie_tD-EPS, sim_schedule.tD.values, right=False )

        movie_period = np.minimum( movie_period, len(sim_schedule) )
        movie_period = np.maximum( movie_period, 1 )

        pore_volume = self.core_length * self.core_area * self.porosity

        if self.experiment_type == "centrifuge":
            # Centrifuge: dynamic injection rates
        
            movie_inj_rate_t = movie_flxw[:,0] + movie_flxn[:,0]
            
            movie_dtime = movie_dtD * pore_volume
            movie_time  = np.cumsum( movie_dtime )

            movie_inj_rate_w = movie_flxw[:, 0]
            movie_inj_rate_n = movie_flxn[:, 0]

            movie_prd_rate_w = movie_flxw[:,-1]
            movie_prd_rate_n = movie_flxn[:,-1]

        else:
            # SS or USS, fixed injection rates
            
            movie_inj_rate_t = sim_schedule.Rate.values[movie_period-1]

            movie_dtime = movie_dtD * pore_volume / movie_inj_rate_t
            movie_time  = np.cumsum( movie_dtime )

            movie_inj_rate_w = (  movie_flxw[:, 0]) * movie_inj_rate_t
            movie_inj_rate_n = (1-movie_flxw[:, 0]) * movie_inj_rate_t

            movie_prd_rate_w = (  movie_flxw[:,-1]) * movie_inj_rate_t
            movie_prd_rate_n = (1-movie_flxw[:,-1]) * movie_inj_rate_t

        movie_inj_volume_w = np.cumsum(movie_inj_rate_w*movie_dtime)
        movie_inj_volume_n = np.cumsum(movie_inj_rate_n*movie_dtime)
        movie_prd_volume_w = np.cumsum(movie_prd_rate_w*movie_dtime)
        movie_prd_volume_n = np.cumsum(movie_prd_rate_n*movie_dtime)

        movie_time  /= 60.0 # in hour
        movie_dtime /= 60.0 # in hour

        tss_table = pd.DataFrame()
        tss_table['TIME'               ] = movie_time  
        tss_table['DTIME'              ] = movie_dtime 
        tss_table['PVinj'              ] = movie_tD
        tss_table['tD'                 ] = movie_tD
        tss_table['dtD'                ] = movie_dtD
        tss_table['InjRate'            ] = movie_inj_rate_t
        tss_table['PVInjRate'          ] = movie_inj_rate_t / self.pore_volume
        tss_table['FracFlowInj'        ] = movie_flxw[:, 0]
        tss_table['FracFlowPrd'        ] = movie_flxw[:,-1]
        tss_table['WATERInj'           ] = movie_inj_rate_w 
        tss_table['OILInj'             ] = movie_inj_rate_n 
        tss_table['WATERProd'          ] = movie_prd_rate_w
        tss_table['OILProd'            ] = movie_prd_rate_n
        tss_table['CumWATERInj'        ] = movie_inj_volume_w
        tss_table['CumOILInj'          ] = movie_inj_volume_n
        tss_table['CumWATER'           ] = movie_prd_volume_w
        tss_table['CumOIL'             ] = movie_prd_volume_n
        tss_table['P_inj'              ] = movie_pw[:, 0] - movie_delp[:,0]
        tss_table['P_inj_wat'          ] = movie_pw[:, 0] 
        tss_table['P_inj_oil'          ] = movie_pn[:, 0] 
        tss_table['P_prod'             ] = movie_pw[:,-1] + movie_delp[:,-1]
        tss_table['P_prod_wat'         ] = movie_pw[:,-1] 
        tss_table['P_prod_oil'         ] = movie_pn[:,-1] 
        tss_table['delta_P'            ] = tss_table.P_inj.values - tss_table.P_prod.values     
        tss_table['delta_P_w'          ] = movie_pw[:,0] - movie_pw[:,-1]
        tss_table['delta_P_o'          ] = movie_pn[:,0] - movie_pn[:,-1]
        tss_table['Sw_avg'             ] = movie_swavg

        if self.experiment_type == "centrifuge":
            tss_table['Acceleration'] = movie_acc

        return dictn(locals())


    def get_tss_table_info( self ):
        return get_tss_table_info()       
 
tss_table_explanation = dict(
    TIME        = 'time [hour], start at 0',
    DTIME       = 'solver time step size taken [hour]',
    PVinj       = 'pore volumes injected [v/v]',
    tD          = 'dimensionless time, equals PVinj',
    dtD         = 'solver dimensionless time step size taken [hour]',
    InjRate     = 'total injection volumetric rate [cm3/min]',
    PVInjRate   = 'total injection volumetric rate measured in pore volumes [1/min]',
    FracFlowInj = 'fractional flow at inlet [v/v]',
    FracFlowPrd = 'fractional flow at outlet [v/v]',
    WATERInj    = 'water injection volumetric rate [cm3/min]',
    OILInj      = 'oil injection volumetric rate [cm3/min]',
    WATERProd   = 'water production volumetric rate [cm3/min]',
    OILProd     = 'oil production volumetric rate [cm3/min]',
    CumWATERInj = 'water injected volume [cm3/]',
    CumOILInj   = 'oil injected volume [cm3]',
    CumWATER    = 'water produced volume [cm3]',
    CumOIL      = 'oil produced volume [cm3]',
    P_inj       = 'pressure at inlet face [bar]',
    P_inj_wat   = 'water phase pressure in first gridblock [bar]',
    P_inj_oil   = 'oil phase pressure in first gridblock [bar]',
    P_prod      = 'pressure at outlet face [bar]',
    P_prod_wat  = 'water phase pressure in last gridblock [bar]',
    P_prod_oil  = 'oil phase pressure in last gridblock [bar]',
    delta_P     = 'pressure drop between inlet and outlet face [bar]',
    delta_P_w   = 'water phase pressure drop [bar]',
    delta_P_o   = 'oil phase pressure drop [bar]',
    Sw_avg      = 'Average water saturation in plug [v/v]',
    Acceleration= 'Acceleration [m/s2] for centrifuge experiment',
)

def get_tss_table_info():
    data = []
    for k,v in tss_table_explanation.items():
        data.append( [k,v])
    return pd.DataFrame( data, columns=['Column','Explanation'])
