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
# - included calculation of total pressure drop, as per MoReS model
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

class dictn(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

@numba.njit
def solve_1D2P_version1(
           N=21, 
           cpr_model=None,
           rlp_model=None,
           cpr_multiplier=1.0,
           viscosity_w=1.0,
           viscosity_n=1.0,
           gvhw = 0.0,
           gvhn = 0.0,
           sw_initial=0,
    
           tD_end=0.2, 
           follow_stops = False,
           t_stops = [], # user imposed stops
           s_stops = [], # scale factors
           f_stops = [], # fractional flows

           max_nr_iter=10, 
           nr_tolerance=1e-3, 
           verbose=1, 

           max_dsw = 0.1,
           max_num_step=1e20, 
           max_step_size = 0.01,
           start_step_size = 0.001,
    
           refine_grid = False,

           reporting = True,

          ):
    '''Stripped 1D2P uni-directional flow solver optimized for compilation with numba.
    
       - 2 phase incompressible flow with capillary pressure and gravity
       - uni-directional flow ONLY, i.e., NO counter-current flow
       - solver uses 1 dummy cell at inlet, will be removed from results
    '''
    
    # Add dummy cell at the inlet
    N1 = N + 1
    
    if refine_grid:
        
        delx = 1.0 / (N1-5-1)
        
        delxi = np.zeros(N1)
        delxi[ 0] = 0.1 * delx # The dummy inlet cell
        delxi[ 1] = 0.1 * delx
        delxi[ 2] = 0.2 * delx
        delxi[ 3] = 0.4 * delx
        delxi[ 4] = 0.8 * delx
        delxi[-1] = 0.1 * delx
        delxi[-2] = 0.2 * delx
        delxi[-3] = 0.4 * delx
        delxi[-4] = 0.8 * delx
        delxi[5:-4] = delx
        
        dtr = np.zeros( N1 )
        dtr[1:-1] = 2.0 / (delxi[2:]+delxi[1:-1])
        dtr[ 0] = 2.0 / delxi[ 1]
        dtr[-1] = 2.0 / delxi[-1]
        
    else:    
        
        delx = 1.0/(N1-1) # First cell is inlet, outlet cell not modelled
        
        delxi = np.ones( N1 ) * delx
        
        dtr = np.ones( N1 )  / delx
        dtr[ 0] = 2.0/delxi[ 0]
        dtr[-1] = 2.0/delxi[-1]
    
    # Mid cell position
    xD = np.zeros(N1)
    xD[0] = -0.5*delxi[0]
    xD[1] =  0.5*delxi[1]
    for i in range(2,N1):
        xD[i] = xD[i-1] + 0.5*(delxi[i-1]+delxi[i])
    
    # Set initial water saturation distribution
    #sw = np.ones(N1,dtype=np.float64)  * sw_initial   trying to fix error in Python 3.11 but that is probalby not the cause
    sw = np.ones(N1).astype(np.float64)  * sw_initial
    
    # Make snapshots at each timestep
    movie_tD     = []
    movie_sw     = []
    movie_dtD    = []
    movie_pw     = []
    movie_pn     = []
    movie_pcw    = []
    movie_delp   = []
    movie_flxw   = []
    movie_nr     = []

    # Initialize simulation time and timestep size   
    t_stops = np.sort( t_stops )
    t_stops = np.unique( t_stops )
    
    timestep = 0
    dtD = start_step_size    
    tD = 0.0
    
    for i_t_stops in range(len(t_stops)):
        if t_stops[i_t_stops]>tD:
            break
    
    dtD, i_t_stops, pres_conv, Fw = update_step( tD, dtD, tD_end, i_t_stops, t_stops, s_stops, f_stops, follow_stops )

    # Pre-allocate arrays for solver
    jac = np.zeros((N1-1,N1-1))
    
    pcw      = np.zeros(N1)
    pcwd     = np.zeros(N1)
    dpcwdx   = np.zeros(N1)
    dpcwdxdu = np.zeros(N1)
    dpcwdxdd = np.zeros(N1)
    
    while tD<tD_end:
      
        if timestep>max_num_step: 
            raise ValueError('FATAL max number of time steps reached - STOP')
            
        sw_prv = sw[1:].copy()

        sw_new = sw[1:]  # note: inlet cell is not active, not solved for

        gvhwD = gvhw * pres_conv
        gvhnD = gvhn * pres_conv
    
        gvhD = gvhwD - gvhnD

        for iter in range(max_nr_iter):
          
            rlpw, rlpn, rlpwd, rlpnd = rlp_model.calc(sw)
            
            pcw, pcwd = cpr_model.calc(sw)
            
            mobw, mobn, mobwd, mobnd = rlpw/viscosity_w, rlpn/viscosity_n, rlpwd/viscosity_w, rlpnd/viscosity_n

            mobt  = mobw  + mobn
            mobtd = mobwd + mobnd
            
            fw    = mobw / mobt
            fwd   = (mobwd * mobt - mobw * mobtd) / mobt**2

            pcw  = pcw  * pres_conv * cpr_multiplier
            pcwd = pcwd * pres_conv * cpr_multiplier
            
            # Derivatives at cell interfaces, convention:
            # - du  derivative to upstream cell
            # - dd  derivative to downstream cell
            
            dpcwdx  [:-1] = (pcw[+1:]-pcw[:-1]) * dtr[:-1]
            dpcwdxdu[:-1] =          -pcwd[:-1] * dtr[:-1]
            dpcwdxdd[:-1] =  pcwd[+1:]          * dtr[:-1]
            
            # At outlet interface: no pc outsize core
            dpcwdx  [-1] = (0-pcw [-1]) * dtr[-1]
            dpcwdxdu[-1] =   -pcwd[-1]  * dtr[-1]            
       
            # flx1: viscous and gravity contribution
            flx1   =  fw  * (1 - mobn*gvhD)
            flx1du =  fwd * (1 - mobn*gvhD) + fw * (-mobnd*gvhD)
 
            # flx2: capillary contribution
            flx2   =  fw  * mobn * dpcwdx
            flx2du =  fwd * mobn * dpcwdx + fw * mobnd * dpcwdx + fw * mobn * dpcwdxdu  
            flx2dd =                                              fw * mobn * dpcwdxdd 
            
            # Impose inflow flux at inlet
            flx1  [0] = Fw
            flx1du[0] = 0
            flx2  [0] = 0
            flx2du[0] = 0
            flx2dd[0] = 0
            
            # Total flux
            flxw   = flx1   + flx2
            flxwdu = flx1du + flx2du
            flxwdd =          flx2dd
            
            if flxw[-1]<0:
                # No backflow at outlet
                flxw   [-1] = 0
                flxwdu [-1] = 0
                flxwdd [-1] = 0
                
            #TODO if np.any( flxw[:-1]<0 ):
            #TODO    raise ValueError('Counter current flow occurred - solver not appropriate STOP')

            rhs = (sw_new - sw_prv)*delxi[1:]/dtD + flxw[1:] - flxw[:-1]

            nr_residual = np.linalg.norm(rhs*dtD/delxi[1:])

            if verbose>1: 
                print('iter,residual',iter,nr_residual)

            if nr_residual < nr_tolerance and iter>0: 
                break
                            
            diag00 = delxi[1:]/dtD + flxwdu[1:  ]  - flxwdd[:-1]                                
            diag10 =                - flxwdu[1:-1] 
            diag01 =                                + flxwdd[1:-1]

            np.fill_diagonal( jac,       diag00 )
            np.fill_diagonal( jac[1:  ], diag10 )
            np.fill_diagonal( jac[:,1:], diag01 )

            d_sw = np.linalg.solve( jac, rhs )

            cutback_factor = np.maximum( np.abs(d_sw).max()/max_dsw, 1.0 )
            
            sw_new -= d_sw/cutback_factor
       

        if nr_residual < nr_tolerance:
            
            # Converged, accept timestep
            
            tD = tD + dtD
            timestep += 1
            
            if reporting:
                
                dpcwdx[0] = (pcw[1]-0.0)*dtr[0]
                
                Sw_inlet = calc_inlet_Sw( Fw, dpcwdx[0], viscosity_w, viscosity_n, gvhwD, gvhnD )*1.0
                
                mobw[0] = (    Sw_inlet)/viscosity_w
                mobn[0] = (1.0-Sw_inlet)/viscosity_n
                mobt[0] = mobw[0] + mobn[0]
                
                delp = -(1 + dpcwdx * mobn + gvhnD * mobn + gvhwD * mobw) / dtr / mobt / pres_conv

                ##delp[0] = -Fw/mobw[1] / dtr[0] / pres_conv
                
                pcw /= pres_conv
                
                # do not use pressure drop over begin/end faces in order to compare to PressureData_SIM  WRONG
                pw = np.zeros( sw.size )
                pw[2:] = np.cumsum( delp[1:-1] )   #TODO
                pw -= pw[-1] - 1.0 + delp[-1] # outer edge at 1 bar
                
                pn = pw + pcw
                
                # Only report cells inside core
                movie_tD    .append(tD         )
                movie_dtD   .append(dtD        )
                movie_sw    .append(sw  [1:].copy())
                movie_delp  .append(delp[0:].copy())
                movie_pw    .append(pw  [1:].copy())
                movie_pn    .append(pn  [1:].copy())
                movie_pcw   .append(pcw [1:].copy())
                movie_flxw  .append(flxw    .copy())
            
                movie_nr.append( iter )
                
            if verbose>0:
                print (timestep,'tD, residual',tD,nr_residual,'dtD',dtD,'nr_iter',iter)
            

            dtD = np.minimum( 2.0*dtD, max_step_size ) #TODO step increment factor
            
            if tD<tD_end:
                dtD, i_t_stops, pres_conv, Fw = \
                    update_step( tD, dtD, tD_end, i_t_stops, t_stops, s_stops, f_stops, follow_stops )
    
        else:
            
            # Not converged, reduce timestepsize and try again
            
            # Reset saturation to previous value
            sw[1:] = sw_prv.copy()
            
            if verbose>0:
                print ('tD, residual,backup',tD,nr_residual,dtD)

            dtD = dtD / 2.0 # TODO step cutback factor
   
    # Drop dummy inlet cell
    xD = xD[1:]
    
    return (xD, movie_tD, movie_dtD, movie_sw, movie_pw, movie_pn, movie_pcw, movie_delp, movie_flxw, movie_nr)

#TODO simplify and clean up

@numba.njit
def update_step( t, delt, t_end, i_t_stops, t_stops, s_stops, f_stops, follow_stops ):
    
    assert len(t_stops)>1
    
    fw = 1
    
    n = len(t_stops)
    
    if i_t_stops < n:
        if follow_stops:
            delt= t_stops[i_t_stops] - t
            if np.abs(delt) < 1e-6:
                if i_t_stops<n-1:
                    i_t_stops += 1
                    delt = t_stops[i_t_stops] - t
                else:
                    delt = t_end - t
            pres_conv = s_stops[i_t_stops-1] # NOTE we take from before last row if t beyoud t_stops!
            fw        = f_stops[i_t_stops-1]
            
            ###print('n tmax i_t_stops t delt',len(t_stops),t_stops[-1],i_t_stops,t,delt)
            
            if delt<1e-19 and i_t_stops<n-1: raise ValueError('delt too small')


        else:
            if t+delt >= t_stops[i_t_stops]:
                delt = t_stops[i_t_stops] - t
                pres_conv = s_stops[i_t_stops-1]
                fw        = f_stops[i_t_stops-1]
                if len(t_stops)<100: print (t, t_stops[i_t_stops], delt, i_t_stops)
                if delt<1e-19: raise ValueError('del_t tiny 2')
                i_t_stops += 1  # FIXME this goes wrong if there is a cutback
    else:
        pres_conv = s_stops[-1]
        
    return delt, i_t_stops, pres_conv, fw

@numba.njit
def calc_inlet_Sw( FW, dPcdxD, vscw, vscn, gvhwD, gvhnD ):
    '''Calculate Sw at inlet set assuming linear relperms at inlet (MoReS procedure)'''
    qw = vscw * FW
    qn = vscn * (1-FW)
    g  = dPcdxD + gvhnD -gvhwD
    
    qtg = qw + qn + g
    
    q = 0.5*(qtg+np.sign(qtg)*np.sqrt( qtg**2 - 4.0*qw*g ))
    
    q = np.float64(q)
    Sw = 0.0
    if qtg>0:
        Sw = qw/q
    else:
        Sw = q/g

    return Sw   

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
                 rate_schedule=None,
                 movie_schedule=None,
                 NX=50,
                 verbose=False,
                 max_step_size=1.0,
                 start_step_size=0.001,
                 refine_grid=True,
                 max_nr_iter=25,
                 nr_tolerance=1e-10,
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
        
        self.time_end           = time_end
        self.rate_schedule      = rate_schedule
        self.movie_schedule     = movie_schedule 
        
        self.NX                 = NX
        self.verbose            = verbose
        self.max_step_size      = max_step_size
        self.start_step_size    = start_step_size
        self.refine_grid        = refine_grid
        self.max_nr_iter        = max_nr_iter
        self.nr_tolerance       = nr_tolerance
        
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
    
        assert 'StartTime' in self.rate_schedule.columns
        assert 'InjRate'   in self.rate_schedule.columns
        
    def prepare(self):
        
        self.pore_volume = self.core_length * self.core_area * self.porosity
        
        sim_schedule = self.prepare_sim_schedule( self.movie_schedule )

        self.sim_schedule = sim_schedule
        
    def prepare_sim_schedule( self, time_stops):

        assert np.all( self.rate_schedule.InjRate.values>0 ), 'Injection rate must be non-zero'
        
        K   = self.permeability * MILLIDARCY
        L   = self.core_length * CENTIMETER
        A   = self.core_area * CENTIMETER**2
        por = self.porosity

        tD_conv = HOUR * CENTIMETER**3/MINUTE/ (L*A*por)

        pres_conv = CENTIMETER**3/MINUTE / A * CENTIPOISE * L / K / BAR

        period_time = self.rate_schedule.StartTime.values
        period_rate = self.rate_schedule.InjRate.values

        sel = period_time < self.time_end
        period_time = np.hstack( [period_time[sel], [self.time_end       ]] )
        period_rate = np.hstack( [period_rate[sel], [period_rate[sel][-1]]] )

        time_stops = np.array( time_stops )

        time_stops = time_stops[ (time_stops>period_time.min()) & (time_stops<period_time.max()) ]

        times = np.unique( np.hstack( [period_time, time_stops] ) )

        df = pd.DataFrame( )

        df['TIME' ] = times
        df['dTIME'] = np.diff( times, append=times[-1])

        b = np.digitize( times, period_time, right=False )

        df['Period'] = b

        df['Rate' ] = period_rate[b-1]

        if 'FracFlow' in self.rate_schedule.columns:
            n = len(self.rate_schedule)
            df['FracFlow'] = self.rate_schedule.FracFlow.values[ np.minimum(b-1,n-1)]
            df['FracFlow'].values[-1] = 1
        else:
            df['FracFlow'] = 1.0

        dtD = df.dTIME.values * df.Rate.values * tD_conv
        tD  = np.hstack( [[0],dtD[:-1].cumsum()] )

        df['tD' ] = tD
        df['dtD'] = dtD
        
        df['P_conv'] = 1.0/(df.Rate.values * pres_conv)

        return df
        
    def solve(self):
        
        sim_schedule = self.sim_schedule
       
        gvh_conv = (gravity_constant * KILOGRAM/METER**3 * CENTIMETER) / BAR

        gvhw_bar = self.density_w * self.core_length * gvh_conv
        gvhn_bar = self.density_n * self.core_length * gvh_conv
        
        solver_result = solve_1D2P_version1( 
            
                N               = self.NX, 
                 
                verbose         = self.verbose,
                
                follow_stops    = True,
                t_stops         = sim_schedule.tD.values,
                s_stops         = sim_schedule.P_conv.values,
                f_stops         = sim_schedule.FracFlow.values,
                tD_end          = sim_schedule.tD.max(),              
                
                start_step_size = self.start_step_size,
                max_step_size   = self.max_step_size,
                nr_tolerance    = self.nr_tolerance,
                max_nr_iter     = self.max_nr_iter,
                max_num_step    = 1e9,

                sw_initial      = self.sw_initial,
                rlp_model       = self.rlp_model,
                cpr_model       = self.cpr_model,
                viscosity_w     = self.viscosity_w,
                viscosity_n     = self.viscosity_n,
                gvhw            = gvhw_bar * self.gravity_multiplier,
                gvhn            = gvhn_bar * self.gravity_multiplier,
                cpr_multiplier  = self.cpr_multiplier,
            
                refine_grid     = self.refine_grid,
        )

        self.results = self.process_solver_result( solver_result )

        return self

    def process_solver_result( self, r ):

        # Unpack solver result
        xD, movie_tD, movie_dtD, movie_sw, movie_pw, movie_pn, movie_pcw, movie_delp, movie_flxw, movie_nr = r

        x = xD * self.core_length

        movie_tD   = np.array ( movie_tD   )
        movie_dtD  = np.array ( movie_dtD  )
        movie_sw   = np.vstack( movie_sw   )
        movie_pw   = np.vstack( movie_pw   )
        movie_pn   = np.vstack( movie_pn   )
        movie_pcw  = np.vstack( movie_pcw  )
        movie_delp = np.vstack( movie_delp )
        movie_flxw = np.vstack( movie_flxw )
        movie_nr   = np.array ( movie_nr   )

        sim_schedule = self.sim_schedule.groupby('Period').first()

        EPS = 1e-9 # make sure value is taken on left when exactly on edge

        movie_period = np.digitize( movie_tD-EPS, sim_schedule.tD.values, right=False )

        movie_period = np.minimum( movie_period, len(sim_schedule) )
        movie_period = np.maximum( movie_period, 1 )

        movie_inj_rate_t = sim_schedule.Rate.values[movie_period-1]

        pore_volume = self.core_length * self.core_area * self.porosity

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
)

def get_tss_table_info():
    data = []
    for k,v in tss_table_explanation.items():
        data.append( [k,v])
    return pd.DataFrame( data, columns=['Column','Explanation'])
