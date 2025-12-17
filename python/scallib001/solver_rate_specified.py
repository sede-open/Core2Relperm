#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

import numpy as np
import numba

@numba.njit
def solve_1D2P_rate_specified(
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
           step_increment_factor = 2.0,
           step_reduction_factor = 2.0,
    
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

                pcw /= pres_conv
                
                pw = np.zeros( sw.size )
                pw[2:] = np.cumsum( delp[1:-1] )
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
            

            dtD = np.minimum( step_increment_factor*dtD, max_step_size )
            
            if tD<tD_end:
                dtD, i_t_stops, pres_conv, Fw = \
                    update_step( tD, dtD, tD_end, i_t_stops, t_stops, s_stops, f_stops, follow_stops )
    
        else:
            
            # Not converged, reduce timestepsize and try again
            
            # Reset saturation to previous value
            sw[1:] = sw_prv.copy()
            
            if verbose>0:
                print ('tD, residual,backup',tD,nr_residual,dtD)

            dtD = dtD / step_reduction_factor
   
    # Drop dummy inlet cell for output:
    xD1 = xD[1:]
    delxD1 = delxi[1:]
    
    return (xD1, delxD1, movie_tD, movie_dtD, movie_sw, movie_pw, movie_pn, movie_pcw, movie_delp, movie_flxw, movie_nr)

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
    '''Calculate Sw at inlet set assuming linear relperms at inlet'''
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

