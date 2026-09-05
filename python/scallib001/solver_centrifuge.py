#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

import numpy as np
import math
import numba

@numba.njit
def check_finite_matrix(a):
    for v in np.nditer(a):
        if not np.isfinite(v.item()):
            raise np.linalg.LinAlgError(
                "Array must not contain infs or NaNs.")

@numba.njit
def lusolve2(A,B,C,d):
    '''
    Linear solver using Thomas algorithm for tri-diagonal 2x2 block system
    B0 x0 + C0 x1                 = d0
    A0 x0 + B1 x1 + C1 x2         = d1
            A1 x1 + B2 x2 + C2 x3 = d2
                    ...           = ..

    '''
    n = B.shape[0]
    
    b = np.zeros( B.shape )
    y = np.zeros( d.shape )
    
    bi = np.zeros( B.shape )

    b[0] = B[0]
    y[0] = d[0]

    r1 = np.zeros((2,2))
    r2 = np.zeros((2))
    r3 = np.zeros((2))

    for i in range(1,n):
        bb = b[i-1]
        
        det = bb[0,0]*bb[1,1] - bb[0,1]*bb[1,0]
        detinv = 1.0/det
        bi[i-1,0,0] =  bb[1,1]*detinv
        bi[i-1,1,1] =  bb[0,0]*detinv
        bi[i-1,1,0] = -bb[1,0]*detinv
        bi[i-1,0,1] = -bb[0,1]*detinv

        bbi = bi[i-1]
        
        r1[0,0] = bbi[0,0]*C[i-1,0,0] + bbi[0,1]*C[i-1,1,0]
        r1[0,1] = bbi[0,0]*C[i-1,0,1] + bbi[0,1]*C[i-1,1,1]
        r1[1,0] = bbi[1,0]*C[i-1,0,0] + bbi[1,1]*C[i-1,1,0]
        r1[1,1] = bbi[1,0]*C[i-1,0,1] + bbi[1,1]*C[i-1,1,1]
        
        r2[0] = bbi[0,0]*y[i-1,0] + bbi[0,1]*y[i-1,1]
        r2[1] = bbi[1,0]*y[i-1,0] + bbi[1,1]*y[i-1,1]

        i1 = i-1
        b[i,0,0] = B[i,0,0] - A[i1,0,0] * r1[0,0] - A[i1,0,1] * r1[1,0]
        b[i,0,1] = B[i,0,1] - A[i1,0,0] * r1[0,1] - A[i1,0,1] * r1[1,1]
        b[i,1,0] = B[i,1,0] - A[i1,1,0] * r1[0,0] - A[i1,1,1] * r1[1,0]
        b[i,1,1] = B[i,1,1] - A[i1,1,0] * r1[0,1] - A[i1,1,1] * r1[1,1]
        
        y[i,0] = d[i,0] - A[i1,0,0]*r2[0] - A[i1,0,1]*r2[1] 
        y[i,1] = d[i,1] - A[i1,1,0]*r2[0] - A[i1,1,1]*r2[1]
    
    bb = b[n-1]

    det = bb[0,0]*bb[1,1] - bb[0,1]*bb[1,0]
    detinv = 1.0/det
    bi[n-1,0,0] =  bb[1,1]*detinv
    bi[n-1,1,1] =  bb[0,0]*detinv
    bi[n-1,1,0] = -bb[1,0]*detinv
    bi[n-1,0,1] = -bb[0,1]*detinv

    bbi = bi[n-1]
    
    x = np.zeros( d.shape )
    
    x[n-1,0] = bbi[0,0] * y[n-1,0] +  bbi[0,1] * y[n-1,1]
    x[n-1,1] = bbi[1,0] * y[n-1,0] +  bbi[1,1] * y[n-1,1]
    
    for i in range(n-2,-1,-1):
        
        r3[0] = y[i,0]-C[i,0,0]*x[i+1,0]-C[i,0,1]*x[i+1,1]
        r3[1] = y[i,1]-C[i,1,0]*x[i+1,0]-C[i,1,1]*x[i+1,1]
        
        x[i,0] =  bi[i,0,0]*r3[0] + bi[i,0,1]*r3[1] 
        x[i,1] =  bi[i,1,0]*r3[0] + bi[i,1,1]*r3[1]
        
    return x        
        
        

@numba.njit
def solve_1D2P_centrifuge(
           N=21, 
           cpr_model=None,
           rlp_model=None,
           cpr_multiplier=1.0,
           viscosity_w=1.0,
           viscosity_n=1.0,
           gvhw = 0.0,
           gvhn = 0.0,
           sw_initial=0,
    
           centrifuge_rmid=0,
           is_drainage=True,
           t_rampup=0.0, 
           ref_acceleration=0.0,
    
           tD_end=0.2, 
           follow_stops = False,
           t_stops = [], # user imposed stops
           s_stops = [], # scale factors
           f_stops = [], # acceleration

           max_nr_iter=10, 
           nr_tolerance=1e-3, 
           verbose=1, 

           max_dsw = 0.1,
           max_num_step=1e20, 
           max_step_size = 0.01,
           min_step_size = 1e-7,
           start_step_size = 0.001,
           step_increment_factor = 2.0,
           step_reduction_factor = 2.0,
    
           refine_grid = False,

           reporting = True,
          ):
    '''1D2P incompressible flow solver for centrifuge experiments optimized for compilation with numba.
       
       The solver uses 1 dummy cell at inlet and outlet, these will be removed from simulation results
    '''
    
    # Add dummy cell at the inlet and outlet
    N2 = N + 2

    if refine_grid:
        
        delx = 1.0 / (N2-5-2)
        
        delxi = np.zeros(N2)
        delxi[ 0] = 0.1 * delx # Dummy inlet cell
        delxi[ 1] = 0.1 * delx
        delxi[ 2] = 0.2 * delx
        delxi[ 3] = 0.4 * delx
        delxi[ 4] = 0.8 * delx
        delxi[-1] = 0.1 * delx # Dummy outlet cell
        delxi[-2] = 0.1 * delx
        delxi[-3] = 0.2 * delx
        delxi[-4] = 0.4 * delx
        delxi[-5] = 0.8 * delx
        delxi[5:-5] = delx
        
        # There are N2-1 internal interfaces
        dtr = np.zeros( N2-1 )
        dtr = 2.0 / (delxi[1:]+delxi[:-1])
        dtr[ 0] = 2.0 / delxi[ 1]
        dtr[-1] = 2.0 / delxi[-2]
        
    else:    
        
        delx = 1.0/N 
        
        delxi = np.ones( N2 ) * delx
        
        # There are N2-1 internal interfaces
        dtr = np.ones( N2-1 )  / delx
        dtr[ 0] = 2.0/delx
        dtr[-1] = 2.0/delx
    
    # Mid cell position
    xD = np.zeros(N2)
    xD[0] = -0.5*delxi[0]
    xD[1] =  0.5*delxi[1]
    for i in range(2,N2):
        xD[i] = xD[i-1] + 0.5*(delxi[i-1]+delxi[i])
    
    # Cell interface position
    xfD = np.cumsum(delxi[:-1])-delxi[0]
    
    # Radial cell face position ratio to middle radius
    rfD_mid = (xfD-0.5 + centrifuge_rmid) / centrifuge_rmid
    
    # Radial mid cell position ratio to middle radius
    rD_mid = (xD-0.5 + centrifuge_rmid) / centrifuge_rmid
    
    # Make snapshots at each timestep
    movie_tD     = []
    movie_sw     = []
    movie_dtD    = []
    movie_pw     = []
    movie_pn     = []
    movie_pcw    = []
    movie_delp   = []
    movie_flxw   = []
    movie_flxn   = []
    movie_acc    = []
    movie_nr     = []

    # Initialize simulation time and timestep size   
    t_stops = np.sort( t_stops )
    t_stops = np.unique( t_stops )
    
    timestep = 0
    dtD = start_step_size
    tD = 0.0

    g_rampup = ref_acceleration

    facc_step_prv = 0.0
    facc_step_cur = 0.0
    facc_step_tD  = tD
    
    i_stop = 0 
    
    dtD, i_stop = update_step( tD, dtD,  i_stop, t_stops, follow_stops )

    pres_conv = s_stops[i_stop-1]
    facc_step = f_stops[i_stop-1]
 
    if facc_step != facc_step_cur:
        facc_step_prv = facc_step_cur
        facc_step_cur = facc_step
        facc_step_tD = tD

        if not follow_stops and dtD > start_step_size:
             dtD = start_step_size

    facc = calc_acceleration(tD, tr=t_rampup, gr=g_rampup, t0=facc_step_tD, g0=facc_step_prv, g1=facc_step_cur)

    dtD = np.minimum(start_step_size, dtD)

    # Surrounding fluid
    if is_drainage:
       # Non-wetting phase is surrounding
       rlpw_surround = 0.0
       rlpn_surround = 1.0 # inflow
       rlpw_surround_end = 1.0
       rlpn_surround_end = 1e-4
    else:
       # Wetting phase is surrounding
       rlpw_surround = 1.0
       rlpn_surround = 1e-4
       rlpw_surround_end = 1.0 # inflow
       rlpn_surround_end = 0.0

    # Pre-allocate arrays for solver
    flx   = np.zeros( (N2-1,2  ) )
    flxdd = np.zeros( (N2-1,2,2) )
    flxdu = np.zeros( (N2-1,2,2) )

    dSwdtd = np.zeros( (N2-2,2,2) )
    
    rhs = np.zeros( (N,2) )
    
    failure_reason = ''

    # Geometric weight used to estimate mean pressure in surrounding fluid
    pmeanwgt = (0.5 * centrifuge_rmid * (rD_mid**2 - rfD_mid[0]**2)[1:-1] * delxi[1:-1]).sum()

    # Set target mean pressure of surrounding fluid (use largest acceleration)
    facc_max = np.abs(f_stops).max()
    if is_drainage:
        pmean = pmeanwgt * gvhn * pres_conv * facc_max / 0.9
    else:
        pmean = pmeanwgt * gvhw * pres_conv * facc_max / 0.9

    # Set initial pressure
    pw = np.ones(N2,dtype=np.float64) * pmean

    # Set initial water saturation distribution
    sw = np.ones(N2,dtype=np.float64) * sw_initial

    while tD<tD_end:
      
        if timestep>max_num_step: 
            failure_reason = 'not converged - reached maximum number of timesteps STOP'
            break
            
        sw_prv = sw[1:-1].copy()
        pw_prv = pw[1:-1].copy()
        
        sw_new = sw[1:-1]
        pw_new = pw[1:-1]
        
        gvhwD = gvhw * pres_conv * np.abs(facc)
        gvhnD = gvhn * pres_conv * np.abs(facc)
    
        gvhwDi = gvhwD * rfD_mid
        gvhnDi = gvhnD * rfD_mid
        
        if is_drainage:
           pw0 = pmean - pmeanwgt * gvhn * pres_conv * np.abs(facc)
           pw[ 0] = pw0
           pw[-1] = pw0 + gvhnD
        else:
           pw0 = pmean - pmeanwgt * gvhw * pres_conv * np.abs(facc)
           pw[ 0] = pw0
           pw[-1] = pw0 + gvhwD
        
        dxdt = delxi[1:-1]/dtD
    
        dSwdtd[:,0,0] = +dxdt
        dSwdtd[:,1,0] = -dxdt

        for iter in range(max_nr_iter):
                     
            rlpw, rlpn, rlpwd, rlpnd = rlp_model.calc(sw)
            
            # Set surrounding fluid
            rlpw [0] = rlpw_surround
            rlpwd[0] = 0.0
            rlpn [0] = rlpn_surround
            rlpnd[0] = 0.0
            rlpw [-1] = rlpw_surround_end
            rlpwd[-1] = 0.0
            rlpn [-1] = rlpn_surround_end
            rlpnd[-1] = 0.0
            
            pcw, pcwd = cpr_model.calc(sw)
            
            # No capillary pressure for surrounding fluid
            pcw [ 0] = 0
            pcwd[ 0] = 0
            pcw [-1] = 0
            pcwd[-1] = 0
            
            mobw, mobn, mobwd, mobnd = rlpw/viscosity_w, rlpn/viscosity_n, rlpwd/viscosity_w, rlpnd/viscosity_n

            pcw  = pcw  * pres_conv * cpr_multiplier
            pcwd = pcwd * pres_conv * cpr_multiplier
            
            # Derivatives at cell interfaces, convention:
            # - du  derivative to upstream cell
            # - dd  derivative to downstream cell
            
            dpcwdx   = (pcw[+1:]-pcw[:-1]) * dtr
            dpcwdxdu =          -pcwd[:-1] * dtr
            dpcwdxdd =  pcwd[+1:]          * dtr
                   
            potw = (pw[1:]-pw[:-1])*dtr - gvhwDi
            potn = (pw[1:]-pw[:-1])*dtr - gvhnDi + dpcwdx
            
            mobwi  = np.where( potw<0, mobw[:-1], mobw[1:] )
            mobni  = np.where( potn<0, mobn[:-1], mobn[1:] )
            
            mobwidu = np.where( potw<0, mobwd[:-1], 0.0)
            mobwidd = np.where( potw<0, 0.0, mobwd[+1:])
            mobnidu = np.where( potn<0, mobnd[:-1], 0.0)
            mobnidd = np.where( potn<0, 0.0, mobnd[+1:])
 
            flx[:,0] = -mobwi*potw
            flx[:,1] = -mobni*potn

            flxdu[:,0,0] = -mobwidu*potw   
            flxdu[:,1,0] = -mobnidu*potn - mobni*dpcwdxdu

            flxdd[:,0,0] = -mobwidd*potw 
            flxdd[:,1,0] = -mobnidd*potn - mobni*dpcwdxdd

            flxdu[:,0,1] = +mobwi*dtr 
            flxdu[:,1,1] = +mobni*dtr

            flxdd[:,0,1] = -mobwi*dtr
            flxdd[:,1,1] = -mobni*dtr

            rhs[:,0] =  (sw_new - sw_prv)*dxdt + flx[1:,0] - flx[:-1,0]
            rhs[:,1] = -(sw_new - sw_prv)*dxdt + flx[1:,1] - flx[:-1,1]

            nr_residual = np.linalg.norm( rhs / dxdt[:,None] )

            if verbose>1: 
               print('iter,residual',iter,nr_residual)

            if nr_residual < nr_tolerance and iter>0: 
               break

            diag00 = dSwdtd + flxdu[1:  ]  - flxdd[:-1]
            diag10 =        - flxdu[1:-1] 
            diag01 =                       + flxdd[1:-1]

            dx_solve = lusolve2( diag10, diag00, diag01, rhs )

            check_finite_matrix( dx_solve )

            d_sw = dx_solve[:,0]
            d_pw = dx_solve[:,1]
                
            cutback_factor = np.maximum( np.abs(d_sw).max()/max_dsw, 1.0 )
            
            if iter>10:
                # Cut cycles by inserting random factor
                factor = 0.1 + 0.9*np.random.rand(1)[0]
                cutback_factor /= factor
                
            pw_new -= d_pw/cutback_factor                      
            sw_new -= d_sw/cutback_factor
       

        if nr_residual < nr_tolerance:
            
            # Converged, accept timestep
            
            tD = tD + dtD
            timestep += 1
                                  
            if reporting:
                
                delp = pw[1:] - pw[:-1]
                
                delp /= pres_conv
                pw1  = pw/pres_conv
                pcw /= pres_conv
               
                pn1 = pw1 + pcw
                
                flxw = flx[:,0] 
                flxn = flx[:,1]  

                # Only report cells inside core
                if is_drainage:
                    movie_tD    .append(tD         )
                    movie_dtD   .append(dtD        )
                    movie_sw    .append(sw   [1:-1].copy())
                    movie_delp  .append(delp       .copy())
                    movie_pw    .append(pw1  [1:-1].copy())
                    movie_pn    .append(pn1  [1:-1].copy())
                    movie_pcw   .append(pcw  [1:-1].copy())
                    movie_flxw  .append(flxw       .copy())
                    movie_flxn  .append(flxn       .copy())
                else:
                    # We reverse the profiles for imbibition, so that
                    # fluids move from left to right
                    movie_tD    .append(tD         )
                    movie_dtD   .append(dtD        )
                    movie_sw    .append(sw   [1:-1][::-1].copy())
                    movie_delp  .append(-delp      [::-1].copy())
                    movie_pw    .append(pw1  [1:-1][::-1].copy())
                    movie_pn    .append(pn1  [1:-1][::-1].copy())
                    movie_pcw   .append(pcw  [1:-1][::-1].copy())
                    movie_flxw  .append(-flxw      [::-1].copy())
                    movie_flxn  .append(-flxn      [::-1].copy())
                
                movie_acc.append( facc )
                movie_nr.append( iter )
                
            if verbose>0:
                print (timestep,'tD, residual',tD,nr_residual,'dtD',dtD,'nr_iter',iter)
            
            dtD = np.minimum( step_increment_factor*dtD, max_step_size )
            
            if tD<tD_end:
                dtD, i_stop = update_step( tD, dtD, i_stop, t_stops, follow_stops )

                pres_conv = s_stops[i_stop-1]
                facc_step = f_stops[i_stop-1]
 
                if facc_step != facc_step_cur:
                    facc_step_prv = facc_step_cur
                    facc_step_cur = facc_step
                    facc_step_tD = tD

                    if not follow_stops and dtD > start_step_size:
                       dtD = start_step_size

                facc = calc_acceleration(tD, tr=t_rampup, gr=g_rampup, t0=facc_step_tD, g0=facc_step_prv, g1=facc_step_cur)
    
        else:
            
            # Not converged, reduce timestepsize and try again
            
            # Reset saturation to previous value
            sw[1:-1] = sw_prv.copy()
            pw[1:-1] = pw_prv.copy()
                                  
            if verbose>0:
                print ('tD, residual,backup',tD,nr_residual,dtD)

            dtD = dtD / step_reduction_factor

            if dtD < min_step_size:
                failure_reason = 'timestep below minimum step size STOP'
                break

    # Drop dummy inlet and outlet cell
    if is_drainage:
       xD1 = xD[1:-1]
       delxD1 = delxi[1:-1]
    else:
       xD1 = 1 - xD[1:-1][::-1]
       delxD1 = delxi[1:-1][::-1]

    return (xD1, delxD1, movie_tD, movie_dtD, movie_sw, movie_pw, 
            movie_pn, movie_pcw, movie_delp, movie_flxw, movie_flxn, 
            movie_acc, movie_nr, failure_reason)


@numba.njit
def update_step(t, delt, i_stop, t_stops, follow_stops, time_tol=1e-6):
    assert len(t_stops) > 1

    n = len(t_stops)

    if t > t_stops[-1]:
        t = t_stops[-1]

    if i_stop > n - 1:
        i_stop = n - 1

    if i_stop < 1:
        i_stop = 1

    while t_stops[i_stop] < t + time_tol:
        if i_stop < n - 1:
            i_stop += 1
        else:
            break

    delt_propose = t_stops[i_stop] - t

    if follow_stops:
        delt_new = delt_propose
    else:
        delt_new = np.minimum(delt, delt_propose)

    return delt_new, i_stop


@numba.njit
def calc_acceleration(t, tr, gr, t0, g0, g1):
    g0_abs = math.fabs(g0)
    g1_abs = math.fabs(g1)

    if gr > 0.0 and tr > 0.0:
        tstartup = tr * math.fabs(math.sqrt(g1_abs / gr) - math.sqrt(g0_abs / gr))
    else:
        tstartup = 0.0

    if t < t0:
        g = g0
    elif t >= t0 + tstartup:
        g = g1
    else:
        if g1_abs < g0_abs:
            acc = -1.0
        else:
            acc = +1.0

        if (g1 + g0) < 0.0:
            mode = -1.0
        else:
            mode = +1.0

        g = (
            g0
            + 2.0 * mode * acc * math.sqrt(gr * g0_abs) * math.fabs(t - t0) / tr
            + mode * gr * ((t - t0) / tr) ** 2
        )

    return g
