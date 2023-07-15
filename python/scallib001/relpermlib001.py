#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------
#
# relpermlib001
# - collection of 2-phase relperm functions
# - calculation of function value and derivative
# - truncation or extrapolation (left lex, right rex)
# - tabular interpolation, linear and cubic, for 1D data
# - tabular interpolation, linear and cubic, for two-phase data
# - analytical Corey and LET two-phase functions
# - compiled using numba for high CPU performance
#
#----------------------------------------------------------------------------

import numba
import numpy as np


@numba.njit
def pchip_estimate_derivatives(x, y, lex=1, rex=1 ):
    '''Estimate derivatives for piecE1ise cubic hermite spline function.'''
    hk = x[1:] - x[:-1]
    mk = (y[1:] - y[:-1]) / hk
    
    w1 = 2*hk[1:] + hk[:-1]
    w2 = hk[1:] + 2*hk[:-1]
    
    m1 = mk[:-1]
    m2 = mk[ 1:]
    
    condition = m1*m2 > 0
    
    w1 = w1[condition]
    w2 = w2[condition]
    m1 = m1[condition]
    m2 = m2[condition]
    
    dk = np.zeros_like(y)
    dk[1:-1][condition] = m1*m2*(w1+w2)/(w1*m2+w2*m1)

    if  lex==0:
        # Truncation
        dk[ 0] = 0
    else:
        # Linear extrapolation
        dk[ 0] = mk[ 0 ]
        
    if rex==0:
        # Truncation
        dk[-1] = 0
    else:
        # Linear extrapolation
        dk[-1] = mk[-1]   
    
    return dk

@numba.njit
def pchip_fit( xi, yi, lex=1, rex=1 ):
    '''Fit coefficients for piecE1ise cubic hermite spline function.'''
    
    # Extend interval T2 enforce truncation or linear extrapolation
    
    n = len(xi)
    
    x0 = np.zeros(n+2)
    y0 = np.zeros(n+2)
    
    x0[ 0] = xi[ 0] - (xi[ 1] - xi[ 0])
    x0[-1] = xi[-1] + (xi[-1] - xi[-2])
    
    x0[1:-1] = xi
    
    if lex==0:
        y0[ 0] = yi[0]
    else:
        y0[0] = yi[0] - (yi[ 1] - yi[ 0])
        
    if rex==0:
        y0[-1] = yi[-1]
    else:
        y0[-1] = yi[-1] + (yi[-1] - yi[-2])
    
    y0[1:-1] = yi
    
    # Estimate derivatives
    d0 = pchip_estimate_derivatives( x0, y0, lex, rex )
    
    # Return extended interval and coefficients
    return x0, pchip_coefs( x0, y0, d0 )

@numba.njit
def pchip_coefs( xi, yi, di ):
  '''Calculate coefficients for piecE1ise cubic hermite spline function.'''
  x1 = xi[:-1]
  x2 = xi[+1:]

  f1 = yi[:-1]
  f2 = yi[+1:]
  
  d1 = di[:-1]
  d2 = di[+1:]
    
  h =    x2 - x1
  df = ( f2 - f1 ) / h

  c0 = f1
  c1 = d1
  c2 = - ( 2.0 * d1 - 3.0 * df + d2 ) / h
  c3 =   (       d1 - 2.0 * df + d2 ) / h / h
    
  return c0,c1,c2,c3  

import numba

# T2DO: speed up

@numba.njit
def pchip_bisect_OLD( x, xi ):
    
    n = len(x)
    
    left = np.zeros( n )
    
    n1 = len(xi)-1
    
    for i in range( n ):
        
        L2w = 0; high = n1
    
        xv = x[i]
        
        if xv < xi[L2w]:
            left[i] = L2w
        elif xv>xi[high]:
            left[i] = high-1
        else:
            while True:
                if L2w==high-1:
                    left[i] = L2w
                    break

                mid = int( (high+L2w)/2 )

                if xi[mid]<=xv:
                    L2w = mid
                else:
                    high = mid
                    
    return left.astype(np.int32)


@numba.njit
def pchip_eval_OLD( x, xi, c0, c1, c2, c3 ):
    
    left = pchip_bisect( x, xi )
    
    dx = x - xi[left]
    
    c0 = c0[left]
    c1 = c1[left]
    c2 = c2[left]
    c3 = c3[left]
    
    f = c0 + dx* ( c1 + dx * (c2 + dx*c3 ) )
    
    d = c1 + dx * ( 2.0 * c2 + dx * 3.0 * c3 )
    
    return f, d

@numba.njit
def pchip_eval( x, xi, c0, c1, c2, c3 ):
    '''Evalulate piecE1ise cubic hermite spline function and its derivative at x.'''
    left = np.digitize( x, xi )-1
    left = np.maximum( left, 0 )
    left = np.minimum( left, len(xi)-2 )
    
    dx = x - xi[left]
    
    c0 = c0[left]
    c1 = c1[left]
    c2 = c2[left]
    c3 = c3[left]
    
    f = c0 + dx* ( c1 + dx * (c2 + dx*c3 ) )
    
    d = c1 + dx * ( 2.0 * c2 + dx * 3.0 * c3 )
    
    return f, d

def make_pchip_OLD( xi, yi, lex=1, rex=1 ):
    
    lex = lex
    rex = rex
    
    x0,(c0,c1,c2,c3) = pchip_fit( xi, yi, lex, rex )   
    
    @numba.njit
    def itp(x):
        return pchip_eval_OLD( x, x0, c0, c1, c2, c3 )

    # compile
    itp(np.ones(2))

    return itp

def make_pchip( xi, yi, lex=1, rex=1 ):
    '''Create piecE1ise cubic hermite spline evaluation function.'''
    lex = lex
    rex = rex
    
    x0,(c0,c1,c2,c3) = pchip_fit( xi, yi, lex, rex )   
    
    @numba.njit
    def itp(x):
        return pchip_eval( x, x0, c0, c1, c2, c3 )

    # compile
    itp(np.ones(2))
    
    return itp

spec = [
    ('xi',    numba.float64[:] ),
    ('yi',    numba.float64[:] ),
    ('lex',   numba.int64      ),
    ('rex',   numba.int64      ),
    ('x0',    numba.float64[:] ),
    ('c0',    numba.float64[:] ),
    ('c1',    numba.float64[:] ),
    ('c2',    numba.float64[:] ),
    ('c3',    numba.float64[:] ),
]

@numba.experimental.jitclass(spec)
class PchipInterpolator(object):
    '''PiecE1ise cubic hermite spline interpolator.'''
    def __init__(self,xi,yi,lex=1,rex=1):
        self.xi = xi
        self.yi = yi
        self.lex = lex
        self.rex = rex
    
        x0,(c0,c1,c2,c3) = pchip_fit( xi, yi, lex, rex )
        
        self.x0 = x0
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
    
    def calc(self,x):
        return pchip_eval( x, self.x0, self.c0, self.c1, self.c2, self.c3 )

make_cubic_interpolator = make_pchip

CubicInterpolator = PchipInterpolator

#-------------------------------------------------------------------------------------
#--- Linear & cubic interpolation of tabular data
#    Corey and LET functions
#-------------------------------------------------------------------------------------

@numba.jit(nopython=True)
def rlp_2p_linear(sat1v,sat1_data,kr1_data,kr2_data,lex=0,rex=0):

    i = np.digitize( sat1v, sat1_data )-1

    i = np.minimum( i, len(sat1_data)-2 )
    i = np.maximum( i, 0 )
    
    xl = sat1_data[i]
    xr = sat1_data[i+1]

    yl = kr1_data[i]
    yr = kr1_data[i+1]

    k1d = (yr-yl)/(xr-xl)
    k1  = yl + k1d*(sat1v-xl)

    yl = kr2_data[i]
    yr = kr2_data[i+1]

    k2d = (yr-yl)/(xr-xl)
    k2  = yl + k2d*(sat1v-xl)

    if lex==0:
        # Truncate left
        l = sat1v<sat1_data[0]
        
        k1[l] = kr1_data[0]
        k2[l] = kr2_data[0]
        
        k1d[l] = 0
        k2d[l] = 0
    
    if rex==0:
        # Truncate right
        r = sat1v>sat1_data[-1]
        
        k1[r] = kr1_data[-1]
        k2[r] = kr2_data[-1]
        
        k1d[r] = 0
        k2d[r] = 0
       
    return k1, k2, k1d, k2d

@numba.njit
def rlp_2p_corey(sat1v,Sr1,Sr2,N1,N2,Ke1,Ke2):

    n = len(sat1v)

    kr1 = np.zeros( n )
    kr2 = np.zeros( n )

    kr1d = np.zeros( n )
    kr2d = np.zeros( n )

    for i in range(n):

        s1 = sat1v[i]

        if s1<Sr1:
            kr2[i] = Ke2
        elif s1>1-Sr2:
            kr1[i] = Ke1
        else:
            ss = (s1-Sr1)/(1-Sr2-Sr1)

            kr1[i] = np.power(   ss, N1 ) * Ke1
            kr2[i] = np.power( 1-ss, N2 ) * Ke2

            kr1d[i] = +np.power(   ss, N1-1 ) * Ke1 * N1 / (1-Sr1-Sr2)
            kr2d[i] = -np.power( 1-ss, N2-1 ) * Ke2 * N2 / (1-Sr1-Sr2)

    return kr1, kr2, kr1d, kr2d


@numba.njit
def rlp_2p_let(sat1v,Sr1,Sr2,L1,E1,T1,L2,E2,T2,Ke1,Ke2):

    n = len(sat1v)

    kr1 = np.zeros( n )
    kr2 = np.zeros( n )

    kr1d = np.zeros( n )
    kr2d = np.zeros( n )

    Ke1s = Ke1 / (1-Sr1-Sr2)
    Ke2s = Ke2 / (1-Sr1-Sr2)
    
    for i in range(n):

        s1 = sat1v[i]

        if s1<Sr1:
            kr2[i] = Ke2
        elif s1>1-Sr2:
            kr1[i] = Ke1
        else:
            ss = (s1-Sr1)/(1-Sr2-Sr1)

            pw0f =  np.power(   ss, L1 )
            po0f =  np.power( 1-ss, L2 )
 
            pw0d = +np.power(   ss, L1-1 ) * L1
            po0d = -np.power( 1-ss, L2-1 ) * L2

            if E1>0:
                
                pw1f =  np.power( 1-ss, T1   )
                pw1d = -np.power( 1-ss, T1-1 ) * T1
               
                dw = (pw0f+E1*pw1f)
                
                kr1 [i] = Ke1  * pw0f/dw 
                kr1d[i] = Ke1s * (pw0d*pw1f - pw0f*pw1d) / (dw*dw) * E1
                
            else:
                
                kr1 [i] = Ke1  * pw0f
                kr1d[i] = Ke1s * pw0d
                
            if E2>0:
                
                po1f =  np.power(   ss, T2   )
                po1d = +np.power(   ss, T2-1 ) * T2
               
                do = (po0f+E2*po1f)
                
                kr2 [i] = Ke2  * po0f/do 
                kr2d[i] = Ke2s * (po0d*po1f - po0f*po1d) / (do*do) * E2
                
            else:
                
                kr2 [i] = Ke2  * po0f
                kr2d[i] = Ke2s * po0d
 
    return kr1, kr2, kr1d, kr2d

spec = [
    ('sat1', numba.float64[:] ),
    ('kr1',  numba.float64[:] ),
    ('kr2',  numba.float64[:] ),
    ('lex',  numba.int64      ),
    ('rex',  numba.int64      ),
]

# T2DO lex rex

@numba.experimental.jitclass(spec)
class Rlp2PLinear(object):
    
    def __init__(self,sat1,kr1,kr2,lex=0,rex=0):
        
        assert len(sat1)>1
        assert len(kr1)>1
        assert len(kr2)>1
        
        # It is computationally more efficient to implement
        # truncation by extending iput data set with edge values 
        
        if lex==0:
            satn = sat1[0] - (sat1[1]-sat1[0])
            
            sat1 = np.hstack( ( np.array([satn  ]), sat1 ) )
            kr1  = np.hstack( ( np.array([kr1[0]]), kr1  ) )                     
            kr2  = np.hstack( ( np.array([kr2[0]]), kr2  ) )
            
        if rex==0:
            satn = sat1[-1] + (sat1[-1]-sat1[-2])
            
            sat1 = np.hstack( ( sat1, np.array([satn   ]) ) )
            kr1  = np.hstack( ( kr1,  np.array([kr1[-1]]) ) )                     
            kr2  = np.hstack( ( kr2,  np.array([kr2[-1]]) ) )

        self.sat1 = sat1
        self.kr1  = kr1
        self.kr2  = kr2
        self.lex  = lex
        self.rex  = rex
    
    def calc(self,sat1v):
        return rlp_2p_linear( sat1v, self.sat1, self.kr1, self.kr2, 
                              lex=1, rex=1 )

    def calc_kr1(self,sat1v):
        return self.calc(sat1v)[0]

    def calc_kr2(self,sat1v):
        return self.calc(sat1v)[1]
 
    def calc_kr1_der(self,sat1v):
        return self.calc(sat1v)[2]

    def calc_kr2_der(self,sat1v):
        return self.calc(sat1v)[3]
    
spec = [
    ('sat1',  numba.float64[:] ),
    ('kr1',   numba.float64[:] ),
    ('kr2',   numba.float64[:] ),
    ('lex',   numba.int64      ),
    ('rex',   numba.int64      ),
    ('w_x0',  numba.float64[:] ),
    ('w_c0',  numba.float64[:] ),
    ('w_c1',  numba.float64[:] ),
    ('w_c2',  numba.float64[:] ),
    ('w_c3',  numba.float64[:] ),
    ('o_x0',  numba.float64[:] ),
    ('o_c0',  numba.float64[:] ),
    ('o_c1',  numba.float64[:] ),
    ('o_c2',  numba.float64[:] ),
    ('o_c3',  numba.float64[:] ),
]

@numba.experimental.jitclass(spec)
class Rlp2PCubic(object):
    
    def __init__(self,sat1,kr1,kr2,lex=1,rex=1):
        self.sat1 = sat1
        self.kr1  = kr1
        self.kr2  = kr2
        self.lex  = lex
        self.rex  = rex
    
        x0,(c0,c1,c2,c3) = pchip_fit( sat1, kr1, lex, rex )
        
        self.w_x0 = x0
        self.w_c0 = c0
        self.w_c1 = c1
        self.w_c2 = c2
        self.w_c3 = c3
        
        x0,(c0,c1,c2,c3) = pchip_fit( sat1, kr2, lex, rex )
        
        self.o_x0 = x0
        self.o_c0 = c0
        self.o_c1 = c1
        self.o_c2 = c2
        self.o_c3 = c3
   
    def calc(self,sat1v):
        k1,k1d = pchip_eval( sat1v, self.w_x0, self.w_c0, self.w_c1, self.w_c2, self.w_c3 )
        k2,k2d = pchip_eval( sat1v, self.o_x0, self.o_c0, self.o_c1, self.o_c2, self.o_c3 )
        return k1,k2,k1d,k2d

    def calc_kr1(self,sat1v):
        return self.calc(sat1v)[0]

    def calc_kr2(self,sat1v):
        return self.calc(sat1v)[1]
 
    def calc_kr1_der(self,sat1v):
        return self.calc(sat1v)[2]

    def calc_kr2_der(self,sat1v):
        return self.calc(sat1v)[3]

spec = [
    ('Sr1', numba.float64 ),
    ('Sr2', numba.float64 ),
    ('L1',  numba.float64 ),
    ('E1',  numba.float64 ),
    ('T1',  numba.float64 ),
    ('L2',  numba.float64 ),
    ('E2',  numba.float64 ),
    ('T2',  numba.float64 ),
    ('Ke1', numba.float64 ),
    ('Ke2', numba.float64 ),
]

@numba.experimental.jitclass(spec)
class Rlp2PLET(object):
    
    def __init__(self,Sr1,Sr2,L1,E1,T1,L2,E2,T2,Ke1,Ke2):
        self.Sr1  = Sr1
        self.Sr2 = Sr2
        self.L1   = L1
        self.E1   = E1
        self.T1   = T1
        self.L2   = L2
        self.E2   = E2
        self.T2   = T2
        self.Ke1 = Ke1
        self.Ke2 = Ke2
        
    def calc(self,sat1v):        
        return rlp_2p_let(sat1v,self.Sr1,self.Sr2,
                          self.L1,self.E1,self.T1,
                          self.L2, self.E2,self.T2,
                          self.Ke1,self.Ke2)

    def calc_kr1(self,sat1v):
        return self.calc(sat1v)[0]

    def calc_kr2(self,sat1v):
        return self.calc(sat1v)[1]
 
    def calc_kr1_der(self,sat1v):
        return self.calc(sat1v)[2]

    def calc_kr2_der(self,sat1v):
        return self.calc(sat1v)[3]

spec = [
    ('Sr1', numba.float64 ),
    ('Sr2', numba.float64 ),
    ('N1',  numba.float64 ),
    ('N2',  numba.float64 ),
    ('Ke1', numba.float64 ),
    ('Ke2', numba.float64 ),
]

@numba.experimental.jitclass(spec)
class Rlp2PCorey(object):
    
    def __init__(self,Sr1,Sr2,N1,N2,Ke1,Ke2):
        self.Sr1 = Sr1
        self.Sr2 = Sr2
        self.N1  = N1
        self.N2  = N2
        self.Ke1 = Ke1
        self.Ke2 = Ke2
        
    def calc(self,sat1v):        
        return rlp_2p_corey(sat1v,self.Sr1,self.Sr2,
                            self.N1,self.N2,self.Ke1,self.Ke2)
    
    def calc_kr1(self,sat1v):
        return self.calc(sat1v)[0]

    def calc_kr2(self,sat1v):
        return self.calc(sat1v)[1]
 
    def calc_kr1_der(self,sat1v):
        return self.calc(sat1v)[2]

    def calc_kr2_der(self,sat1v):
        return self.calc(sat1v)[3]
