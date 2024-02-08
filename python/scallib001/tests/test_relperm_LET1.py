#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

import pytest
import numpy as np
from scallib001 import relpermlib001 as rlplib


def test_LET_regular_derivatives1():

    # When T<1 infinite saturation derivatives occur at the end points, unless regulated using eps>0
    swi = 0.20
    sor = 0.15
    krw = 0.35
    kro = 0.80
    lw = 2.0
    ew = 0.5
    tw = 0.85 # Note: tw < 1, would cause infinite derivative
    lo = 3.0
    eo = 0.60
    to = 0.95 # Note: to < 1, would cause infinite derivative

    eps = 1e-4

    rlp_model = rlplib.Rlp2PLET(
        swi,
        sor,
        lw,
        ew,
        tw,
        lo,
        eo,
        to,
        krw,
        kro,
        eps=eps,
    )

    # Choose to evaluate at the endpoints
    swv = np.array([swi, 1-sor])

    rlpw, rlpo, drlpw, drlpo = rlp_model.calc( swv )  

    assert np.isfinite(drlpo[0])
    assert np.isfinite(drlpw[1])

def test_LET_linear_model():

    # Check derivatives when L=1 and E=0 (linear relperm)
    swi = 0.00
    sor = 0.00
    krw = 1.00
    kro = 1.00
    lw = 1.0
    ew = 0.0
    tw = 0.85 # note not relevant when ew = 0
    lo = 1.0
    eo = 0.00
    to = 0.95 # note not relevant when e0 = 0

    eps = 1e-4 # note not relevant for this case

    rlp_model = rlplib.Rlp2PLET(
        swi,
        sor,
        lw,
        ew,
        tw,
        lo,
        eo,
        to,
        krw,
        kro,
        eps=eps,
    )

    # Choose to evaluate including the endpoints
    swv = np.linspace(swi, 1-sor, 11)

    rlpw, rlpo, drlpw, drlpo = rlp_model.calc( swv )  

    print('rlpw', rlpw)
    print('rlpo', rlpo)
    print('drlpw', drlpw)
    print('drlpo', drlpo)

    assert np.allclose( rlpw, swv )
    assert np.allclose( rlpo, 1.0-swv )
    assert np.allclose( drlpw, +1.0 )
    assert np.allclose( drlpo, -1.0 )

def test_LET_endpoints1():

    # uses corey exponents < 1

    swi = 0.20
    sor = 0.15
    krw = 0.35
    kro = 0.80
    lw = 2.0
    ew = 0.5
    tw = 0.85 # Note: tw < 1
    lo = 3.0
    eo = 0.60
    to = 0.95 # Note: to < 1

    eps = 1e-4 

    rlp_model = rlplib.Rlp2PLET(
        swi,
        sor,
        lw,
        ew,
        tw,
        lo,
        eo,
        to,
        krw,
        kro,
        eps=eps,
    )

    # Choose to evaluate including the endpoints
    swv = np.array([0.0, swi, 1-sor, 1.0])

    rlpw, rlpo, drlpw, drlpo = rlp_model.calc( swv )  

    assert np.allclose( rlpw, [0.0, 0.0, krw, krw] )
    assert np.allclose( rlpo, [kro, kro, 0.0, 0.0] )
    assert np.allclose( [drlpw[0], drlpw[-1]],  [0.0, 0.0] )
    assert np.allclose( [drlpo[0], drlpo[-1]],  [0.0, 0.0] )

def test_LET_endpoints2():

    # uses corey exponents = 1

    swi = 0.20
    sor = 0.15
    krw = 0.35
    kro = 0.80
    lw = 2.0
    ew = 0.5
    tw = 1.00
    lo = 3.0
    eo = 0.60
    to = 1.00 

    eps = 1e-4 

    rlp_model = rlplib.Rlp2PLET(
        swi,
        sor,
        lw,
        ew,
        tw,
        lo,
        eo,
        to,
        krw,
        kro,
        eps=eps,
    )

    # Choose to evaluate including the endpoints
    swv = np.array([0.0, swi, 1-sor, 1.0])

    rlpw, rlpo, drlpw, drlpo = rlp_model.calc( swv )  

    assert np.allclose( rlpw, [0.0, 0.0, krw, krw] )
    assert np.allclose( rlpo, [kro, kro, 0.0, 0.0] )
    assert np.allclose( [drlpw[0], drlpw[-1]],  [0.0, 0.0] )
    assert np.allclose( [drlpo[0], drlpo[-1]],  [0.0, 0.0] )

def test_LET_endpoints3():

    # uses corey exponents > 1

    swi = 0.20
    sor = 0.15
    krw = 0.35
    kro = 0.80
    lw = 2.0
    ew = 0.5
    tw = 2.00
    lo = 3.0
    eo = 0.60
    to = 2.00 

    eps = 1e-4 

    rlp_model = rlplib.Rlp2PLET(
        swi,
        sor,
        lw,
        ew,
        tw,
        lo,
        eo,
        to,
        krw,
        kro,
        eps=eps,
    )

    # Choose to evaluate including the endpoints
    swv = np.array([0.0, swi, 1-sor, 1.0])

    rlpw, rlpo, drlpw, drlpo = rlp_model.calc( swv )  

    assert np.allclose( rlpw, [0.0, 0.0, krw, krw] )
    assert np.allclose( rlpo, [kro, kro, 0.0, 0.0] )
    assert np.allclose( [drlpw[0], drlpw[-1]],  [0.0, 0.0] )
    assert np.allclose( [drlpo[0], drlpo[-1]],  [0.0, 0.0] )

def test_LET_derivative1():

    # uses corey exponents < 1 

    Sr1 = 0.23
    Sr2 = 0.15
    Ke1 = 0.51
    Ke2 = 0.70
    L1 = 2.2
    E1 = 0.8
    T1 = 0.85 # note: T1 < 1
    L2 = 3.3
    E2 = 0.80
    T2 = 0.90 # note: T2 < 1

    eps = 1e-4
 
    rlp_model = rlplib.Rlp2PLET(Sr1=Sr1,Sr2=Sr2,
                                L1=L1,E1=E1,T1=T1,
                                L2=L2,E2=E2,T2=T2,
                                Ke1=Ke1,Ke2=Ke2,
                                eps=eps)

    deps = 0.01
    
    swv = np.linspace(Sr1+deps, 1-Sr2-deps, 1001)
    
    krw, kro, dkrw, dkro = rlp_model.calc(swv)
    
    dkrw_num = np.gradient(krw, swv, edge_order=2)
    dkro_num = np.gradient(kro, swv, edge_order=2)
    
    print('dkrw abs error', np.abs(dkrw - dkrw_num).max(), 'rel error', np.abs((dkrw - dkrw_num)/dkrw).max())
    print('dkro abs error', np.abs(dkro - dkro_num).max(), 'rel error', np.abs((dkro - dkro_num)/dkro).max())
    
    assert np.allclose(dkrw, dkrw_num, atol=1e-3, rtol=1e-2)
    assert np.allclose(dkro, dkro_num, atol=1e-3, rtol=1e-2)

def test_LET_derivative2():

    # uses corey exponents = 1 

    Sr1 = 0.23
    Sr2 = 0.15
    Ke1 = 0.51
    Ke2 = 0.70
    L1 = 2.2
    E1 = 0.8
    T1 = 1.00 
    L2 = 3.3
    E2 = 0.80
    T2 = 1.00

    eps = 1e-4
 
    rlp_model = rlplib.Rlp2PLET(Sr1=Sr1,Sr2=Sr2,
                                L1=L1,E1=E1,T1=T1,
                                L2=L2,E2=E2,T2=T2,
                                Ke1=Ke1,Ke2=Ke2,
                                eps=eps)

    deps = 0.01
    
    swv = np.linspace(Sr1+deps, 1-Sr2-deps, 1001)
    
    krw, kro, dkrw, dkro = rlp_model.calc(swv)
    
    dkrw_num = np.gradient(krw, swv, edge_order=2)
    dkro_num = np.gradient(kro, swv, edge_order=2)
    
    print('dkrw abs error', np.abs(dkrw - dkrw_num).max(), 'rel error', np.abs((dkrw - dkrw_num)/dkrw).max())
    print('dkro abs error', np.abs(dkro - dkro_num).max(), 'rel error', np.abs((dkro - dkro_num)/dkro).max())
    
    assert np.allclose(dkrw, dkrw_num, atol=1e-3, rtol=1e-2)
    assert np.allclose(dkro, dkro_num, atol=1e-3, rtol=1e-2)

def test_LET_derivative3():

    # uses corey exponents > 1 

    Sr1 = 0.23
    Sr2 = 0.15
    Ke1 = 0.51
    Ke2 = 0.70
    L1 = 2.2
    E1 = 0.8
    T1 = 2.00 
    L2 = 3.3
    E2 = 0.80
    T2 = 2.00

    eps = 1e-4
 
    rlp_model = rlplib.Rlp2PLET(Sr1=Sr1,Sr2=Sr2,
                                L1=L1,E1=E1,T1=T1,
                                L2=L2,E2=E2,T2=T2,
                                Ke1=Ke1,Ke2=Ke2,
                                eps=eps)

    deps = 0.01
    
    swv = np.linspace(Sr1+deps, 1-Sr2-deps, 1001)
    
    krw, kro, dkrw, dkro = rlp_model.calc(swv)
    
    dkrw_num = np.gradient(krw, swv, edge_order=2)
    dkro_num = np.gradient(kro, swv, edge_order=2)
    
    print('dkrw abs error', np.abs(dkrw - dkrw_num).max(), 'rel error', np.abs((dkrw - dkrw_num)/dkrw).max())
    print('dkro abs error', np.abs(dkro - dkro_num).max(), 'rel error', np.abs((dkro - dkro_num)/dkro).max())
    
    assert np.allclose(dkrw, dkrw_num, atol=1e-3, rtol=1e-2)
    assert np.allclose(dkro, dkro_num, atol=1e-3, rtol=1e-2)

def test_LET_vs_Corey1():
    
    # LET model defaults to Corey when E = 0

    Sr1 = 0.20
    Sr2 = 0.15
    Ke1 = 0.35
    Ke2 = 0.80
    L1 = 2.0
    E1 = 0.0
    T1 = 0.85 # irrelevant since E1 = 0
    L2 = 3.0
    E2 = 0.00
    T2 = 0.95 # irrelevant since E2 = 0

    eps = 1e-4
 
    rlp_model1 = rlplib.Rlp2PLET(Sr1=Sr1,Sr2=Sr2,
                                L1=L1,E1=E1,T1=T1,
                                L2=L2,E2=E2,T2=T2,
                                Ke1=Ke1,Ke2=Ke2,
                                eps=eps)

    rlp_model2 = rlplib.Rlp2PCorey(Sr1=Sr1,Sr2=Sr2,
                                N1=L1,N2=L2,
                                Ke1=Ke1,Ke2=Ke2,
                                eps=eps)
    deps = 0.01
    
    swv = np.linspace(Sr1+deps, 1-Sr2-deps, 1001)
    
    krw1, kro1, dkrw1, dkro1 = rlp_model1.calc(swv)
    krw2, kro2, dkrw2, dkro2 = rlp_model2.calc(swv)
    
    assert np.allclose(krw1, krw2)
    assert np.allclose(kro1, kro2)
    assert np.allclose(dkrw1, dkrw2)
    assert np.allclose(dkro1, dkro2)
