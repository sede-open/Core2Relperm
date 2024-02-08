#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

import pytest
import numpy as np
from scallib001 import relpermlib001 as rlplib


def test_Corey_regular_derivatives1():

    # When corey exponent < 1 infinite saturation derivatives occur at the end points, unless regulated using eps>0
    swi = 0.20
    sor = 0.15
    krw = 0.35
    kro = 0.80
    lw = 0.5 # lw < 1, would cause infinite derivative at endpoint
    lo = 0.5 # lo < 1, would cause infinite derivative at endpoint

    eps = 1e-4

    rlp_model = rlplib.Rlp2PCorey(
        swi,
        sor,
        lw,
        lo,
        krw,
        kro,
        eps=eps,
    )

    # Choose to evaluate at the endpoints
    swv = np.array([swi, 1-sor])

    rlpw, rlpo, drlpw, drlpo = rlp_model.calc( swv )  

    assert np.isfinite(drlpo[0])
    assert np.isfinite(drlpw[1])

def test_Corey_linear_model():

    # Check derivatives for linear relperm
    swi = 0.00
    sor = 0.00
    krw = 1.00
    kro = 1.00
    lw = 1.0
    lo = 1.0

    eps = 1e-4 

    rlp_model = rlplib.Rlp2PCorey(
        swi,
        sor,
        lw,
        lo,
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

def test_Corey_endpoints():

    swi = 0.20
    sor = 0.15
    krw = 0.35
    kro = 0.80
    lw = 2.0
    lo = 3.0

    eps = 1e-4 

    rlp_model = rlplib.Rlp2PCorey(
        swi,
        sor,
        lw,
        lo,
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

def test_Corey_derivative1():
    
    Sr1 = 0.20
    Sr2 = 0.15
    Ke1 = 0.35
    Ke2 = 0.80
    N1 = 2.4
    N2 = 3.0

    eps = 1e-4
        
    rlp_model = rlplib.Rlp2PCorey(Sr1=Sr1,Sr2=Sr2,
                                N1=N1,N2=N2,
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
