#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

import pytest
from scallib001 import relpermlib001 as rlplib

def test_Rlp2PLET_input():
    
    Sr1 = 0.20
    Sr2 = 0.15
    Ke1 = 0.35
    Ke2 = 0.80
    L1 = 2.0
    E1 = 0.5
    T1 = 0.85 # note: T1 < 1
    L2 = 3.0
    E2 = 0.60
    T2 = 0.95 # note: T2 < 1

    eps = 1e-4


    def Rlp2PLET(Sr1=Sr1,Sr2=Sr2,
                 L1=L1,E1=E1,T1=T1,
                 L2=L2,E2=E2,T2=T2,
                 Ke1=Ke1,Ke2=Ke2,
                 eps=eps):
        
        rlp_model = rlplib.Rlp2PLET(Sr1=Sr1,Sr2=Sr2,
                                L1=L1,E1=E1,T1=T1,
                                L2=L2,E2=E2,T2=T2,
                                Ke1=Ke1,Ke2=Ke2,
                                eps=eps)
        
    with pytest.raises(ValueError):
        Rlp2PLET(Sr1=-Sr1)

    with pytest.raises(ValueError):
        Rlp2PLET(Sr2=-Sr2)

    with pytest.raises(ValueError):
        Rlp2PLET(L1=-L1)

    with pytest.raises(ValueError):
        Rlp2PLET(E1=-E1)

    with pytest.raises(ValueError):
        Rlp2PLET(T1=-T1)

    with pytest.raises(ValueError):
        Rlp2PLET(L2=-L2)

    with pytest.raises(ValueError):
        Rlp2PLET(E2=-E2)

    with pytest.raises(ValueError):
        Rlp2PLET(T2=-T2)

    with pytest.raises(ValueError):
        Rlp2PLET(Ke1=-Ke1)

    with pytest.raises(ValueError):
        Rlp2PLET(Ke2=-Ke2)

    with pytest.raises(ValueError):
        Rlp2PLET(eps=-eps)
        
    with pytest.raises(ValueError):
        Rlp2PLET(Sr1=2.0)

    with pytest.raises(ValueError):
        Rlp2PLET(Sr2=2.0)
        
    with pytest.raises(ValueError):
        Rlp2PLET(Sr1=1.0)

    with pytest.raises(ValueError):
        Rlp2PLET(Sr2=1.0)  
        
    with pytest.raises(ValueError):
        Rlp2PLET(Sr1=0.6, Sr2=0.6)
        
    with pytest.raises(ValueError):
        Rlp2PLET(E1=1.0, T1=0.5, E2=1.0, T2=2.0, eps=0)        
        
    with pytest.raises(ValueError):
        Rlp2PLET(E2=1.0, T2=0.5, E1=1.0, T1=2.0, eps=0)          
        
    with pytest.raises(ValueError):
        Rlp2PLET(L1=0.5, L2=2.0, T1=2.0, T2=2.0, eps=0)        
        
    with pytest.raises(ValueError):
        Rlp2PLET(L2=0.5, L1=2.0, T1=2.0, T2=2.0, eps=0)          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

def test_Rlp2PCorey_input():
    
    Sr1 = 0.20
    Sr2 = 0.15
    Ke1 = 0.35
    Ke2 = 0.80
    N1 = 2.0
    N2 = 3.0

    eps = 1e-4


    def Rlp2PCorey(Sr1=Sr1,Sr2=Sr2,
                 N1=N1, N2=N2,
                 Ke1=Ke1,Ke2=Ke2,
                 eps=eps):
        
        rlp_model = rlplib.Rlp2PCorey(Sr1=Sr1,Sr2=Sr2,
                                N1=N1,N2=N2,
                                Ke1=Ke1,Ke2=Ke2,
                                eps=eps)
        
    with pytest.raises(ValueError):
        Rlp2PCorey(Sr1=-Sr1)

    with pytest.raises(ValueError):
        Rlp2PCorey(Sr2=-Sr2)

    with pytest.raises(ValueError):
        Rlp2PCorey(N1=-N1)

    with pytest.raises(ValueError):
        Rlp2PCorey(N2=-N2)

    with pytest.raises(ValueError):
        Rlp2PCorey(Ke1=-Ke1)

    with pytest.raises(ValueError):
        Rlp2PCorey(Ke2=-Ke2)

    with pytest.raises(ValueError):
        Rlp2PCorey(eps=-eps)
        
    with pytest.raises(ValueError):
        Rlp2PCorey(Sr1=2.0)

    with pytest.raises(ValueError):
        Rlp2PCorey(Sr2=2.0)
        
    with pytest.raises(ValueError):
        Rlp2PCorey(Sr1=1.0)

    with pytest.raises(ValueError):
        Rlp2PCorey(Sr2=1.0)  
        
    with pytest.raises(ValueError):
        Rlp2PCorey(Sr1=0.6, Sr2=0.6)       
        
    with pytest.raises(ValueError):
        Rlp2PCorey(N1=0.5, N2=2.0, eps=0)        
        
    with pytest.raises(ValueError):
        Rlp2PCorey(N2=0.5, N1=2.0, eps=0)          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
