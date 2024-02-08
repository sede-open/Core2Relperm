#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from scallib001.displacementmodel1D2P001 import DisplacementModel1D2P
import scallib001.relpermlib001 as rlplib

def test_solver1(benchmark):

    if benchmark is None:
       return

    KRWE = 0.65
    KROE = 0.759
    SWC = 0.08
    SORW = 0.14
    NW = 2.5
    NOW = 3.0
   
    rlp_model1 = rlplib.Rlp2PCorey(SWC, SORW, NW, NOW, KRWE, KROE)
   
    cpr_model1 = rlplib.CubicInterpolator(
        np.array(
            [
                0.08,
                0.1,
                0.15,
                0.3,
                0.4,
                0.5,
                0.59,
                0.68,
                0.73,
                0.78,
                0.81,
                0.82,
                0.858,
                0.86,
            ]
        ),
        np.array(
            [
                -0.0,
                -0.001,
                -0.0054,
                -0.015,
                -0.0193,
                -0.0241,
                -0.0435,
                -0.0923,
                -0.1284,
                -0.18,
                -0.21,
                -0.24,
                -0.9,
                -1.2,
            ]
        ),
    )

    T_END = 200.0

    Movie_times = np.linspace(0, T_END, 200)

    Schedule = pd.DataFrame(
        [
            [0.0, 1.0, 0.0],
            [5.2, 1.0, 0.01],
            [21.5, 1.0, 0.05],
            [37.6, 1.0, 0.1],
            [46.4, 1.0, 0.25],
            [62.5, 1.0, 0.5],
            [78.5, 1.0, 0.75],
            [94.6, 1.0, 0.9],
            [110.6, 1.0, 0.95],
            [128.3, 1.0, 0.99],
            [144.3, 1.0, 1.0],
            [166.4, 2.0, 1.0],
            [178.9, 5.0, 1.0],
        ],
        columns=["StartTime", "InjRate", "FracFlow"],
    )

    # Schedule
   
    model1 = DisplacementModel1D2P(
        NX=50,
        core_length=3.9,  # cm
        core_area=12.0,  # cm2
        permeability=250.0,  # mDarcy
        porosity=0.18,  # v/v
        sw_initial=SWC,  # v/v
        viscosity_w=1.0,  # cP
        viscosity_n=2.0,  # cP
        density_w=1000.0,  # kg/m3
        density_n=800.0,  # kg/m3
        rlp_model=rlp_model1,
        cpr_model=cpr_model1,
        time_end=T_END,  # hour
        rate_schedule=Schedule,
        movie_schedule=Movie_times,
    )

    results = model1.solve().results

    def wrapper():
        model1.solve().results

    benchmark(wrapper)
