#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

# Setup

import numpy as np
import pandas as pd

from scallib001.displacementmodel1D2P001 import DisplacementModel1D2P
import scallib001.relpermlib001 as rlplib


def setup_and_solve():

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

    return results



# Testing below


def test_tss_table(RTOL=0.001):

    results = setup_and_solve()

    # check that simulation results are identical to reference simulation

    # data created at 2024-02-01 22:52:46.104500
    
    tss_table = results.tss_table
    
    assert np.allclose( tss_table.TIME           .values[-1],   2.00000000E+02, rtol=RTOL)
    assert np.allclose( tss_table.DTIME          .values[-1],   1.00502513E+00, rtol=RTOL)
    assert np.allclose( tss_table.PVinj          .values[-1],   2.11467236E+03, rtol=RTOL)
    assert np.allclose( tss_table.tD             .values[-1],   2.11467236E+03, rtol=RTOL)
    assert np.allclose( tss_table.dtD            .values[-1],   3.57914931E+01, rtol=RTOL)
    assert np.allclose( tss_table.InjRate        .values[-1],   5.00000000E+00, rtol=RTOL)
    assert np.allclose( tss_table.PVInjRate      .values[-1],   5.93542260E-01, rtol=RTOL)
    assert np.allclose( tss_table.FracFlowInj    .values[-1],   1.00000000E+00, rtol=RTOL)
    assert np.allclose( tss_table.FracFlowPrd    .values[-1],   1.00000000E+00, rtol=RTOL)
    assert np.allclose( tss_table.WATERInj       .values[-1],   5.00000000E+00, rtol=RTOL)
    assert np.allclose( tss_table.OILInj         .values[-1],   0.00000000E+00, rtol=RTOL)
    assert np.allclose( tss_table.WATERProd      .values[-1],   5.00000000E+00, rtol=RTOL)
    assert np.allclose( tss_table.CumWATERInj    .values[-1],   1.35361800E+04, rtol=RTOL)
    assert np.allclose( tss_table.CumOILInj      .values[-1],   4.27782000E+03, rtol=RTOL)
    assert np.allclose( tss_table.CumWATER       .values[-1],   1.35305281E+04, rtol=RTOL)
    assert np.allclose( tss_table.CumOIL         .values[-1],   4.28347193E+03, rtol=RTOL)
    assert np.allclose( tss_table.P_inj          .values[-1],   1.27795742E+00, rtol=RTOL)
    assert np.allclose( tss_table.P_inj_wat      .values[-1],   1.27783120E+00, rtol=RTOL)
    assert np.allclose( tss_table.P_inj_oil      .values[-1],   1.00305627E+00, rtol=RTOL)
    assert np.allclose( tss_table.P_prod         .values[-1],   1.00000000E+00, rtol=RTOL)
    assert np.allclose( tss_table.P_prod_wat     .values[-1],   1.01114291E+00, rtol=RTOL)
    assert np.allclose( tss_table.P_prod_oil     .values[-1],   1.00000340E+00, rtol=RTOL)
    assert np.allclose( tss_table.delta_P        .values[-1],   2.77957419E-01, rtol=RTOL)
    assert np.allclose( tss_table.delta_P_w      .values[-1],   2.66688294E-01, rtol=RTOL)
    assert np.allclose( tss_table.delta_P_o      .values[-1],   3.05286721E-03, rtol=RTOL)

    assert results.movie_nr.sum() == 424
