#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

import numpy as np

import scallib001.relpermlib001 as rlplib

KRWE = 0.65
KROE = 0.759
SWC = 0.08
SORW = 0.14
NW = 2.5
NOW = 3.0

rlp_corey1 = rlplib.Rlp2PCorey(SWC, SORW, NW, NOW, KRWE, KROE)

cpr_data_sat = np.array(
    [0.08, 0.1, 0.15, 0.3, 0.4, 0.5, 0.59, 0.68, 0.73, 0.78, 0.81, 0.82, 0.858, 0.86]
)
cpr_data_cpr = np.array(
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
)

cpr_cubic1_lex0_rex0 = rlplib.CubicInterpolator(
    cpr_data_sat, cpr_data_cpr, lex=0, rex=0
)
cpr_cubic1_lex1_rex0 = rlplib.CubicInterpolator(
    cpr_data_sat, cpr_data_cpr, lex=1, rex=0
)
cpr_cubic1_lex0_rex1 = rlplib.CubicInterpolator(
    cpr_data_sat, cpr_data_cpr, lex=0, rex=1
)
cpr_cubic1_lex1_rex1 = rlplib.CubicInterpolator(
    cpr_data_sat, cpr_data_cpr, lex=1, rex=1
)

sv = np.linspace(SWC, 1 - SORW, 51)

kr1 = rlp_corey1.calc_kr1(sv)
kr2 = rlp_corey1.calc_kr2(sv)


rlp_cubic1_lex0_rex0 = rlplib.Rlp2PCubic(sv, kr1, kr2, lex=0, rex=0)
rlp_cubic1_lex0_rex1 = rlplib.Rlp2PCubic(sv, kr1, kr2, lex=0, rex=1)
rlp_cubic1_lex1_rex0 = rlplib.Rlp2PCubic(sv, kr1, kr2, lex=1, rex=0)
rlp_cubic1_lex1_rex1 = rlplib.Rlp2PCubic(sv, kr1, kr2, lex=1, rex=1)

Swc = 0.08
Sorw = 0.14
krwe = 0.65
kroe = 0.759
Lw = 1.50000000
Ew = 9.64238306
Tw = 1.27247992
Lo = 2.05310839
Eo = 3.34184238
To = 1.00000000

rlp_LET1 = rlplib.Rlp2PLET(Swc, Sorw, Lw, Ew, Tw, Lo, Eo, To, krwe, kroe)


def test_cpr_cubic1():

    assert np.allclose(
        cpr_cubic1_lex0_rex0.calc(np.array([0.000000]))[0], -0.00000000000000e00
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex0.calc(np.array([0.300000]))[0], -1.50000000000000e-02
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex0.calc(np.array([0.600000]))[0], -4.69908046260090e-02
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex0.calc(np.array([1.000000]))[0], -1.20000000000000e00
    )

    assert np.allclose(
        cpr_cubic1_lex0_rex1.calc(np.array([0.000000]))[0], -0.00000000000000e00
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex1.calc(np.array([0.300000]))[0], -1.50000000000000e-02
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex1.calc(np.array([0.600000]))[0], -4.69908046260090e-02
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex1.calc(np.array([1.000000]))[0], -2.21999999999810e01
    )

    assert np.allclose(
        cpr_cubic1_lex1_rex0.calc(np.array([0.000000]))[0], 4.00000000000000e-03
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex0.calc(np.array([0.300000]))[0], -1.50000000000000e-02
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex0.calc(np.array([0.600000]))[0], -4.69908046260090e-02
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex0.calc(np.array([1.000000]))[0], -1.20000000000000e00
    )

    assert np.allclose(
        cpr_cubic1_lex1_rex1.calc(np.array([0.000000]))[0], 4.00000000000000e-03
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex1.calc(np.array([0.300000]))[0], -1.50000000000000e-02
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex1.calc(np.array([0.600000]))[0], -4.69908046260090e-02
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex1.calc(np.array([1.000000]))[0], -2.21999999999810e01
    )


def test_cpr_cubic_drv1():

    assert np.allclose(
        cpr_cubic1_lex0_rex0.calc(np.array([0.000000]))[1], 0.00000000000000e00
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex0.calc(np.array([0.300000]))[1], -5.07749077490775e-02
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex0.calc(np.array([0.600000]))[1], -3.87853770059434e-01
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex0.calc(np.array([1.000000]))[1], 0.00000000000000e00
    )

    assert np.allclose(
        cpr_cubic1_lex0_rex1.calc(np.array([0.000000]))[1], 0.00000000000000e00
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex1.calc(np.array([0.300000]))[1], -5.07749077490775e-02
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex1.calc(np.array([0.600000]))[1], -3.87853770059434e-01
    )
    assert np.allclose(
        cpr_cubic1_lex0_rex1.calc(np.array([1.000000]))[1], -1.49999999999590e02
    )

    assert np.allclose(
        cpr_cubic1_lex1_rex0.calc(np.array([0.000000]))[1], -5.00000000000000e-02
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex0.calc(np.array([0.300000]))[1], -5.07749077490775e-02
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex0.calc(np.array([0.600000]))[1], -3.87853770059434e-01
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex0.calc(np.array([1.000000]))[1], 0.00000000000000e00
    )

    assert np.allclose(
        cpr_cubic1_lex1_rex1.calc(np.array([0.000000]))[1], -5.00000000000000e-02
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex1.calc(np.array([0.300000]))[1], -5.07749077490775e-02
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex1.calc(np.array([0.600000]))[1], -3.87853770059434e-01
    )
    assert np.allclose(
        cpr_cubic1_lex1_rex1.calc(np.array([1.000000]))[1], -1.49999999999590e02
    )


def test_corey1():

    assert np.allclose(rlp_corey1.calc_kr1(np.array([0.000000])), 0.00000000000000e00)
    assert np.allclose(rlp_corey1.calc_kr1(np.array([0.300000])), 2.74620878417945e-02)
    assert np.allclose(rlp_corey1.calc_kr1(np.array([0.600000])), 2.35876790045787e-01)
    assert np.allclose(rlp_corey1.calc_kr1(np.array([1.000000])), 6.50000000000000e-01)
    assert np.allclose(rlp_corey1.calc_kr2(np.array([0.000000])), 7.59000000000000e-01)
    assert np.allclose(rlp_corey1.calc_kr2(np.array([0.300000])), 2.80880797046478e-01)
    assert np.allclose(rlp_corey1.calc_kr2(np.array([0.600000])), 2.81111111111111e-02)
    assert np.allclose(rlp_corey1.calc_kr2(np.array([1.000000])), 0.00000000000000e00)
    assert np.allclose(
        rlp_corey1.calc_kr1_der(np.array([0.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_corey1.calc_kr1_der(np.array([0.300000])), 3.12069180020392e-01
    )
    assert np.allclose(
        rlp_corey1.calc_kr1_der(np.array([0.600000])), 1.13402302906629e00
    )
    assert np.allclose(
        rlp_corey1.calc_kr1_der(np.array([1.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_corey1.calc_kr2_der(np.array([0.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_corey1.calc_kr2_der(np.array([0.300000])), -1.50471855560613e00
    )
    assert np.allclose(
        rlp_corey1.calc_kr2_der(np.array([0.600000])), -3.24358974358974e-01
    )
    assert np.allclose(
        rlp_corey1.calc_kr2_der(np.array([1.000000])), 0.00000000000000e00
    )


def test_LET1():

    assert np.allclose(rlp_LET1.calc_kr1(np.array([0.000000])), 0.00000000000000e00)
    assert np.allclose(rlp_LET1.calc_kr1(np.array([0.300000])), 1.50374459293526e-02)
    assert np.allclose(rlp_LET1.calc_kr1(np.array([0.600000])), 1.20881285388824e-01)
    assert np.allclose(rlp_LET1.calc_kr1(np.array([1.000000])), 6.50000000000000e-01)
    assert np.allclose(rlp_LET1.calc_kr2(np.array([0.000000])), 7.59000000000000e-01)
    assert np.allclose(rlp_LET1.calc_kr2(np.array([0.300000])), 2.65282532284267e-01)
    assert np.allclose(rlp_LET1.calc_kr2(np.array([0.600000])), 3.41035522091661e-02)
    assert np.allclose(rlp_LET1.calc_kr2(np.array([1.000000])), 0.00000000000000e00)
    assert np.allclose(rlp_LET1.calc_kr1_der(np.array([0.000000])), 0.00000000000000e00)
    assert np.allclose(
        rlp_LET1.calc_kr1_der(np.array([0.300000])), 1.33534981167988e-01
    )
    assert np.allclose(
        rlp_LET1.calc_kr1_der(np.array([0.600000])), 7.65437448200503e-01
    )
    assert np.allclose(rlp_LET1.calc_kr1_der(np.array([1.000000])), 0.00000000000000e00)
    assert np.allclose(rlp_LET1.calc_kr2_der(np.array([0.000000])), 0.00000000000000e00)
    assert np.allclose(
        rlp_LET1.calc_kr2_der(np.array([0.300000])), -1.41703141664956e00
    )
    assert np.allclose(
        rlp_LET1.calc_kr2_der(np.array([0.600000])), -3.19837747167613e-01
    )
    assert np.allclose(rlp_LET1.calc_kr2_der(np.array([1.000000])), 0.00000000000000e00)


def test_cubic1():

    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr1(np.array([0.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr1(np.array([0.300000])), 2.74612991537836e-02
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr1(np.array([0.600000])), 2.35876264264027e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr1(np.array([1.000000])), 6.50000000000000e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr2(np.array([0.000000])), 7.59000000000000e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr2(np.array([0.300000])), 2.80881685206823e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr2(np.array([0.600000])), 2.81120093122634e-02
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr2(np.array([1.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr1_der(np.array([0.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr1_der(np.array([0.300000])), 3.11757260370099e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr1_der(np.array([0.600000])), 1.13417044470738e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr1_der(np.array([1.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr2_der(np.array([0.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr2_der(np.array([0.300000])), -1.50437014506902e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr2_der(np.array([0.600000])), -3.24617955386615e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex0.calc_kr2_der(np.array([1.000000])), 0.00000000000000e00
    )

    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr1(np.array([0.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr1(np.array([0.300000])), 2.74612991537836e-02
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr1(np.array([0.600000])), 2.35876264264027e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr1(np.array([1.000000])), 9.37306286678924e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr2(np.array([0.000000])), 7.59000000000000e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr2(np.array([0.300000])), 2.80881685206823e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr2(np.array([0.600000])), 2.81120093122634e-02
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr2(np.array([1.000000])), -5.44923076923068e-05
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr1_der(np.array([0.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr1_der(np.array([0.300000])), 3.11757260370099e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr1_der(np.array([0.600000])), 1.13417044470738e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr1_der(np.array([1.000000])), 2.05218776199232e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr2_der(np.array([0.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr2_der(np.array([0.300000])), -1.50437014506902e00
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr2_der(np.array([0.600000])), -3.24617955386615e-01
    )
    assert np.allclose(
        rlp_cubic1_lex0_rex1.calc_kr2_der(np.array([1.000000])), -3.89230769230756e-04
    )

    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr1(np.array([0.000000])), -1.88561808316413e-04
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr1(np.array([0.300000])), 2.74612991537836e-02
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr1(np.array([0.600000])), 2.35876264264027e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr1(np.array([1.000000])), 6.50000000000000e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr2(np.array([0.000000])), 9.87898830769230e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr2(np.array([0.300000])), 2.80881685206823e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr2(np.array([0.600000])), 2.81120093122634e-02
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr2(np.array([1.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr1_der(np.array([0.000000])), 2.35702260395516e-03
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr1_der(np.array([0.300000])), 3.11757260370099e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr1_der(np.array([0.600000])), 1.13417044470738e00
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr1_der(np.array([1.000000])), 0.00000000000000e00
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr2_der(np.array([0.000000])), -2.86123538461536e00
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr2_der(np.array([0.300000])), -1.50437014506902e00
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr2_der(np.array([0.600000])), -3.24617955386615e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex0.calc_kr2_der(np.array([1.000000])), 0.00000000000000e00
    )

    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr1(np.array([0.000000])), -1.88561808316413e-04
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr1(np.array([0.300000])), 2.74612991537836e-02
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr1(np.array([0.600000])), 2.35876264264027e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr1(np.array([1.000000])), 9.37306286678924e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr2(np.array([0.000000])), 9.87898830769230e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr2(np.array([0.300000])), 2.80881685206823e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr2(np.array([0.600000])), 2.81120093122634e-02
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr2(np.array([1.000000])), -5.44923076923068e-05
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr1_der(np.array([0.000000])), 2.35702260395516e-03
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr1_der(np.array([0.300000])), 3.11757260370099e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr1_der(np.array([0.600000])), 1.13417044470738e00
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr1_der(np.array([1.000000])), 2.05218776199232e00
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr2_der(np.array([0.000000])), -2.86123538461536e00
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr2_der(np.array([0.300000])), -1.50437014506902e00
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr2_der(np.array([0.600000])), -3.24617955386615e-01
    )
    assert np.allclose(
        rlp_cubic1_lex1_rex1.calc_kr2_der(np.array([1.000000])), -3.89230769230756e-04
    )
