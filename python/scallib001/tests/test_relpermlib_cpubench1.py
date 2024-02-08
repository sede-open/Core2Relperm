#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

import numpy as np

import scallib001.relpermlib001 as rlplib


def test_setup_rlp_models( benchmark ):

    if benchmark is None:

       return

    else:
       
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
       
       
       sv101 = np.linspace(0, 1, 101)
       
       # Force jit compilation before benchmarking
       rlp_corey1.calc(sv101)
       rlp_cubic1_lex0_rex1.calc(sv101)
       rlp_LET1.calc(sv101)
       cpr_cubic1_lex0_rex1.calc(sv101)
       
       globals()['sv101'] = sv101
       globals()['sv1001'] = sv1001
       globals()['rlp_corey1'] = rlp_corey1
       globals()['rlp_cubic1_lex0_rex1'] = rlp_cubic1_lex0_rex1
       globals()['rlp_LET1'] = rlp_LET1
       globals()['cpr_cubic1_lex0_rex1'] = cpr_cubic1_lex0_rex1



def test_rlp_corey1_sv101(benchmark):

    if benchmark is None:
       return

    def t():
        rlp_corey1.calc(sv101)

    benchmark(t)


def test_rlp_cubic1_sv101(benchmark):

    if benchmark is None:
       return

    def t():
        rlp_cubic1_lex0_rex1.calc(sv101)

    benchmark(t)


def test_rlp_LET1_sv101(benchmark):

    if benchmark is None:
       return

    def t():
        rlp_LET1.calc(sv101)

    benchmark(t)


def test_cpr_cubic1_sv101(benchmark):

    if benchmark is None:
       return

    def t():
        cpr_cubic1_lex0_rex1.calc(sv101)

    benchmark(t)


sv1001 = np.linspace(0, 1, 1001)


def test_corey1_sv1001(benchmark):

    if benchmark is None:
       return

    def t():
        rlp_corey1.calc(sv1001)

    benchmark(t)


def test_cubic1_sv1001(benchmark):

    if benchmark is None:
       return

    def t():
        rlp_cubic1_lex0_rex1.calc(sv1001)

    benchmark(t)


def test_LET1_sv1001(benchmark):

    if benchmark is None:
       return

    def t():
        rlp_LET1.calc(sv1001)

    benchmark(t)


def test_cpr_cubic1_sv1001(benchmark):

    if benchmark is None:
       return

    def t():
        cpr_cubic1_lex0_rex1.calc(sv1001)

    benchmark(t)
