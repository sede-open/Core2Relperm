#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

import numpy as np
import scallib001.relpermlib001 as rlplib


def test_power_eps1():

    eps = 1e-4

    # check power_eps(0.0, 0.0, eps), i.e., zero power at zero

    exact_fd_check1 = 1/eps / np.log(1 + 1/eps)

    assert np.allclose( rlplib.power_eps(0.0, 0.0, eps), [0.0, exact_fd_check1] )
    assert np.allclose( rlplib.power_eps(0.0, 1e-10, eps), [0.0, exact_fd_check1] )

    # check power_eps(x, 0.0, eps), i.e., zero power at zero and finite x

    sv2 = np.logspace(-6, 0, 51)

    exact_f_check2 = np.log(1.0 + sv2/eps)/np.log(1.0 + 1/eps)
    exact_fd_check2 = 1/(1.0 + sv2/eps)/eps/np.log(1.0 + 1/eps)

    f2, fd2 = rlplib.power_eps(sv2, 0.0, eps)

    assert np.allclose(f2, exact_f_check2)
    assert np.allclose(fd2, exact_fd_check2)

    

def test_power_eps2():

    eps = 1e-6
    
    sv1 = np.logspace(-6, 0, 51)

    # check that numba compiled version equals python version for N = 0

    numba_f1, numba_fd1 = rlplib.power_eps( sv1, 0.0, eps)
    python_f1, python_fd1 = rlplib.power_eps.py_func( sv1, 0.0, eps)

    assert np.allclose(numba_f1, python_f1)
    assert np.allclose(numba_fd1, python_fd1)
    
    # check that numba compiled version equals python version for N = 0.3

    numba_f2, numba_fd2 = rlplib.power_eps( sv1, 0.3, eps)
    python_f2, python_fd2 = rlplib.power_eps.py_func( sv1, 0.3, eps)

    assert np.allclose(numba_f2, python_f2)
    assert np.allclose(numba_fd2, python_fd2)
    
    # check that numba compiled version equals python version for N = 1.0

    numba_f3, numba_fd3 = rlplib.power_eps( sv1, 1.0, eps)
    python_f3, python_fd3 = rlplib.power_eps.py_func( sv1, 1.0, eps)

    assert np.allclose(numba_f3, python_f3)
    assert np.allclose(numba_fd3, python_fd3)
    
    # check that numba compiled version equals python version for N = 3.1

    numba_f4, numba_fd4 = rlplib.power_eps( sv1, 3.1, eps)
    python_f4, python_fd4 = rlplib.power_eps.py_func( sv1, 3.1, eps)

    assert np.allclose(numba_f4, python_f4)
    assert np.allclose(numba_fd4, python_fd4)
    
def test_power_eps3():

    # Take a very small value for eps
    eps = 1e-12
    
    sv1 = np.logspace(-6, 0, 51)

    # check that numba compiled version equals python version for N = 0

    numba_f1, numba_fd1 = rlplib.power_eps( sv1, 0.0, eps)
    python_f1, python_fd1 = rlplib.power_eps.py_func( sv1, 0.0, eps)

    assert np.allclose(numba_f1, python_f1)
    assert np.allclose(numba_fd1, python_fd1)
    
    # check that numba compiled version equals python version for N = 0.3

    numba_f2, numba_fd2 = rlplib.power_eps( sv1, 0.3, eps)
    python_f2, python_fd2 = rlplib.power_eps.py_func( sv1, 0.3, eps)

    assert np.allclose(numba_f2, python_f2)
    assert np.allclose(numba_fd2, python_fd2)
    
    # check that numba compiled version equals python version for N = 1.0

    numba_f3, numba_fd3 = rlplib.power_eps( sv1, 1.0, eps)
    python_f3, python_fd3 = rlplib.power_eps.py_func( sv1, 1.0, eps)

    assert np.allclose(numba_f3, python_f3)
    assert np.allclose(numba_fd3, python_fd3)
    
    # check that numba compiled version equals python version for N = 3.1

    numba_f4, numba_fd4 = rlplib.power_eps( sv1, 3.1, eps)
    python_f4, python_fd4 = rlplib.power_eps.py_func( sv1, 3.1, eps)

    assert np.allclose(numba_f4, python_f4)
    assert np.allclose(numba_fd4, python_fd4)
    
def test_power_eps4():

    eps = 0.0
    
    sv1 = np.logspace(-6, 0, 51)

    # check that for eps=0 power_eps equals standard np.power for N = 0.0

    numba_f1, numba_fd1 = rlplib.power_eps( sv1, 0.0, eps)
    numpy_f1, numpy_fd1 = np.power( sv1, 0.0), np.power(sv1, -1.0)*0.0

    assert np.allclose(numba_f1, numpy_f1)
    assert np.allclose(numba_fd1, numpy_fd1)
    
    # check that for eps=0 power_eps equals standard np.power for N = 0.3

    numba_f2, numba_fd2 = rlplib.power_eps( sv1, 0.3, eps)
    numpy_f2, numpy_fd2 = np.power( sv1, 0.3), np.power( sv1, 0.3-1.0 )*0.3

    assert np.allclose(numba_f2, numpy_f2)
    assert np.allclose(numba_fd2, numpy_fd2)
    
    # check that for eps=0 power_eps equals standard np.power for N = 1.0

    numba_f3, numba_fd3 = rlplib.power_eps( sv1, 1.0, eps)
    numpy_f3, numpy_fd3 = np.power( sv1, 1.0), np.power( sv1, 1.0-1.0 )*1.0

    assert np.allclose(numba_f3, numpy_f3)
    assert np.allclose(numba_fd3, numpy_fd3)
    
    # check that for eps=0 power_eps equals standard np.power for N = 3.1

    numba_f4, numba_fd4 = rlplib.power_eps( sv1, 3.1, eps)
    numpy_f4, numpy_fd4 = np.power( sv1, 3.1), np.power( sv1, 3.1-1.0 )*3.1

    assert np.allclose(numba_f4, numpy_f4)
    assert np.allclose(numba_fd4, numpy_fd4)
    
def test_power_eps5():

    eps = 1e-8

    
    sv1 = np.logspace(-6, 0, 51)

    # check that N is clipped to zero when N < 0

    numba_f1, numba_fd1 = rlplib.power_eps( sv1, -3.0, eps)
    ref_f1, ref_fd1 = rlplib.power_eps( sv1, 0.0, eps)

    assert np.allclose( numba_f1, ref_f1 )
    assert np.allclose( numba_fd1, ref_fd1 )

    # check that s is clipped to zero when s < 0, N = 3.0
    sv2 = -np.logspace(-6, 0, 51)

    numba_f2, numba_fd2 = rlplib.power_eps( sv2, 3.0, eps)
    ref_f2, ref_fd2 = rlplib.power_eps( 0.0, 3.0, eps)

    assert np.allclose( numba_f2, ref_f2 )
    assert np.allclose( numba_fd2, ref_fd2 )

    # check that s is clipped to zero when s < 0, N = 1.0
    sv3 = -np.logspace(-6, 0, 51)

    numba_f3, numba_fd3 = rlplib.power_eps( sv3, 1.0, eps)
    ref_f3, ref_fd3 = rlplib.power_eps( 0.0, 1.0, eps)

    assert np.allclose( numba_f3, ref_f3 )
    assert np.allclose( numba_fd3, ref_fd3 )

    # check that s is clipped to zero when s < 0, N = 0.5
    sv4 = -np.logspace(-6, 0, 51)

    numba_f4, numba_fd4 = rlplib.power_eps( sv4, 0.5, eps)
    ref_f4, ref_fd4 = rlplib.power_eps( 0.0, 0.5, eps)

    assert np.allclose( numba_f4, ref_f4 )
    assert np.allclose( numba_fd4, ref_fd4 )

    # check that s is clipped to zero when s < 0, N = 0.0
    sv5 = -np.logspace(-6, 0, 51)

    numba_f5, numba_fd5 = rlplib.power_eps( sv5, 0.0, eps)
    ref_f5, ref_fd5 = rlplib.power_eps( 0.0, 0.0, eps)

    assert np.allclose( numba_f5, ref_f5 )
    assert np.allclose( numba_fd5, ref_fd5 )

    # check that s is clipped to zero when s < 0, and N to zero when N = -1.0
    sv6 = -np.logspace(-6, 0, 51)

    numba_f6, numba_fd6 = rlplib.power_eps( sv6, -1.0, eps)
    ref_f6, ref_fd6 = rlplib.power_eps( 0.0, 0.0, eps)

    assert np.allclose( numba_f6, ref_f6 )
    assert np.allclose( numba_fd6, ref_fd6 )
