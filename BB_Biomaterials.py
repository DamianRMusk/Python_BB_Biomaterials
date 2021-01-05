#!/usr/bin/env python

"""
The following program, written by D. R. Musk, takes the Bergström-Boyce model with a Williams-Landel-Ferry shift 
representation for time-temperature superposition, evaluating stress-strain behavior in polymers (given 
biomaterial data: cellulose, xylan hemicellulose, and high-methoxyl pectin) under an extra climate stressor. 
Procedure and inverse Langevin approximation inspired by DOI: 10.1016/s0022-5096(97)00075-6.
"""

import scipy.integrate 
from matplotlib.pyplot import *
from numpy import *
from numpy.linalg import *
from pylab import *
import scipy.optimize

def ramp(x): # Mathematical ramp functions
    return (x + abs(x)) / 2.0

def inv_Langevin(x): # Mathematical inverse Langevin function
    EPS = spacing(1)
    if type(x) == float or type(x) == float64:
        if x >= 1-EPS: x = 1 - EPS
        if x <= -1+EPS: x = -1 + EPS
        if abs(x) < 0.839:
            return 1.31435 * tan(1.59*x) + 0.911249*x 
        return 1.0 / (sign(x) - x)
    x[x >= 1-EPS] = 1 - EPS
    x[x <= -1+EPS] = -1 + EPS 
    res = zeros(size(x))
    index = abs(x) < 0.839 
    res[index] = 1.31435 * tan(1.59*x[index]) + 0.911249*x[index]
    index = abs(x) >= 0.839 
    res[index] = 1.0 / (sign(x[index]) - x[index])
    return res 

def EC_3D(stretch, param): # Function defining used material tensors in eight chain model, completing procedure item 1
    L1 = stretch[0]
    L2 = stretch[1] 
    L3 = stretch[2] 
    F = array([[L1, 0, 0], [0, L2, 0], [0, 0, L3]], dtype='float64')
    J = det(F) 
    bstar = J**(-2.0/3.0) * dot(F, F.T) 
    lamChain = sqrt(trace(bstar)/3)
    devbstar = bstar - trace(bstar)/3 * eye(3) 
    return param[0]/(J*lamChain) * inv_Langevin(lamChain/param[1]) / inv_Langevin(1/param[1]) * devbstar + param[2]*(J-1) * eye(3)

def to_vec(A): # Helper vector conversion function
    return array([A[0][0], A[1][1], A[2][2]], dtype='float64')

def uniaxial_stress_visco(model, timeVec, trueStrainVec, params, aT): # Simulates uniaxial stress conditions, completing procedure item 2
    stress = zeros(len(trueStrainVec))
    lam2_1 = 1.0
    FBv1 = array([1.0, 1.0, 1.0], dtype='float64')
    for i in range(1, len(trueStrainVec)):
        time0 = aT*timeVec[i-1]
        time1 = aT*timeVec[i]
        lam1_0 = exp(trueStrainVec[i-1])
        lam1_1 = exp(trueStrainVec[i])
        lam2_0 = lam2_1
        F0 = array([lam1_0, lam2_0, lam2_0], dtype='float64')
        F1 = array([lam1_1, lam2_1, lam2_1], dtype='float64')
        FBv0 = FBv1.copy()
        calcS22Abs = lambda x: abs(model(F0, array([lam1_1, x, x], dtype='float64'), FBv0, time0, time1, params)[0][1])
        lam2_1 = scipy.optimize.fmin(calcS22Abs, x0=lam2_0, xtol=1e-9, ftol=1e-9, disp=False)
        res = model(F0, array([lam1_1, lam2_1, lam2_1], dtype='float64'), FBv0, time0, time1, params)
        stress[i] = res[0][0]
        FBv1 = res[1]
    return stress

def BB_timeDer_3D(Fv, t, params, time0, time1, F0, F1): # Simulates uniaxial stress conditions, completing procedure item 3
    mu, lamL, kappa, s, xi, C, tauBase, m, tauCut = params[:9]
    F = F0 + (t-time0) / (time1-time0) * (F1 - F0)
    Fe = F / Fv 
    Stress = toVec(EC_3D(Fe, [s*mu, lamL, kappa]))
    devStress = Stress - sum(Stress)/3
    tau = norm(devStress)
    lamCh = sqrt(sum(Fv*Fv)/3.0)
    lamFac = lamCh - 1.0 + xi
    gamDot = lamFac**C * (ramp(tau/tauBase-tauCut)**m)
    prefac = 0.0
    if tau > 0: prefac = gamDot / tau 
    FeInv = array([1.0, 1.0, 1.0], dtype='float64') / Fe 
    FvDot = prefac * (FeInv * devStress * F)
    return FvDot 

def BB_3D(F0, F1, FBv0, time0, time1, params): # Elastic component function solver and final component
    muA, lamL, kappa, s = params[:4]
    StressA = to_vec(EC_3D(F1, [muA, lamL, kappa]))
    FBv1 = scipy.integrate.odeint(BB_timeDer_3D, FBv0, array([time0, time1], dtype='float64'), args=(params, time0, time1, F0, F1))[1]
    FBe1 = F1 / FBv1 
    StressB = to_vec(EC_3D(FBe1, [s*muA, lamL, kappa]))
    Stress = StressA + StressB 
    return (Stress, FBv1)

N = 100 
timeVec = linspace(0, 10.0, N)
trueStrain = linspace(0, 0.2, N)
celluloseParams = [5.6, 5.597, 20, 2.0, 0.05, -1.0, 0.5, 8.0, 0.01]
xylanParams = [0.3, 15.553, 0.5, 2.0, 0.05, -1.0, 0.5, 8.0, 0.01]
highMethoxylPectinParams = [3.25, 12.177, 10.52, 2.0, 0.05, -1.0, 0.5, 8.0, 0.01]
temp = 4.51
initialTemp = 3.45
aT = exp(17.4*(temp-initialTemp)/(51.6 + temp-initialTemp))
trueCelluloseStressSansTempChange = uniaxial_stress_visco(BB_3D, timeVec, trueStrain, celluloseParams, 1)
trueCelluloseStressWithTempChange = uniaxial_stress_visco(BB_3D, timeVec, trueStrain, celluloseParams, aT)
trueXylanStressSansTempChange = uniaxial_stress_visco(BB_3D, timeVec, trueStrain, xylanParams, 1)
trueXylanStressWithTempChange = uniaxial_stress_visco(BB_3D, timeVec, trueStrain, xylanParams, aT)
truePectinStressSansTempChange = uniaxial_stress_visco(BB_3D, timeVec, trueStrain, highMethoxylPectinParams, 1)
truePectinStressWithTempChange = uniaxial_stress_visco(BB_3D, timeVec, trueStrain, highMethoxylPectinParams, aT)

if __name__ == "__main__":
    plot(trueStrain, trueCelluloseStressSansTempChange, 'b-', label='Cellulose without temp. conditions')
    plot(trueStrain, trueCelluloseStressWithTempChange, 'r-', label='Cellulose with temp. conditions')
    plot(trueStrain, trueXylanStressSansTempChange, 'g-', label='Xylan without temp. conditions')
    plot(trueStrain, trueXylanStressWithTempChange, 'm-', label='Xylan with temp. conditions')
    plot(trueStrain, truePectinStressSansTempChange, 'c-', label='HM Pectin without temp. conditions')
    plot(trueStrain, truePectinStressWithTempChange, 'y-', label='HM Pectin with temp. conditions')
    plt.legend(loc='upper left')
    xlabel('True Strain')
    ylabel('True Stress (MPa)')
    grid('on')
    show()
