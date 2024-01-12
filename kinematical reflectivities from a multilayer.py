# kinematical reflectivities from a multilayer
#Specific case of W/Si, 10 bilayers of [10 Angs W, 40 Angs Si]

import numpy as np
import matplotlib.pyplot as plt


r0=2.82e-5 #Thompson scattering length in Angs
Q=np.arange(0.01,0.3,0.001) # Wavevector transfer in 1/Angs
lambd = 1.54 # wavelength in Angs
rhoA=4.678 #density of W
rhoB=0.699 #density of Si
muA=33.235e-6 # density and absorption coefficient of W
muB=1.399e-6 # density and absorption coefficient of Si

bl=[rhoA*r0+1j*muA, rhoB*r0+1j*muB] # bilayer scattering factor
dbl=[10, 40] # bilayer d-spacings
#ml=[bl, bl, bl, bl, bl, bl, bl, bl, bl, bl, 0.1e-20] # multilayer scattering factor
ml = np.tile(bl, 10)
ml = np.append(ml, 0.1e-20)
# bilayer d-spacings
dbl = np.array([10, 40])
#dml=[dbl, dbl, dbl, dbl, dbl, dbl, dbl, dbl, dbl, dbl] # multilayer d-spacings
dml = np.tile(dbl, 10)
sml=np.zeros(290) # roughness at each interface


def kinematicalR(Q, lambda_, sld, sigma, N, Lambda, Gamma):
    muA = np.imag(sld[0])
    muB = np.imag(sld[1]) 
    Dsld = np.real(sld[0]) - np.real(sld[1])
    
    
    # Prevent division by zero in zeta
    zeta = np.where(Q != 0, Q / (2 * np.pi) * Lambda, 1e-10)
    beta = 2 * Lambda * Lambda * (muA * Gamma + muB * (1 - Gamma)) / lambda_ / zeta
    r_1 = -2j * Dsld * Lambda * Lambda * Gamma / zeta
    r_1 = r_1 * np.sin(np.pi * Gamma * zeta) / (np.pi * Gamma * zeta)
    r_N = r_1 * (1 - np.exp(1j * 2 * np.pi * zeta * N) * 
                 np.exp(-beta * N)) / (1 - np.exp(1j * 2 * np.pi * zeta) * 
                                       np.exp(-beta))
    
    
    r_N =  [r_N * np.exp(-((Q*sigma) ** 2 / 2))  ] 
    #print(r_N)
    R = r_N * np.conj(r_N)
    #print(np.conj(r_N).shape)
    #print(np.conj(r_N))
    return  np.transpose(R.real)#R.real #np.transpose(R.real)
# sld --> scattering length density 1/Angs^2


# Calculate kinematical reflectivity

N = 10 #o q são esses parametros 
Lambda = 50 #o q são esses parametros 
Gamma = 0.2 #o q são esses parametros 
R_kinematical = kinematicalR(Q, lambd, bl, sml, N, Lambda, Gamma)

# Create a figure and plot the results
plt.figure(figsize=(10, 6))
plt.plot(Q, R_kinematical, label='Kinematical Reflectivity')
plt.yscale('log')
plt.xlabel('Wavevector transfer Q (Å^{-1})')
plt.ylabel('Reflectivity')
plt.legend()
plt.title('Kinematical')
plt.show()
