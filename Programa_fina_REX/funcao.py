import numpy as np
def abeles_python(q, coefs):
    M_4PI=4*np.pi
    nlayers = int(coefs[0])  # Número de camadas
    reflectivity = np.zeros_like(q)  # Inicializa a refletividade com zeros

    for idx, q_val in enumerate(q):
        q2 = (q_val ** 2) / 4.0
        kn = q_val / 2.0
        mrtot00 = 1.0
        mrtot11 = 1.0
        mrtot01 = 0.0
        mrtot10 = 0.0

        for ii in range(nlayers + 1):
            if ii == nlayers:
                SLD_Ar=coefs[2]*2.82*10**(-5) #Convertendo a densidade eletronica para SLD                
                SLD_meio=coefs[4]*2.82*10**(-5) #Convertendo a densidade eletronica para SLD
                ISLD=coefs[5]/(1.54056*2) #Conveertendo a densiadade eletrinica para ISLD, ja é do subestrato
                SLD_real = M_4PI * (SLD_meio - SLD_Ar) 
                SLD_imag = M_4PI * ((ISLD) + 1e-30)  *(1e-6) 
                rough = -2 * coefs[7] ** 2
            else:
                SLD_Ar=coefs[2]*2.82*10**(-5)
                SLD_meio=coefs[4*ii+9]*2.82*10**(-5)
                ISLD=coefs[4*ii+10]/(1.54056*2)
                SLD_real = M_4PI * (SLD_meio - SLD_Ar) 
                SLD_imag = M_4PI * (np.abs(ISLD) + 1e-30) *(1e-6)  
                rough = -2 * coefs[4*ii + 11] ** 2
            
            SLD = SLD_real + 1j * SLD_imag
            k_next = np.sqrt(q2 - SLD)
            rj = (kn - k_next) / (kn + k_next)
            rj *= np.exp(kn * k_next * rough)

            if ii == 0:
                mrtot01 = rj
                mrtot10 = rj
            else:
                thick = coefs[4*(ii - 1) + 8]
                beta = np.exp(1j * kn * thick)

                mi00 = beta
                mi11 = 1.0 / beta
                mi10 = rj * mi00
                mi01 = rj * mi11

                p0 = mrtot00 * mi00 + mrtot10 * mi01
                p1 = mrtot00 * mi10 + mrtot10 * mi11
                mrtot00 = p0
                mrtot10 = p1

                p0 = mrtot01 * mi00 + mrtot11 * mi01
                p1 = mrtot01 * mi10 + mrtot11 * mi11
                mrtot01 = p0
                mrtot11 = p1

            kn = k_next

        r = mrtot01 / mrtot00
        r *= np.conj(r)
        reflectivity[idx] = np.real(r * coefs[1] + coefs[6])

    return reflectivity
