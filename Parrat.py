#n sei o q to fazendo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import minimize

lambda_=1.54
r0=2.82e-5 #espalhamento de thompson
k = 2 * np.pi / lambda_
background=1e-8
A=1.3
data = pd.read_excel("Dados Refletometria João (1).xlsx", sheet_name="#1_HfO2_XRRfino", decimal=",")
#Trabalhando com os dados
x_dados1= data["X"].to_numpy()
y_dados = data["Y"].to_numpy()
norma=max(y_dados)
y_noramalizado=np.zeros(len(y_dados))

cont=0
for i in y_dados:
    y_noramalizado[cont]=i/norma
    cont+=1
    
#Suavizando a curva 
window_size = 15  # Adjust the window size according to your data
order = 1  # Adjust the order of the polynomial
y_smoothed = savgol_filter(y_noramalizado, window_size, order)

x_trans=np.zeros(len(x_dados1))

somatorio=0
for i in x_dados1:
    x_trans[somatorio]=2*k*np.sin(i/2*np.pi/180) # o angulo esta sobrado 2\theta 
    somatorio+=1
print('------------->',max(x_trans), min(x_trans))


# Define the Parratt function
def parratt(params, Q):
    d1, d2, sigma1, sigma2,rhoA,muA,rhoB,muB = params
    N=10
    bl = np.array([rhoA * r0 + 1j * muA, rhoB * r0 + 1j * muB])
    sld = np.array([bl[0], bl[1]])
    sld=np.tile(sld,N)
    sld=np.append(sld,0.1e-20)
    
    
    d=np.array([d1, d2])
    d=np.tile(d,N)
    sigma = np.array([sigma1, sigma2])
    sigma=np.tile(sigma,N+1)
    delta = lambda_**2 * np.real(sld) / (2 * np.pi)
    beta = lambda_ / (4 * np.pi) * np.imag(sld)
    n = len(sld)
    
    #n = len(sld+1)
    nu = 1 - delta + 1j * beta
    
    Q = np.reshape(Q, (1, len(Q)))
    x = np.arcsin(Q / (2 * k))
    Qp = np.zeros((n + 1, len(Q[0])), dtype=complex)
    for j in range(n):
        Qp[j-1, :] = np.sqrt(Q[0]**2 - 8 * k**2 * delta[j-1] + 1j * 8 * k**2 * beta[j-1])
    Qp = np.vstack((Q, Qp))
    r = np.zeros((n, len(Q[0])), dtype=complex)
    for j in range(n):
        r[j-1, :] = ((Qp[j-1, :] - Qp[j, :]) / (Qp[j-1, :] + Qp[j, :])) * np.exp(-0.5 * (Qp[j-1, :] * Qp[j, :]) * sigma[j]**2)
    RR = r[0, :]
    if n > 1:
        R = np.zeros((n-1, len(Q[0])), dtype=complex)
        R[0, :] = (r[n - 2, :] + r[n - 1, :] * np.exp(1j * Qp[n-1, :] * d[n - 2])) / (1 + r[n - 2, :] * r[n - 1, :] * np.exp(1j * Qp[n-1, :] * d[n - 2]))
    if n > 2:
        for j in range(2, n):
            R[j - 1, :] = (r[n - j - 1, :] + R[j - 2, :] * np.exp(1j * Qp[n - j, :] * d[n - j-1])) / (1 + r[n - j - 1, :] * R[j - 2, :] * np.exp(1j * Qp[n - j , :] * d[n - j-1]))
    if n == 1:
        RR = r[0, :]
    else:
        RR = R[n - 2, :]
    RR = np.abs(RR)**2
    return RR*A+background

# Define a residual function for least_squares
def fun_resid(params, x_trans, y_smoothed):
    RR_model = parratt(params, x_trans)
    return RR_model - y_smoothed

#chute inicial

# sld1_chute=10
# sld2_chute=8
d_chute=46.44
d2_chute=1.13
sigma1_chute=7.13
sigma2_chute=5.62
rhoA_chute=0.54
muA_chute=0.0000033235
rhoB_chute=0.10
muB_chute=0.000000033235

#sld1, sld2, d1, d2, sigma1, sigma2 = params
chute_inicial=[d_chute,d2_chute,sigma1_chute,sigma2_chute,rhoA_chute,muA_chute,rhoB_chute,muB_chute]
#p0 = [10, 0.699, 10, 40, 33.235e-6, 1.399e-6]





# Use least_squares to find the optimal parameters
res = least_squares(fun_resid, chute_inicial, args=(x_trans, y_smoothed))

# res.x contains the optimal parameters
d1_opt, d2_opt, sigma1_opt, sigma2_opt,rhoA_opt,muA_opt,rhoB_opt,muB_opt = res.x
print(type(res.x))
parametros_otimizados=res.x


print("Parâmetros otimizados:")
print("d1_opt:", parametros_otimizados[0])
print("d2_opt:", parametros_otimizados[1])
print("sigma:", parametros_otimizados[2])
print("sigma2_opt:", parametros_otimizados[3])
print("rhoA_opt:", parametros_otimizados[4])
print("muA_opt:", parametros_otimizados[5])
print("rhoB_opt:", parametros_otimizados[6])
print("muB_opt:", parametros_otimizados[7])

# Calculate the fitted reflectivity using the optimal parameters
lista=np.array([46.44,1.13,7.13,5.62,0.54,0.0000033235,0.10,0.000000033235])
y_teste=parratt(lista,x_trans)
R_parratt_opt = parratt(res.x, x_trans)


plt.figure()
plt.scatter(x_trans, y_smoothed, marker='o', label='Experimental data',s=1)
plt.plot(x_trans, R_parratt_opt, 'r', label='Fitted reflectivity')
plt.plot(x_trans,y_teste,'r', label='test', color='black')
plt.yscale('log')
plt.ylabel('Reflectivity')
plt.xlabel('$Q=2ksin \\theta$ ')
plt.title('Parratt')
plt.legend()
plt.grid()

width_cm = 25
height_cm = 12
dpi = 2.54  # 1 inch = 2.54 cm
fig = plt.gcf()
fig.set_size_inches(width_cm / dpi, height_cm / dpi)

plt.show()
