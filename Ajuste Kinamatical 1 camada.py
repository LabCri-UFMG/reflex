import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.signal import correlate
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
#--------------------------------------------------------------TRABALHANDO COM OS DADOS--------------------------------------------------------------------------------------
r0=2.82e-5 #espalhamento de thompson
labda=1.54
k=2*np.pi/labda# wave vector 
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
    
x_trans=np.zeros(len(x_dados1))

somatorio=0
for i in x_dados1:
    x_trans[somatorio]=2*k*np.sin(i/2*np.pi/180) # o angulo esta sobrado 2\theta 
    somatorio+=1

#------------------------------------------------------------Selecioando o angulo critico -----------------------------------------------------------------------
# Plot the data
# plt.scatter(x_trans, y_dados,marker= '.', label='Data', s=1)
# plt.xlabel('$sin \\theta\\degree $ ')
# plt.ylabel('Intensidade')
# plt.yscale('log')
# plt.grid()
# plt.title('Angulo critico')

# # Allow the user to click on points
# selected_points = plt.ginput(n=2, timeout=0)

# # Print the selected points
# # print("Selected Points:")
# # for point in selected_points:
# #     print(f"X: {point[0]}, Y: {point[1]}")

# # Find indices of the selected points in the original data
# if len(selected_points) == 2:
#     start_point, end_point = selected_points

#     # Find the indices of the closest points in the original data
#     start_index = np.argmin(np.abs(np.array(x_trans) - start_point[0]))
#     end_index = np.argmin(np.abs(np.array(x_trans) - end_point[0]))

#     # Print the indices
#     # print("\nIndices of the Selected Points in the Original Data:")
#     # print(f"Start Index: {start_index}, End Index: {end_index}")

#     # Plot the selected points and the indices
#     plt.scatter(*zip(*selected_points), color='red', label='Selected Points')
#     plt.scatter([x_trans[start_index], x_trans[end_index]], [y_dados[start_index], y_dados[end_index]], color='green', marker='s', label='Indices')
#     plt.legend()
#         # Create a subset of x_dados between the selected indices
#     parte = x_trans[start_index:end_index + 1]
#     partey=y_noramalizado[start_index:end_index + 1]
#     print("\nSubset of x_dados:")
#     print(parte)
#     angulo_cri=np.mean(parte)/(2*np.pi)
#     print("valor angulo critico", angulo_cri,'\n Qc=',np.mean(parte)) #pegando a media 
# plt.show()
#----------------------------------------------------Outro metodo para encontra o angulo critico e o Delta------------------------------------------------------
# Plot the data
# Apply Savitzky-Golay smoothing
window_size = 15  # Adjust the window size according to your data
order = 1  # Adjust the order of the polynomial
y_smoothed = savgol_filter(y_noramalizado, window_size, order)

plt.plot(x_trans, y_smoothed, label='Signal')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Finding Peaks in a Signal')

# Find peaks using scipy's find_peaks
peaks, _ = find_peaks(y_smoothed, height=1e-4)

# Plot the identified peaks
plt.yscale('log')
plt.plot(x_trans[peaks],y_smoothed[peaks], 'rx', label='Peaks')
plt.legend()

plt.show()

# Print the indices and values of the identified peaks
print("Indices of Peaks:", peaks)
print("Values of Peaks:", y_smoothed[peaks])
picos=x_trans[peaks]
print('picos em X', picos)
angulo_cri=(picos[1]+picos[2])/(4*k)
print('angulo critico', angulo_cri, '\n','Qc=',angulo_cri*(2*k))

#------------------------------------------------------------------encontrando o Delta-----------------------------------------------------------------------

def ajuste(x,a,b):
    return a*x+b

# Perform the linear least squares fit
params, covariance = curve_fit(ajuste, picos,np.arange(len(picos)))

# Extract the optimized parameters and standard errors
a, b = params
std_err_a, std_err_b = np.sqrt(np.diag(covariance))

# Print the optimized parameters and standard errors
print("Optimized Parameters:")
print("a:", a, "±", std_err_a)
print("b:", b, "±", std_err_b)

y_fit = ajuste(picos, a, b)

print(picos)
plt.scatter( picos,np.arange(len(picos)))
plt.plot(picos, y_fit, label='Linear Fit', color='red')
plt.xlabel('picos em X')
plt.ylabel('indice')
plt.title('linearização dos picos')
plt.show()
# Calculate the differences between consecutive peaks
delta_ang= np.diff(picos)

soma=0
delta=np.zeros(len((delta_ang)))
for i in delta_ang:
    delta[soma]=labda/(i/(k))
    soma+=1
print('delta',delta)
Delta= delta[3:-1]
print(Delta)
Delta=sum(Delta)/len(Delta)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#chute incial para os parametros (Em Amestrogns)

#rho=9.86 #densidade eletronica valor tabelando mas n encotrado nesses dados 
#b=0.0409 #parametro b_mu n é cte relacionado a absorção do material
#Delta--> espesura
#sigma--> rugosidade
#rho=2.74

rho=(k*angulo_cri)**2/(4*np.pi*r0)

Delta1=Delta
sigma1=9
#b_mu_chute=1/(np.sin(angulo_cri))**2#NÃO POSSO APENAS USAR ESSA FORMULA VER PAGINA  93
b_mu_chute=0
print('Chute', rho,b_mu_chute, angulo_cri)

#------------------------------------------------------------------------FUNÇÃO----------------------------------------------------------------------------------


# A função kinematical para o ajuste
def kinematical(Q, Delta, b,sigma,rho):#é o eixo y
    Qc =4*(np.pi*r0*rho)**.5 #angulo critico
    q=Q/Qc
    # b=(-np.real(q)**2+np.imag(q)**2+1)/(2j)
    # print (b)
    Qp = Qc * (q**2 - 1 + 2 * 1j * b)**0.5
    rQ = (Q - Qp) / (Q + Qp)
    r_slab = (rQ * (1 - np.exp(Qp * 1j * Delta))) / (1 - rQ**2 * np.exp(Qp * 1j * Delta))
    r_slab1 = r_slab * np.exp(-Q ** 2 * sigma ** 2 / 2)
    eixoy = np.abs(r_slab1 * np.conj(r_slab1))
    return np.transpose(eixoy) 

# Define the residual function for least squares 9.68
def fun_resid(params, x_trans, y_smoothed):
    Delta, b,sigma,rho = params
    return kinematical(x_trans, Delta,b, sigma,rho) - y_smoothed

#Dando o chute inicial

chute_inicial = [Delta1,b_mu_chute,sigma1,rho]
print('chute', chute_inicial)
print(type(chute_inicial))
# Perform least squares optimization
res_ajuste = least_squares(fun_resid, chute_inicial, args=(x_trans, y_smoothed))

# Ajustar a função aos dados
parametros_otimizados = res_ajuste.x

y_ajustado = kinematical(x_trans, *parametros_otimizados) 
y_teste=kinematical(x_trans,346.40,1.57e-6,8,2.246)

# # # Calculate cross-correlation
# correlation = correlate(y_noramalizado, y_ajustado, mode='full')
# shift = np.argmax(correlation) - len(x_trans) + 1

# # Shift the second curve based on the cross-correlation result
# y2_values_aligned = np.roll(y_noramalizado, -shift)

print("Parâmetros otimizados:")
print("Delta:", parametros_otimizados[0])
print("b:", parametros_otimizados[1])
print("sigma:", parametros_otimizados[2])
print("rho:", parametros_otimizados[3])


#GRAFICO-------------------------------------------------------------------------------------


texto ='Parametros ajustados\n'\
       f'Espessura ($\\AA$): {parametros_otimizados[0]}\n' \
       f'Coedificente de absorção : {parametros_otimizados[1]}\n'\
       f'Rugosidadeo ($\\AA$) : {parametros_otimizados[2]}\n'\
       f'densidade eletronica ($\\rho/\\AA$): {parametros_otimizados[3]}\n'\
       'Parametros De entrada\n' \
       f'Espessura ($\\AA$): {chute_inicial[0]}\n' \
       f'Coeficiente de Absorção : {chute_inicial[1]}\n'\
       f'Rugosidade ($\\AA$): {chute_inicial[2]}\n'\
       f'densidade eletronica ($\\rho/\\AA$): {chute_inicial[3]}' 
# texto1 ='Parametros De entrada\n' \
#        f'Espessura ($\\AA$): {chute_inicial[0]}\n' \
#        f'Rugosidade ($\\AA$): {chute_inicial[1]}\n'\
#        f'Coeficiente de Absorção : {chute_inicial[2]}\n'\
#        f'densidade eletronica ($\\rho/\\AA$): {chute_inicial[3]}'       
#plt.text(1,1,f'Delta: {parametros_otimizados[0]}', fontsize=5, color='black')

plt.text(0.01, 0.015,texto ,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
         fontsize=7, color='black', transform=plt.gca().transAxes)

# plt.text(0.5, 0.65,texto1 ,
#          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
#          fontsize=7, color='black', transform=plt.gca().transAxes)


#plt.plot(x_trans,y_ajustado, label='Ajuste1', color='red')





# plt.plot(x_trans1,y2_values_aligned, label='Ajuste', color='red')
plt.plot(x_trans, y_ajustado, label='normalizado sem o shift', color='red')


# plt.scatter(x_dados1,y_dados, label='dados')
#plt.plot(x_trans, y2_values_aligned, label='Ajuste',marker='_', color='red', markersize=1)
plt.plot(x_trans, y_teste,color='green')
plt.scatter(x_trans,  y_smoothed, label='Dados', marker='o', s=7)
plt.xlabel('$2ksin \\theta(\AA^{-1})$ ') #(angulo em rad)
plt.yscale('log')
plt.ylabel('$|r_{slab}|^2$')
plt.legend()
plt.title('Curva ajustada')
plt.grid()
# Set the figure size in centimeters
width_cm = 25
height_cm = 12
dpi = 2.54  # 1 inch = 2.54 cm
fig = plt.gcf()
fig.set_size_inches(width_cm / dpi, height_cm / dpi)
plt.show()
