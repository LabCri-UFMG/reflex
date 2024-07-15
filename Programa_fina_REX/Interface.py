import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import mplcursors
import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk, ImageSequence
import customtkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from tkinterdnd2 import TkinterDnD, DND_FILES
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import os
from funcao import abeles_python

# Objective function for optimization
def objective_function(params, x, y):
    y_pred = abeles_python(x, params)
    return np.mean(np.abs(np.log1p(y_pred) - np.log1p(y)))

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("green")

class ReflectometryGUI(customtkinter.CTk, TkinterDnD.Tk):
    def __init__(self):
        customtkinter.CTk.__init__(self)
        TkinterDnD.Tk.__init__(self)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.loading_screen = None

        self.title("Reflectometry Curve Fitting")
        self.geometry("1100x580")
        self.initial_params = [2, 1.01582, 2.00, 0, 6.36, 0, 1.7e-5, 5.91437, 13.6488, 3.47, 0, 3] + [0] * 18
        self.params = self.initial_params.copy()
        self.params_tabs = []
        self.tab_names = []
        self.x = np.array([])
        self.y = np.array([])
        self.optimization_message = ""

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.loading_frame = None
        self.gif_path = r"C:/Users/emanu/Downloads/WhatsApp Video 2024-06-03 at 13.gif"
        self.gif_frames = []
        self.current_frame = 0
        self.animation_running = False

        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.drop)

        self.checkbox_vars = {i: tk.IntVar() for i in range(len(self.params))}

        self.init_ui()

    def on_closing(self):
        self.animation_running = False
        self.cancel_all_callbacks()
        super().destroy()

    def cancel_all_callbacks(self):
        if self.loading_screen is not None:
            try:
                self.loading_screen.after_cancel(self.loading_screen)
            except Exception as e:
                print(f"Error cancelling callbacks: {e}")

    def destroy(self):
        self.cancel_all_callbacks()
        super().destroy()

    def drop(self, event):
        self.data_path = event.data.strip('{}')
        if self.data_path.lower().endswith(('.txt', '.csv', '.xy')):
            self.show_loading_screen()
            threading.Thread(target=self.load_data, args=(self.data_path,)).start()
        else:
            print("Invalid file type. Please drop a valid data file (.txt or .csv).")

    def load_data(self, filename):
        try:
            if filename.lower().endswith('.xy'):
                data = pd.read_csv(filename, sep='\s+', header=None, names=['Q', 'Intensity'])
            else:
                data = pd.read_csv(filename)
            self.x = data['Q'].to_numpy()
            self.y = data['Intensity'].to_numpy()
            self.x = 2 * (2 * np.pi / 1.54056) * np.sin(np.deg2rad(self.x / 2))
            self.y = self.y / (1e8)
            self.update_plot()
        except Exception as e:
            print(f"Failed to load data file: {e}")
        finally:
            self.hide_loading_screen()

    def update_plot(self):
        self.ax.clear()
        scatter = self.ax.scatter(self.x, self.y, label='Experimental Data', s=0.7, color='darkgreen')
        y_pred = abeles_python(self.x, self.params)
        self.ax.plot(self.x, y_pred, label='Chute', color='navy')
        self.ax.set_xlabel('Q ($\\AA^{-1}$)')
        self.ax.set_ylabel('Refletividade')
        self.ax.set_title('Curva de Refletometria')
        self.ax.set_yscale('log')
        self.ax.grid(True, which="both", ls=":")
        self.ax.legend()
        self.canvas.draw()
        cursor = mplcursors.cursor(scatter)

    def angulo_critico(self):
        self.show_loading_screen()
        threading.Thread(target=self.perform_angulo_critico).start()

    def perform_angulo_critico(self):
        x = [i for i in self.x if i < 0.05]
        y = [self.y[idx] for idx, i in enumerate(self.x) if i < 0.05]
        inflection_point_x = None

        for i in range(1, len(y) - 5):
            if y[i] > y[i - 1] and y[i] > y[i + 1]:
                for j in range(1, 5):
                    if y[i + j] >= y[i + j - 1]:
                        break
                else:
                    inflection_point_x = x[i]
                    break

        if inflection_point_x is None:
            print("Não tem ponto de inflexão")
            self.hide_loading_screen()
            return

        Qc = inflection_point_x
        angulo = np.arcsin(Qc * 1.54056 / (4 * np.pi))

        fig, ax = plt.subplots(figsize=(5, 3))
        scatter = ax.scatter(x, y, label='Dados Experimentais', color='darkgreen')
        ax.axvline(x=Qc, color='red', linestyle='--', label=f'Ângulo Crítico {angulo}')
        ax.set_xlabel('Q ($\\AA^{-1}$)')
        ax.set_ylabel('Refletividade')
        ax.set_title('Ajuste do Ângulo Crítico')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls=":")
        ax.legend()

        new_window = customtkinter.CTkToplevel(self)
        new_window.title("Gráfico do Ângulo Crítico")
        new_window.geometry("600x400")
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        cursor = mplcursors.cursor(scatter)
        self.hide_loading_screen()

    def init_ui(self):
        # Configura as colunas e linhas da grade do layout principal
        self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure((2, 3), weight=0)
        self.grid_columnconfigure(2, weight=0) # é a 2 que eu quero manter fixo 
        self.grid_rowconfigure(2,weight=0)
        self.grid_columnconfigure(3, weight=0)
        self.grid_rowconfigure((0, 1), weight=1)

        # Cria e configura a moldura da barra lateral
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        # Adiciona um rótulo de logo na barra lateral
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="RefX", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Adiciona botões na barra lateral
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text='Carregar arquivos', command=self.load_data_from_dialog)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=(10, 5))

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text='Otimizar', command=self.optimize)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=(5, 10))

        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text='Autoajuste', command=self.auto_adjust)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=(5, 10))

        # Adiciona botões para aumentar e diminuir camadas
        increase_button = customtkinter.CTkButton(self.sidebar_frame, text="Aumentar Camadas", command=lambda: self.adjust_layers(1))
        increase_button.grid(row=4, column=0, padx=20, pady=(10, 5))

        decrease_button = customtkinter.CTkButton(self.sidebar_frame, text="Diminuir Camadas", command=lambda: self.adjust_layers(-1))
        decrease_button.grid(row=5, column=0, padx=20, pady=(5, 10))

        # Adiciona um rótulo para exibir a quantidade de camadas
        self.layer_label = customtkinter.CTkLabel(self.sidebar_frame, text=f": {int(self.params[0])}", font=customtkinter.CTkFont(size=15))
        self.layer_label.grid(row=4, column=1, padx=20, pady=(10, 5), sticky="w")

        # Adiciona um botão para calcular ângulo crítico
        self.angulo_critico_button = customtkinter.CTkButton(self.sidebar_frame, text='Ângulo crítico', command=self.angulo_critico)
        self.angulo_critico_button.grid(row=6, column=0, padx=20, pady=(5, 10))

        # Adiciona opções de tema
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Tema:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))

        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["System" ,"Light", "Dark"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))

        # Adiciona opções de escala da UI
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=10, column=0, padx=20, pady=(10, 0))

        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 20))

        # Cria e configura a moldura para o plot
        self.plotframe = customtkinter.CTkFrame(master=self, height=570, width=350, fg_color="white")
        self.plotframe.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # Configura o canvas do matplotlib para exibir gráficos
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plotframe)
        self.canvas.get_tk_widget().pack(side=customtkinter.TOP, fill=customtkinter.BOTH, expand=1)

        # Cria e configura a visão de abas
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        


        # Cria abas para parâmetros gerais e de suporte
        self.create_general_parameters_tab()
        self.create_backing_parameters_tab()
        # self.crate_parameters_tab(1)
        # self.add_parameters_tab(10)

        # Adiciona abas para parâmetros adicionais com base na quantidade de camadas
        for i in range(1, int(self.params[0]) + 1):
            self.add_parameters_tab(i)

        # Adiciona um botão para salvar em PDF
        self.save_button = customtkinter.CTkButton(self.sidebar_frame, text='Salvar em PDF', command=self.save_to_pdf)
        self.save_button.grid(row=12, column=0, padx=20, pady=(5, 10))


    def load_data_from_dialog(self):
        # Abre um diálogo para o usuário selecionar um arquivo
        filename = fd.askopenfilename()
        if filename:
            # Mostra a tela de carregamento e inicia uma thread para carregar os dados
            self.show_loading_screen()
            threading.Thread(target=self.load_data, args=(filename,)).start()

    def create_general_parameters_tab(self):
        # Cria uma aba para parâmetros gerais
        tab_name = "Parametros do meio"
        general_tab = self.tabview.add(tab_name)
        # Adiciona controles deslizantes na aba para ajustar parâmetros específicos
        self.create_slider(general_tab, 3, "Escala", 0, 2, 100, self.update_params(1), 1)
        self.create_slider(general_tab, 6, "Densidade eletrônica Ar", 0, 10, 100, self.update_params(2), 2)
        self.create_slider(general_tab, 9, "Absorção do Ar (1e6)", 0, 50, 100, self.update_params(3), 3)

    def create_backing_parameters_tab(self):
        # Cria uma aba para parâmetros do substrato
        tab_name = "Substrato"
        backing_tab = self.tabview.add(tab_name)
        # Adiciona controles deslizantes na aba para ajustar parâmetros específicos
        self.create_slider(backing_tab, 0, "Densidade eletrônica", 0, 10, 100, self.update_params(4), 4)
        self.create_slider(backing_tab, 3, "Absorção(1e6)", 0, 50, 100, self.update_params(5), 5)
        self.create_slider(backing_tab, 6, "Background", 1e-9, 1e-3, 0.001, self.update_params(6), 6)
        self.create_slider(backing_tab, 9, "Rugosidade (Å)", 0, 100, 100, self.update_params(7), 7)

    # def add_parameters_tab(self, layer_number):
    def add_parameters_tab(self,layer_number):

        # Cria uma nova aba para parâmetros da camada específica
        tab_name = f"Camada {layer_number}"
        # tab_name = "Camada "

        new_tab = self.tabview.add(tab_name)
        # Adiciona controles deslizantes na aba para ajustar parâmetros específicos
        self.create_slider(new_tab, 0, "Espessura (Å)", 0, 700, 100, self.update_params(4 * (layer_number - 1) + 8), 4 * (layer_number - 1) + 8)
        self.create_slider(new_tab, 3, "Densidade eletrônica", 0, 10, 100, self.update_params(4 * (layer_number - 1) + 9), 4 * (layer_number - 1) + 9)
        self.create_slider(new_tab, 6, "Absorção(1e6)", 0, 50, 100, self.update_params(4 * (layer_number - 1) + 10), 4 * (layer_number - 1) + 10)
        self.create_slider(new_tab, 9, "Rugosidade (Å)", 0, 50, 100, self.update_params(4 * (layer_number - 1) + 11), 4 * (layer_number - 1) + 11)
        self.params_tabs.append(new_tab)
        self.tab_names.append(tab_name)

    def remove_parameters_tab(self, layer_number):
        # Remove uma aba de parâmetros com base no número da camada
        tab_name = f"Camada {layer_number}"
        if tab_name in self.tab_names:
            self.tabview.delete(tab_name)
            self.params_tabs.pop()
            self.tab_names.remove(tab_name)

    def adjust_layers(self, change):
        # Ajusta o número de camadas, adicionando ou removendo abas conforme necessário
        new_layer_count = int(self.params[0]) + change
        if 0 <= new_layer_count <= 10:
            self.params[0] = new_layer_count
            self.layer_label.configure(text=f": {new_layer_count}")
            if change > 0:
                self.add_parameters_tab(new_layer_count)
            elif change < 0:
                self.remove_parameters_tab(new_layer_count + 1)
                for i in range(4):
                    self.params[4 * new_layer_count + 8 + i] = 0 #erro aqui 
            self.update_plot()
        else:
            print("Layer count must be between 0 and 10")

    def update_params(self, index):
        # Cria um callback para atualizar os parâmetros e o gráfico
        def callback(value):
            self.params[index] = float(value)
            self.update_plot()
        return callback

    def create_slider(self, tab, row, text, from_, to, steps, command, param_index):
        # Cria e configura um controle deslizante com rótulo e entrada de valor
        slider_text = customtkinter.CTkLabel(tab, text=text, font=customtkinter.CTkFont(size=20, weight="bold"))
        slider_text.grid(row=row, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        slider_frame = customtkinter.CTkFrame(tab)
        slider_frame.grid(row=row + 1, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        slider = customtkinter.CTkSlider(slider_frame, from_=from_, to=to, number_of_steps=steps)
        slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        value_frame = customtkinter.CTkFrame(tab)
        value_frame.grid(row=row + 1, column=1, padx=(0, 20), pady=(10, 10), sticky="ew")

        entry = customtkinter.CTkEntry(value_frame, width=50)
        entry.pack(side="left", padx=(0, 10))

        # Cria e posiciona o checkbox usando uma variável tkinter
        checkbox = customtkinter.CTkCheckBox(value_frame, text="Tá podendo", variable=self.checkbox_vars[param_index])
        checkbox.pack(side="left")

        slider_value = customtkinter.CTkLabel(tab, text=f'Value = {slider.get()}', font=customtkinter.CTkFont(size=15, weight="normal"))
        slider_value.grid(row=row + 2, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        def update_slider(value):
            slider_value.configure(text=f'Value = {value:.2e}')
            command(value)

        def slider_command(value):
            entry.delete(0, "end")
            entry.insert(0, f"{value:.2f}")
            update_slider(value)

        def entry_command(event):
            try:
                value = float(entry.get())
                slider.set(value)
                update_slider(value)
            except ValueError:
                pass

        slider.configure(command=slider_command)
        entry.bind("<Return>", entry_command)
        entry.insert(0, f"{slider.get():.2f}")

        return slider, slider_value


    def show_loading_screen(self):
        if self.loading_screen is not None and self.loading_screen.winfo_exists():
            return
        self.loading_screen = customtkinter.CTkToplevel(self)
        self.loading_screen.geometry("300x150")
        self.loading_screen.title("Carregando...")

        self.label = customtkinter.CTkLabel(self.loading_screen, text="Carregando...")
        self.label.pack(pady=10)

        self.gif_frames = self.load_gif()
        self.current_frame = 0
        self.animation_running = True
        self.update_gif()

    def load_gif(self):
        gif = Image.open(self.gif_path)
        frames = []
        try:
            while True:
                frames.append(ImageTk.PhotoImage(gif.copy().convert("RGBA")))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        return frames

    def update_gif(self):
        if self.animation_running and self.gif_frames:
            if self.loading_screen and self.loading_screen.winfo_exists():
                self.label.configure(image=self.gif_frames[self.current_frame])
                self.current_frame = (self.current_frame + 1) % len(self.gif_frames)
                self.loading_screen.after(50, self.update_gif)
            else:
                self.animation_running = False

    def hide_loading_screen(self):
        if self.loading_screen is not None and self.loading_screen.winfo_exists():
            try:
                self.loading_screen.after_cancel(self.update_gif)
                self.loading_screen.destroy()
            except Exception as e:
                print(f"Error hiding loading screen: {e}")
            self.loading_screen = None
        self.animation_running = False

    def optimize(self):
        threading.Thread(target=self.perform_optimization).start()

    def perform_optimization(self):
        self.show_loading_screen()
        variable_flags = [self.checkbox_vars[i].get() for i in range(len(self.params))]
        bounds = [
            (param, param) if not flag else (param * 0, param * 3)
            for param, flag in zip(self.params, variable_flags)
        ]

        before_params = self.params.copy()
        print("Antes da otimização:", before_params)
        result = differential_evolution(objective_function, bounds, args=(self.x, self.y), seed=42)
        optimized_params = result.x
        print("Depois da otimização:", optimized_params)
        self.params = optimized_params
        self.update_plot()

        message_lines = ["Resultados da Otimização:"]
        for i in range(int(self.params[0]) + 2):
            if i == 0:
                layer_name = "Parâmetros Gerais"
                indices = [1, 2, 3]
                param_names = ["Fator de Escala", "Densidade eletrônica do Ar  ", "absorção do Ar "]
            elif i == 1:
                layer_name = "Parâmetros do Substrato"
                indices = [4, 5, 6, 7]
                param_names = ["Densidade eletrônica ", "Absorção  ", "Background", "Rugosidade  "]
            else:
                layer_name = f"Parâmetros da Camada {i - 1}"
                indices = [4 * (i - 2) + 8, 4 * (i - 2) + 9, 4 * (i - 2) + 10, 4 * (i - 2) + 11]
                param_names = ["Espessura ", "Densidade eletrônica ", "Absorção  ", "Rugosidade "]

            message_lines.append(f"\n{layer_name}:")
            for idx, param_name in zip(indices, param_names):
                message_lines.append(f"{param_name}:")
                message_lines.append(f"  Antes: {before_params[idx]}")
                message_lines.append(f"  Depois: {optimized_params[idx]}")

        self.optimization_message = "\n".join(message_lines)
        self.hide_loading_screen()
        self.after(0, self.show_optimization_results)

    def show_optimization_results(self):
        message_window = customtkinter.CTkToplevel(self)
        message_window.title("Resultados da Otimização")
        message_window.geometry("400x600")

        sframe = customtkinter.CTkScrollableFrame(message_window, height=550, width=380)
        sframe.pack(fill="both", expand=True, padx=20, pady=20)

        message_lines = self.optimization_message.split("\n")
        for line in message_lines:
            label = customtkinter.CTkLabel(sframe, text=line)
            label.pack(pady=5, padx=5)

        message_window.mainloop()

    def auto_adjust(self):
        self.show_loading_screen()
        threading.Thread(target=self.perform_auto_adjust).start()

    def perform_auto_adjust(self):
        backing_params = [self.params[4], self.params[5], self.params[6], self.params[7]]
        layer_params = []
        num_layers = int(self.params[0])
        for i in range(num_layers):
            start_index = 8 + 4 * i
            layer_params.extend(self.params[start_index:start_index + 4])

        all_params = [self.params[0], self.params[1], self.params[2], self.params[3]] + backing_params + layer_params
        variable_flags = [False, False, False, False] + [True] * len(backing_params) + [True] * len(layer_params)
        bounds = [
            (param, param) if not flag else (param * 0, param * 1000)
            for param, flag in zip(all_params, variable_flags)
        ]

        before_params = all_params.copy()
        print("Primeira otimização:", before_params)
        result = differential_evolution(objective_function, bounds, args=(self.x, self.y), seed=42)
        optimized_params = result.x
        bounds2 = [
            (param, param) if i == 0 else (0, param * 2)
            for i, param in enumerate(optimized_params)
        ]
        print("Depois da otimização:", optimized_params)
        result2 = differential_evolution(objective_function, bounds2, args=(self.x, self.y), seed=42)
        optimized_params2 = result2.x
        self.params = optimized_params2

        self.update_plot()
        self.hide_loading_screen()
        
        message_lines = ["Resultados da Otimização:"]
        for i in range(int(self.params[0]) + 2):
            if i == 0:
                layer_name = "Parâmetros Gerais"
                indices = [1, 2, 3]
                param_names = ["Fator de Escala", "Densidade eletrônica do Ar  ", "absorção do Ar "]
            elif i == 1:
                layer_name = "Parâmetros do Substrato"
                indices = [4, 5, 6, 7]
                param_names = ["Densidade eletrônica ", "Absorção  ", "Background", "Rugosidade  "]
            else:
                layer_name = f"Parâmetros da Camada {i - 1}"
                indices = [4 * (i - 2) + 8, 4 * (i - 2) + 9, 4 * (i - 2) + 10, 4 * (i - 2) + 11]
                param_names = ["Espessura ", "Densidade eletrônica ", "Absorção  ", "Rugosidade "]

            message_lines.append(f"\n{layer_name}:")
            for idx, param_name in zip(indices, param_names):
                message_lines.append(f"{param_name}:")
                message_lines.append(f"  Antes: {before_params[idx]}")
                message_lines.append(f"  Depois: {optimized_params[idx]}")


        self.optimization_message = "\n".join(message_lines)
        self.show_optimization_results()

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def save_to_pdf(self):
        file_path = fd.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if not file_path:
            return

        buffer = io.BytesIO()
        self.fig.savefig(buffer, format='png')
        buffer.seek(0)

        c = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica", 10)
        text_lines = self.optimization_message.split("\n")
        y_position = height - 50
        for line in text_lines:
            c.drawString(50, y_position, line)
            y_position -= 12
            if y_position < 100:
                c.showPage()
                c.setFont("Helvetica", 10)
                y_position = height - 50

        self.ax.clear()
        scatter = self.ax.scatter(self.x, self.y, label='Experimental Data', s=0.7, color='darkgreen')
        y_pred = abeles_python(self.x, self.params)
        plt.plot(self.x, y_pred, label='Chute', color='navy')
        plt.xlabel('Q ($\\AA^{-1}$)')
        plt.ylabel('Refletividade')
        plt.title('Curva de Refletometria')
        plt.yscale('log')
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.savefig(file_path[:-4] + ".png", dpi=600)

        c.showPage()
        img = ImageReader(file_path[:-4] + ".png")
        c.drawImage(img, 50, height - 500, width=500, height=400)
        c.save()
        print(f"PDF saved to {file_path}")

