# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:20:03 2017

@author: Gergo
"""

#%%

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

from tkinter import filedialog
import tkinter as tk
from tkinter import ttk


LARGE_FONT= ("Verdana", 12)


class GUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        self.iconbitmap("Race_car.ico")
        tk.Tk.wm_title(self, "Racing car simulator")
        self.geometry("800x800")
        self.grid_columnconfigure(0, weight=1)
        container = tk.Frame(self)
        container.grid(sticky = "nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, Settings, PageTwo, PageThree):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky=tk.N+tk.E+tk.S+tk.W)

        self.show_frame(Settings)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Settings",
                            command=lambda: controller.show_frame(Settings))
        button.pack()

        button2 = ttk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

        button3 = ttk.Button(self, text="Graph Page",
                            command=lambda: controller.show_frame(PageThree))
        button3.pack()


class Settings(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        main_container = tk.Frame(self,bg="brown")
        main_container.grid_columnconfigure(0, weight = 1)
        main_container.grid( sticky="nsew")
        self.grid_columnconfigure(0, weight = 1)
        
        top_container = tk.Frame(main_container,bg="pink")
        top_container.grid_columnconfigure(0, weight = 1)
        top_container.grid(sticky="nsew")
        frame_name = tk.Label(top_container, text="Settings", font=LARGE_FONT)
        frame_name.grid()
        
        settings_container = tk.Frame(main_container,bg = "red")
        settings_container.grid(sticky=tk.N+tk.E+tk.S+tk.W)
        left_container = tk.Frame(settings_container,bg = "blue")
        left_container.grid(row = 1,column = 0, sticky="nsew")
        mid_container = tk.Frame(settings_container,bg = "white")
        mid_container.grid(row = 1,column = 1, sticky="nsew")
        right_container = tk.Frame(settings_container,bg = "green")
        right_container.grid(row = 1,column = 2, sticky="nsew")
        
        footer_container = tk.Frame(main_container,bg="pink")
        footer_container.grid_columnconfigure(0, weight = 1)
        footer_container.grid(sticky = "nsew")
        
        left_header = tk.Label(left_container, text = "Car", font=LARGE_FONT)
        left_header.grid(columnspan = 2, sticky="nsew")
        mid_header = tk.Label(mid_container, text = "Track", font=LARGE_FONT)
        mid_header.grid(columnspan = 2, sticky="nsew")
        right_header = tk.Label(right_container, text = "NN", font=LARGE_FONT)
        right_header.grid(columnspan = 2, sticky="nsew")
        
        for col_num in range(settings_container.grid_size()[0]):
            settings_container.grid_columnconfigure(col_num, weight = 1)
# =============================================================================
#        Car settings 
# =============================================================================

        load_gg_label = tk.Label(left_container, text = "Default_GG.bmp")
        load_gg_label.grid(row = 1, column = 1, sticky = "w", padx=4)
        load_gg_btn = tk.Button(left_container, text =  "Select GG",
                                command = lambda: filedialog.askopenfilename())
        load_gg_btn.grid(row = 1, column = 0, sticky = "e", padx=4)
        length_label = tk.Label(left_container, text = "Length:")
        length_label.grid(row = 2, column = 0, sticky = "e", padx=4)
        length_entry = tk.Entry(left_container)
        length_entry.grid(row = 2, column = 1, sticky = "w", padx=4)
        width_label = tk.Label(left_container, text = "Width:")
        width_label.grid(row = 3, column = 0, sticky = "e", padx=4)
        width_entry = tk.Entry(left_container)
        width_entry.grid(row = 3, column = 1, sticky = "w", padx=4)
        map_the_track_label = tk.Label(left_container, text = "Map the track?")
        map_the_track_label.grid(row = 4, column = 0, sticky = "e", padx=4)
        map_the_track_check = tk.Checkbutton(left_container)
        map_the_track_check.grid(row = 4, column = 1, sticky = "w", padx=4)
        sensor_mode_label = tk.Label(left_container, text = "Sensor mode")
        sensor_mode_label.grid(row = 5 , column = 0, sticky = "e", padx=4)    
        sensor_mode = ttk.Combobox ( left_container, values = ("local_LIDAR","global_LIDAR"))
        sensor_mode.grid(row = 5, column = 1, sticky = "w", padx=4)
        
        for row_num in range(left_container.grid_size()[1]):
            left_container.grid_rowconfigure(row_num, minsize=30)
        
# =============================================================================
#       Track settings  
# =============================================================================
        load_track_label = tk.Label(mid_container, text = "Default_track.bmp")
        load_track_label.grid(row = 1, column = 1, sticky = "w", padx=4)
        load_track_btn = tk.Button(mid_container, text =  "Select track",
                                command = lambda: filedialog.askopenfilename())
        load_track_btn.grid(row = 1, column = 0, sticky = "e", padx = 4)
        time_step_lable = tk.Label (mid_container, text = "Timestep")
        time_step_lable.grid(row = 2, column = 0, stick = "e", padx = 4)
        time_step_entry = tk.Entry(mid_container)
        time_step_entry.grid(row = 2, column = 1, sticky = "w", padx = 4)
        obstacles_label = tk.Label(mid_container, text = "Obstacles?")
        obstacles_label.grid(row = 3, column = 0, sticky = "e", padx=4)
        obstacles_check = tk.Checkbutton(mid_container)
        obstacles_check.grid(row = 3, column = 1, sticky = "w", padx=4)
        obstacles_type_label = tk.Label(mid_container, text = "Static/Dinamic")
        obstacles_type_label.grid(row = 4 , column = 0, sticky = "e", padx=4)    
        obstacles_type = ttk.Combobox ( mid_container, values = ("Static","Dinamic"))
        obstacles_type.grid(row = 4, column = 1, sticky = "w", padx=4)
        draw_track_label = tk.Label(mid_container, text = "Draw track?")
        draw_track_label.grid(row = 5, column = 0, sticky = "e", padx=4)
        draw_track_check = tk.Checkbutton(mid_container)
        draw_track_check.grid(row = 5, column = 1, sticky = "w", padx=4)
# =============================================================================
#       NN Settings
# =============================================================================
        epochs_input = tk.Entry(right_container)
        epochs_input.grid(row = 1 ,  column = 1, sticky = "w", padx = 4)
        epochs_label = tk.Label (right_container, text = "Number of epochs:")
        epochs_label.grid(row = 1, column = 0, sticky = "e", padx = 4)
        
        load_nn_checkbox = tk.Checkbutton(right_container)
        load_nn_checkbox.grid(row = 2, column = 1, sticky = "w", padx = 4)
        load_nn_label = tk.Label(right_container,text = "Load NN model?")
        load_nn_label.grid(row = 2, column = 0, sticky = "e", padx = 4)
        
        load_nn_btn_label = tk.Label(right_container, text = "Default_NN.bmp")
        load_nn_btn_label.grid(row = 3, column = 1, sticky = "w", padx=4)
        load_nn_btn = tk.Button(right_container, text =  "Select NN",
                                command = lambda: filedialog.askopenfilename())
        load_nn_btn.grid(row = 3, column = 0, sticky = "e", padx = 4)  
        
        save_nn_checkbox = tk.Checkbutton(right_container)
        save_nn_checkbox.grid(row = 4, column = 1, sticky = "w", padx = 4)
        save_nn_label = tk.Label(right_container,text = "Save NN model?")
        save_nn_label.grid(row = 4, column = 0, sticky = "e", padx = 4)
       
# =============================================================================
#       Footer  
# =============================================================================
        go_btn = tk.Button(footer_container, text =  "GO!",
                                command = (lambda: print("GO!!!")))
        go_btn.grid(sticky = "nsew", pady = 4, padx = 4)

class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page Two!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page One",
                            command=lambda: controller.show_frame(PageOne))
        button2.pack()


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        

        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        

gui = GUI()
gui.mainloop()