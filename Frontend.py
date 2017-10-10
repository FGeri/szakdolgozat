# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:20:03 2017

@author: Gergo
"""

#%%
import matplotlib
import matplotlib.pyplot  as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

matplotlib.use("TkAgg")
LARGE_FONT= ("Verdana", 12)


class GUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tmp_kwargs={}
        for k,v in kwargs.items():
            if not k =='handler':
                tmp_kwargs[k]=v
        tk.Tk.__init__(self, *args, **tmp_kwargs)
        self.start_simulation_handler=kwargs['handler']
# =============================================================================
#        Variable declarations
# =============================================================================
# =============================================================================
#         Car
# =============================================================================
        self.gg_path = tk.StringVar()
        self.gg_path.set("Default_GG.bmp")
     
        self.length = tk.DoubleVar()
        self.length.set(12.0)
        
        self.width = tk.DoubleVar()
        self.width.set(6.0)
        
        self.map_track = tk.IntVar()
        self.map_track.set(0)
        
        self.sensor_mode = tk.StringVar()
        self.sensor_mode.set("local_LIDAR")
# =============================================================================
#         Track
# =============================================================================
        self.track_path = tk.StringVar()
        self.track_path.set("track_tmp.png")
        
        self.time_step = tk.DoubleVar()
        self.time_step.set(1.0)
        
        self.draw_track = tk.IntVar()
        self.draw_track.set(0)
        
        self.obstacles = tk.IntVar()
        self.obstacles.set(0)
        
        self.obstacles_type = tk.StringVar()
        self.obstacles_type.set("Static")
        
# =============================================================================
#         NN
# =============================================================================
        self.epochs = tk.IntVar()
        self.epochs.set("10000")
        self.load_nn = tk.IntVar()
        self.load_nn.set("0")
        
        self.nn_path = tk.StringVar()
        self.nn_path.set("Default_NN.bmp")
        
        self.save_nn = tk.IntVar()
        self.save_nn.set("0")
        
        self.iconbitmap("Race_car.ico")
        tk.Tk.wm_title(self, "Racing car simulator")
        self.geometry("800x800")
        self.grid_columnconfigure(0, weight=1)
        container = tk.Frame(self)
        container.grid(sticky = "nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, SettingsPage, SimulatorPage, PageThree):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky=tk.N+tk.E+tk.S+tk.W)

        self.show_frame(SettingsPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()
        
    def start_simulation(self):
        self.show_frame(SimulatorPage)
        self.start_simulation_handler(self)
    
    def draw_track_callback(self):
#        self
        DPI = self.track_figure_handle.get_dpi()
        self.track_figure_handle.set_size_inches((self.track_img.size[0]/float(DPI),self.track_img.size[1]/float(DPI)))
        ax = self.track_figure_handle.gca()
        ax.set_axis_off()
        self.track_canvas_handle.draw()
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Settings",
                            command=lambda: controller.show_frame(SettingsPage))
        button.pack()

        button2 = ttk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(SimulatorPage))
        button2.pack()

        button3 = ttk.Button(self, text="Graph Page",
                            command=lambda: controller.show_frame(PageThree))
        button3.pack()


class SettingsPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        main_container = tk.Frame(self)
        main_container.grid_columnconfigure(0, weight = 1)
        main_container.grid( sticky="nsew")
        self.grid_columnconfigure(0, weight = 1)
        
        top_container = tk.Frame(main_container)
        top_container.grid_columnconfigure(0, weight = 1)
        top_container.grid(sticky="nsew")
        frame_name = tk.Label(top_container, text="Settings", font=LARGE_FONT)
        frame_name.grid()
        
        settings_container = tk.Frame(main_container)
        settings_container.grid(sticky=tk.N+tk.E+tk.S+tk.W)
        left_container = tk.Frame(settings_container)
        left_container.grid(row = 1,column = 0, sticky="nsew")
        mid_container = tk.Frame(settings_container)
        mid_container.grid(row = 1,column = 1, sticky="nsew")
        right_container = tk.Frame(settings_container)
        right_container.grid(row = 1,column = 2, sticky="nsew")
        
        footer_container = tk.Frame(main_container)
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


        load_gg_label = ttk.Label(left_container, textvariable = parent.master.gg_path)
        load_gg_label.grid(row = 1, column = 1, sticky = "w", padx=4)
        load_gg_btn = ttk.Button(left_container, text =  "Select GG",
                                command = lambda: self.load_gg(parent.master))
        load_gg_btn.grid(row = 1, column = 0, sticky = "e", padx=4)
        

        length_label = ttk.Label(left_container, text = "Length:")
        length_label.grid(row = 2, column = 0, sticky = "e", padx=4)
        length_entry = ttk.Entry(left_container,
                                 textvariable = parent.master.length)
        length_entry.grid(row = 2, column = 1, sticky = "w", padx=4)


        width_label = ttk.Label(left_container, text = "Width:")
        width_label.grid(row = 3, column = 0, sticky = "e", padx=4)
        width_entry = ttk.Entry(left_container ,
                                textvariable = parent.master.width)
        width_entry.grid(row = 3, column = 1, sticky = "w", padx=4)
        
        
        map_the_track_label = ttk.Label(left_container, text = "Map the track?")
        map_the_track_label.grid(row = 4, column = 0, sticky = "e", padx=4)
        map_the_track_check = ttk.Checkbutton(left_container, variable = parent.master.map_track)
        map_the_track_check.grid(row = 4, column = 1, sticky = "w", padx=4)
        
        
        sensor_mode_label = ttk.Label(left_container, text = "Sensor mode")
        sensor_mode_label.grid(row = 5 , column = 0, sticky = "e", padx=4)    
        sensor_mode = ttk.Combobox ( left_container, values = ("local_LIDAR","global_LIDAR"),
                                    textvariable = parent.master.sensor_mode)
        sensor_mode.grid(row = 5, column = 1, sticky = "w", padx=4)
        
        for row_num in range(left_container.grid_size()[1]):
            left_container.grid_rowconfigure(row_num, minsize=30)
        
# =============================================================================
#       Track settings  
# =============================================================================
        
        load_track_label = ttk.Label(mid_container, textvariable = parent.master.track_path)
        load_track_label.grid(row = 1, column = 1, sticky = "w", padx=4)
        load_track_btn = ttk.Button(mid_container, text =  "Select track",
                                command = lambda: self.load_track(parent.master))
        load_track_btn.grid(row = 1, column = 0, sticky = "e", padx = 4)
        
        
        time_step_lable = ttk.Label (mid_container, text = "Timestep")
        time_step_lable.grid(row = 2, column = 0, stick = "e", padx = 4)
        time_step_entry = ttk.Entry(mid_container, textvariable = parent.master.time_step)
        time_step_entry.grid(row = 2, column = 1, sticky = "w", padx = 4)
        
        
        obstacles_label = ttk.Label(mid_container, text = "Obstacles?")
        obstacles_label.grid(row = 3, column = 0, sticky = "e", padx=4)
        obstacles_check = ttk.Checkbutton(mid_container, variable = parent.master.obstacles)
        obstacles_check.grid(row = 3, column = 1, sticky = "w", padx=4)
        
        
        obstacles_type_label = ttk.Label(mid_container, text = "Static/Dinamic")
        obstacles_type_label.grid(row = 4 , column = 0, sticky = "e", padx=4)    
        obstacles_type = ttk.Combobox ( mid_container, values = ("Static","Dinamic"),
                                       textvariable = parent.master.obstacles_type)
        obstacles_type.grid(row = 4, column = 1, sticky = "w", padx=4)
        

        draw_track_label = ttk.Label(mid_container, text = "Draw track?")
        draw_track_label.grid(row = 5, column = 0, sticky = "e", padx=4)
        draw_track_check = ttk.Checkbutton(mid_container, variable = parent.master.draw_track)
        draw_track_check.grid(row = 5, column = 1, sticky = "w", padx=4)
# =============================================================================
#       NN Settings
# =============================================================================
        
        
        epochs_input = ttk.Entry(right_container, textvariable = parent.master.epochs )
        epochs_input.grid(row = 1 ,  column = 1, sticky = "w", padx = 4)
        epochs_label = ttk.Label (right_container, text = "Number of epochs:")
        epochs_label.grid(row = 1, column = 0, sticky = "e", padx = 4)
        
        
        load_nn_checkbox = ttk.Checkbutton(right_container, variable = parent.master.load_nn)
        load_nn_checkbox.grid(row = 2, column = 1, sticky = "w", padx = 4)
        load_nn_label = ttk.Label(right_container,text = "Load NN model?")
        load_nn_label.grid(row = 2, column = 0, sticky = "e", padx = 4)
        
        
        
        load_nn_btn_label = ttk.Label(right_container, textvariable = parent.master.nn_path)
        load_nn_btn_label.grid(row = 3, column = 1, sticky = "w", padx=4)
        load_nn_btn = ttk.Button(right_container, text =  "Select NN",
                                command = lambda:  self.load_nn(parent.master))
        load_nn_btn.grid(row = 3, column = 0, sticky = "e", padx = 4)  
        
        
        save_nn_checkbox = ttk.Checkbutton(right_container, variable = parent.master.save_nn)
        save_nn_checkbox.grid(row = 4, column = 1, sticky = "w", padx = 4)
        save_nn_label = ttk.Label(right_container,text = "Save NN model?")
        save_nn_label.grid(row = 4, column = 0, sticky = "e", padx = 4)
       
# =============================================================================
#       Footer  
# =============================================================================
        go_btn = ttk.Button(footer_container, text =  "GO!",
                            
                                command=lambda: controller.start_simulation())
        go_btn.grid(sticky = "nsew", pady = 4, padx = 4)
  
    
# =============================================================================
#   Handler functions
# =============================================================================
    def load_gg(self,parent):
        path = filedialog.askopenfilename()
        if path:
            parent.gg_path.set( path )
    def load_track(self,parent):
        path = filedialog.askopenfilename()
        if path:
            parent.track_path.set( path )
    def load_nn(self,parent):
        path = filedialog.askopenfilename()
        if path:
            parent.nn_path.set( path )

        
        
class SimulatorPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        main_container = tk.Frame(self)
        main_container.grid_columnconfigure(0, weight = 3)
        main_container.grid( sticky="nsew")
        self.grid_columnconfigure(0, weight = 1)
        
        top_container = tk.Frame(main_container)
        top_container.grid_columnconfigure(0, weight = 1)
        top_container.grid(sticky="nsew")
        frame_name = tk.Label(top_container, text="Settings", font=LARGE_FONT)
        frame_name.grid()
        
        settings_container = tk.Frame(main_container)
        settings_container.grid(sticky=tk.N+tk.E+tk.S+tk.W)
        left_container = ttk.Frame(settings_container)
        left_container.grid(row = 1,column = 0, sticky="nsew")
        right_container = ttk.Frame(settings_container)
        right_container.grid(row = 1,column = 1, sticky="nsew")
        
        left_header = tk.Label(left_container, text = "Track", font=LARGE_FONT)
        left_header.grid(columnspan = 2, sticky="nsew")
        right_header = tk.Label(right_container, text = "Controls", font=LARGE_FONT)
        right_header.grid(columnspan = 2, sticky="nsew")
        for col_num in range(settings_container.grid_size()[0]):
            settings_container.grid_columnconfigure(col_num, weight = 1)

# =============================================================================
#       Track and progress bar
# =============================================================================

#       Track
            
#       TODO REMOVE NEXT LINE

        path = "track_tmp.png"
        controller.track_img = Image.open(path)
        track = plt.figure()
        
        DPI = track.get_dpi()
        track.set_size_inches((controller.track_img.size[0]/float(DPI),controller.track_img.size[1]/float(DPI)))
        plt.imshow(controller.track_img,aspect='auto')
        ax = track.gca()
        ax.set_axis_off()
        controller.track_figure_handle = track
        canvas = FigureCanvasTkAgg(track, left_container) 
        
        controller.track_canvas_handle = canvas
        canvas.show()
        canvas.get_tk_widget().grid()

#        toolbar = NavigationToolbar2TkAgg(canvas, left_container)
#        toolbar.update()
        canvas._tkcanvas.grid()

#       Progressbar
        progress = tk.IntVar() 
        progressbar = ttk.Progressbar(left_container, mode="determinate",
                                      variable = progress, maximum = 100)
        
        progressbar.grid()
        progressbar.step()
        progress.set(50)
        
# =============================================================================
#       Simulator control
# =============================================================================

        btn1=ttk.Button(right_container, text="Settings",
                            command=lambda: controller.show_frame(SettingsPage))
        btn1.grid(row = 1)
        btn2=ttk.Button(right_container, text="Stop",
                            command=lambda: print ("Stop"))
        btn2.grid(row = 2)
        btn3=ttk.Button(right_container, text="Start",
                            command=lambda: print ("Start"))
        btn3.grid(row = 3)
        

        draw_track_label = ttk.Label(right_container, text = "Draw track?")
        draw_track_label.grid(row = 4, column = 0, sticky = "e", padx=4)
        draw_track_check = ttk.Checkbutton(right_container, variable = parent.master.draw_track)
        draw_track_check.grid(row = 4, column = 1, sticky = "w", padx=4)
        
        

class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()



        
