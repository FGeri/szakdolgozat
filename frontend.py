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
        self.protocol("WM_DELETE_WINDOW", self.close_handler)
        self.start_simulation_handler=kwargs['handler']
        self.debug_active = tk.BooleanVar()
        self.debug_active.set(False)
# =============================================================================
#        Variable declarations
# =============================================================================
        self.close_flag = False
        self.test_flag = False
        self.progress = tk.IntVar()
        self.progress.set(0)
        self.progress_label = tk.StringVar()
        self.progress_label.set("0%")
# =============================================================================
#         Car
# =============================================================================
     
        self.length = tk.DoubleVar()
        self.length.set(8.0)
        
        self.width = tk.DoubleVar()
        self.width.set(4.0)
        
        self.sensor_mode = tk.StringVar()
        self.sensor_mode.set("LIDAR")

        self.lidar_res = tk.IntVar()
        self.lidar_res.set(20)
        
        self.lidar_range = tk.IntVar()
        self.lidar_range.set(120)
        
        self.max_acc = tk.DoubleVar()
        self.max_acc.set(8)
        
        self.acc_res = tk.IntVar()
        self.acc_res.set(4)
        
        self.max_steering = tk.IntVar()
        self.max_steering.set(30)
        
        self.steering_res = tk.IntVar()
        self.steering_res.set(8)

# =============================================================================
#         Track
# =============================================================================
        self.track_path = tk.StringVar()
        self.track_path.set(".\\Maps\\track_2.png")
        
        self.track_img = Image.open(self.track_path.get())
        
        self.time_step = tk.DoubleVar()
        self.time_step.set(0.25)
        
        self.draw_track = tk.BooleanVar()
        self.draw_track.set(False)
        
        self.obstacles = tk.BooleanVar()
        self.obstacles.set(False)

# =============================================================================
#         NN
# =============================================================================
        self.episodes = tk.IntVar()
        self.episodes.set(100000)
        
        self.memory_size = tk.IntVar()
        self.memory_size.set(500000)
        
        self.enable_training = tk.BooleanVar()
        self.enable_training.set(True)
        
        self.load_nn = tk.BooleanVar()
        self.load_nn.set(False)
        
        self.nn_path = tk.StringVar()
        self.nn_path.set("criticmodel_best.h5")
        
        self.batch_size = tk.IntVar()
        self.batch_size.set(256)
        
        self.learning_rate = tk.DoubleVar()
        self.learning_rate.set(0.00005)
        
        self.exploration_function = tk.StringVar()
        self.exploration_function.set("Softmax,proportional decay")
        
        self.exploration_decay = tk.DoubleVar()
        self.exploration_decay.set(800.0)
        
        self.gamma = tk.DoubleVar()
        self.gamma.set(0.95)
        
        self.algorithm = tk.StringVar()
        self.algorithm.set("Monte Carlo")
        
        self.save_nn = tk.BooleanVar()
        self.save_nn.set(True)
        
# =============================================================================
#        GUI settings       
# =============================================================================
        self.iconbitmap("Race_car.ico")
        tk.Tk.wm_title(self, "Racing car simulator")
        self.geometry("800x800")
        self.grid_columnconfigure(0, weight=1)
        container = tk.Frame(self)
        container.grid(sticky = "nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (SettingsPage, SimulatorPage):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky=tk.N+tk.E+tk.S+tk.W)

        self.show_frame(SettingsPage)
    
    def close_handler(self):
        self.close_flag = True
        print("App is about to close")
        self.after(200,self.close_app)
    def close_app(self):
        self.destroy()
        
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
        ax.set_aspect('auto')
        self.track_canvas_handle.draw()
    def enter_debug_mode(self):
        self.debug_active.set(True)
    
    def set_test_flag(self):
        self.test_flag = True

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
#        length
        length_label = ttk.Label(left_container, text = "Length(m):")
        length_label.grid(row = 1, column = 0, sticky = "e", padx=4)
        length_entry = ttk.Entry(left_container,
                                 textvariable = parent.master.length)
        length_entry.grid(row = 1, column = 1, sticky = "w", padx=4)
        
#        width
        width_label = ttk.Label(left_container, text = "Width(m):")
        width_label.grid(row = 2, column = 0, sticky = "e", padx=4)
        width_entry = ttk.Entry(left_container ,
                                textvariable = parent.master.width)
        width_entry.grid(row = 2, column = 1, sticky = "w", padx=4)
              
#        sensor_mode
        sensor_mode_label = ttk.Label(left_container, text = "Sensor mode")
        sensor_mode_label.grid(row = 3 , column = 0, sticky = "e", padx=4)    
        sensor_mode = ttk.Combobox ( left_container, values = ("LIDAR","GLOBAL"),
                                    textvariable = parent.master.sensor_mode)
        sensor_mode.grid(row = 3, column = 1, sticky = "w", padx=4)
        
#        lidar_res
        lidar_res_label = ttk.Label(left_container, text = "Liadr resolution:")
        lidar_res_label.grid(row = 4, column = 0, sticky = "e", padx=4)
        lidar_res_entry = ttk.Entry(left_container,
                                 textvariable = parent.master.lidar_res)
        lidar_res_entry.grid(row = 4, column = 1, sticky = "w", padx=4)
        
#        lidar_range
        lidar_range_label = ttk.Label(left_container, text = "Lidar range(m):")
        lidar_range_label.grid(row = 5, column = 0, sticky = "e", padx=4)
        lidar_range_entry = ttk.Entry(left_container,
                                 textvariable = parent.master.lidar_range)
        lidar_range_entry.grid(row = 5, column = 1, sticky = "w", padx=4)

#        max_acc
        max_acc_label = ttk.Label(left_container, text = "Max acceleration(m/s^2):")
        max_acc_label.grid(row = 6, column = 0, sticky = "e", padx=4)
        max_acc_entry = ttk.Entry(left_container,
                                 textvariable = parent.master.max_acc)
        max_acc_entry.grid(row = 6, column = 1, sticky = "w", padx=4)

#        acc_res
        acc_res_label = ttk.Label(left_container, text = "Acceleration resolution:")
        acc_res_label.grid(row = 7, column = 0, sticky = "e", padx=4)
        acc_res_entry = ttk.Entry(left_container,
                                 textvariable = parent.master.acc_res)
        acc_res_entry.grid(row = 7, column = 1, sticky = "w", padx=4)

#        max_steering
        max_steering_label = ttk.Label(left_container, text = "Max steering(°):")
        max_steering_label.grid(row = 8, column = 0, sticky = "e", padx=4)
        max_steering_entry = ttk.Entry(left_container,
                                 textvariable = parent.master.max_steering)
        max_steering_entry.grid(row = 8, column = 1, sticky = "w", padx=4)

#        steering_res
        steering_res_label = ttk.Label(left_container, text = "Steering resolution:")
        steering_res_label.grid(row = 9, column = 0, sticky = "e", padx=4)
        steering_res_entry = ttk.Entry(left_container,
                                 textvariable = parent.master.steering_res)
        steering_res_entry.grid(row = 9, column = 1, sticky = "w", padx=4)        
        
        
        for row_num in range(left_container.grid_size()[1]):
            left_container.grid_rowconfigure(row_num, minsize=30)
        
# =============================================================================
#       Track settings  
# =============================================================================
#        load_track
        load_track_label = ttk.Label(mid_container, textvariable = parent.master.track_path)
        load_track_label.grid(row = 1, column = 1, sticky = "w", padx=4)
        load_track_btn = ttk.Button(mid_container, text =  "Select track",
                                command = lambda: self.load_track(parent.master))
        load_track_btn.grid(row = 1, column = 0, sticky = "e", padx = 4)
        
#        time_step
        time_step_lable = ttk.Label (mid_container, text = "Timestep(s)")
        time_step_lable.grid(row = 2, column = 0, stick = "e", padx = 4)
        time_step_entry = ttk.Entry(mid_container, textvariable = parent.master.time_step)
        time_step_entry.grid(row = 2, column = 1, sticky = "w", padx = 4)
        
#        obstacles
        obstacles_label = ttk.Label(mid_container, text = "Obstacles?")
        obstacles_label.grid(row = 3, column = 0, sticky = "e", padx=4)
        obstacles_check = ttk.Checkbutton(mid_container, variable = parent.master.obstacles)
        obstacles_check.grid(row = 3, column = 1, sticky = "w", padx=4)
        
#        draw_track
        draw_track_label = ttk.Label(mid_container, text = "Draw track?")
        draw_track_label.grid(row = 4, column = 0, sticky = "e", padx=4)
        draw_track_check = ttk.Checkbutton(mid_container, variable = parent.master.draw_track)
        draw_track_check.grid(row = 4, column = 1, sticky = "w", padx=4)
        
        for row_num in range(mid_container.grid_size()[1]):
            mid_container.grid_rowconfigure(row_num, minsize=30)
            
        self.track_img = Image.open(controller.track_path.get())
        self.track_preview = plt.figure()
        ax=self.track_preview.add_axes([0,0,1,1])
        ax.set_axis_off()
        img = ax.imshow(self.track_img,aspect='auto')
        DPI = self.track_preview.get_dpi()
        self.track_preview.set_size_inches((self.track_img.size[0]/2/float(DPI),self.track_img.size[1]/2/float(DPI)))
        ax.set_aspect('auto')
        self.canvas = FigureCanvasTkAgg(self.track_preview, mid_container) 
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row = 5, column = 0,columnspan=2, sticky = "w", padx=4)
        
# =============================================================================
#       NN Settings
# =============================================================================
        
#        episodes
        episodes_input = ttk.Entry(right_container, textvariable = parent.master.episodes )
        episodes_input.grid(row = 1 ,  column = 1, sticky = "w", padx = 4)
        episodes_label = ttk.Label (right_container, text = "Episodes:")
        episodes_label.grid(row = 1, column = 0, sticky = "e", padx = 4)
        
#        load_nn
        load_nn_checkbox = ttk.Checkbutton(right_container, variable = parent.master.load_nn)
        load_nn_checkbox.grid(row = 2, column = 1, sticky = "w", padx = 4)
        load_nn_label = ttk.Label(right_container,text = "Load NN model?")
        load_nn_label.grid(row = 2, column = 0, sticky = "e", padx = 4)
        
        
#        load_nn_btn
        load_nn_btn_label = ttk.Label(right_container, textvariable = parent.master.nn_path)
        load_nn_btn_label.grid(row = 3, column = 1, sticky = "w", padx=4)
        load_nn_btn = ttk.Button(right_container, text =  "Select NN",
                                command = lambda:  self.load_nn(parent.master))
        load_nn_btn.grid(row = 3, column = 0, sticky = "e", padx = 4)  
        
#        enable_training
        enable_training_checkbox = ttk.Checkbutton(right_container, variable = parent.master.enable_training)
        enable_training_checkbox.grid(row = 4, column = 1, sticky = "w", padx = 4)
        enable_training_label = ttk.Label(right_container,text = "Train model:")
        enable_training_label.grid(row = 4, column = 0, sticky = "e", padx = 4)
        
#        learning_rate
        learning_rate_input = ttk.Entry(right_container, textvariable = parent.master.learning_rate )
        learning_rate_input.grid(row = 5 ,  column = 1, sticky = "w", padx = 4)
        learning_rate_label = ttk.Label (right_container, text = "Learning rate:")
        learning_rate_label.grid(row = 5, column = 0, sticky = "e", padx = 4)
        
#        memory_size
        memory_size_input = ttk.Entry(right_container, textvariable = parent.master.memory_size )
        memory_size_input.grid(row = 6 ,  column = 1, sticky = "w", padx = 4)
        memory_size_label = ttk.Label (right_container, text = "Memory size:")
        memory_size_label.grid(row = 6, column = 0, sticky = "e", padx = 4)
        
#        batch_size
        batch_size_input = ttk.Entry(right_container, textvariable = parent.master.batch_size )
        batch_size_input.grid(row = 7 ,  column = 1, sticky = "w", padx = 4)
        batch_size_label = ttk.Label (right_container, text = "Batch size:")
        batch_size_label.grid(row = 7, column = 0, sticky = "e", padx = 4)
        
#        gamma
        gamma_input = ttk.Entry(right_container, textvariable = parent.master.gamma )
        gamma_input.grid(row = 8 ,  column = 1, sticky = "w", padx = 4)
        gamma_label = ttk.Label (right_container, text = "Gamma:")
        gamma_label.grid(row = 8, column = 0, sticky = "e", padx = 4)
        
#        algorithm
        algorithm_label = ttk.Label(right_container, text = "RL Algorithm")
        algorithm_label.grid(row = 9 , column = 0, sticky = "e", padx=4)    
        algorithm = ttk.Combobox ( right_container, values = ("DQN","DDQN","Monte Carlo"),
                                    textvariable = parent.master.algorithm)
        algorithm.grid(row = 9, column = 1, sticky = "w", padx=4)
        
#        exploration_function
        exploration_function_label = ttk.Label(right_container, text = "Exploration func")
        exploration_function_label.grid(row = 10 , column = 0, sticky = "e", padx=4)    
        exploration_function = ttk.Combobox ( right_container, values = ("E-Greedy","Softmax,proportional decay"),
                                    textvariable = parent.master.exploration_function)
        exploration_function.grid(row = 10, column = 1, sticky = "w", padx=4)
        
#        exploration_decay
        exploration_decay_input = ttk.Entry(right_container, textvariable = parent.master.exploration_decay )
        exploration_decay_input.grid(row = 11 ,  column = 1, sticky = "w", padx = 4)
        exploration_decay_label = ttk.Label (right_container, text = "Exploration decay:")
        exploration_decay_label.grid(row = 11, column = 0, sticky = "e", padx = 4)
        
#        save_nn
        save_nn_checkbox = ttk.Checkbutton(right_container, variable = parent.master.save_nn)
        save_nn_checkbox.grid(row = 12, column = 1, sticky = "w", padx = 4)
        save_nn_label = ttk.Label(right_container,text = "Save NN model?")
        save_nn_label.grid(row = 12, column = 0, sticky = "e", padx = 4)
        
        for row_num in range(right_container.grid_size()[1]):
            right_container.grid_rowconfigure(row_num, minsize=30)
       
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
            parent.track_img = Image.open(parent.track_path.get())
            self.track_preview.clear()
            ax=self.track_preview.add_axes([0,0,1,1])
            ax.set_axis_off()
            img = ax.imshow(parent.track_img,aspect='auto')
            DPI = self.track_preview.get_dpi()
            self.track_preview.set_size_inches((parent.track_img.size[0]/2/float(DPI),parent.track_img.size[1]/2/float(DPI)))
            ax.set_aspect('auto')
            self.canvas.draw()
            
            
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
        track_figure = plt.figure()
        
        DPI = track_figure.get_dpi()
        track_figure.set_size_inches((controller.track_img.size[0]/float(DPI),controller.track_img.size[1]/float(DPI)))
        ax=track_figure.add_axes([0,0,1,1])
        ax.set_axis_off()        
        ax.imshow(controller.track_img,aspect='auto')
        ax.set_aspect('auto')
        controller.track_figure_handle = track_figure
        canvas = FigureCanvasTkAgg(track_figure, left_container) 
        
        controller.track_canvas_handle = canvas
        canvas.show()
        canvas.get_tk_widget().grid()
        canvas._tkcanvas.grid(sticky = "nw", pady = 4, padx = 4)
        
# =============================================================================
#       Simulator control
# =============================================================================

        btn1=ttk.Button(right_container, text="Settings",
                            command=lambda: controller.show_frame(SettingsPage))
        btn1.grid(row = 1, column = 0, sticky = "e")
        btn2=ttk.Button(right_container, text="DEBUG",
                            command=lambda: controller.enter_debug_mode())
        btn2.grid(row = 2, column = 0, sticky = "e")
        btn3=ttk.Button(right_container, text="Test",
                            command=lambda: controller.set_test_flag())
        btn3.grid(row = 3, column = 0, sticky = "e")
        

        draw_track_label = ttk.Label(right_container, text = "Draw track?")
        draw_track_label.grid(row = 4, column = 0, sticky = "e", padx=4)
        draw_track_check = ttk.Checkbutton(right_container, variable = parent.master.draw_track)
        draw_track_check.grid(row = 4, column = 1, sticky = "w", padx=4)

#       Progressbar
        
        progressbar = ttk.Progressbar(right_container, mode="determinate",
                                      variable = parent.master.progress, maximum = 100)
        
        progress_label = ttk.Label(right_container, text = "Progress")
        progress_label.grid(row = 9, column = 0, sticky = "e", padx=4)
        
        progressbar.grid(row = 10, column = 0, sticky = "e", padx=4)
        progressbar.step()
        progress_numeric_label = ttk.Label(right_container, textvariable = parent.master.progress_label)
        progress_numeric_label.grid(row = 10, column = 1, sticky = "w", padx=4)
        



        
