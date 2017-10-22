# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:33:26 2017

@author: Gergo
"""

#%%

from actor import Actor
from critic import Critic
from buffer import Buffer
import json
import time
import numpy as np
import tensorflow as tf
import math
import matplotlib
import matplotlib.pyplot  as plt
from environment import Environment
from car import Car
import pandas as pd
#import keras
from frontend import *

from keras import backend as K
from copy import deepcopy

def start_simulation(GUI):
    np.random.seed(123)
    LIDAR_RESOLUTION = 30
    LIDAR_MAX_RANGE = 80
    ACC_SCALE = 10.
    STEER_ANGLE_SCALE = math.pi/2
    
# =============================================================================
#     Hyper parameters
# =============================================================================
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Learning rate for Critic    
    
    action_dim = 2  #Steering/Acceleration
    if GUI.sensor_mode.get()=="LIDAR":
        state_dim = LIDAR_RESOLUTION+2
    else:
        state_dim = 4  
    
    sess = tf.Session()
    K.set_session(sess)

    actor = Actor(sess,GUI.sensor_mode.get(), state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = Critic(sess,GUI.sensor_mode.get(), state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = Buffer(BUFFER_SIZE)
    
    if GUI.load_nn.get():
        try:
            actor.model.load_weights("actormodel.h5")
            critic.model.load_weights("criticmodel.h5")
            actor.target_model.load_weights("actormodel.h5")
            critic.target_model.load_weights("criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")
    
    EXPLORE = 100000.
#    episode_count = 2000
#    max_steps = 100000
    
    epsilon = 1
    obstacles = []
    
    if GUI.obstacles:
        obstacles = [(0,10),(0,1)]
    max_simulation_laps = 100000
    env = Environment(GUI.track_path.get(),np.array([[252,30],[252,80]]),
                np.array([[251,30],[251,80]]),obstacles,np.array([255,255,255]),time_step=1)
    car = Car(GUI.width.get(),GUI.length.get(),GUI.gg_path.get(),env.start_position,env.start_speed,env.start_dir)

    theta = car.dir
    rot_matrix = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=float)
    # =============================================================================
#             Draw the track and car
# =============================================================================
    chassis = np.array([[0,car.width,car.width,0],[0,0,car.length,car.length]],dtype=float)
    chassis = rot_matrix.dot(chassis)
    chassis[0,:] = chassis[0,:] + car.pos[0] - car.width/2 
    chassis[1,:] = chassis[1,:] + car.pos[1] - car.length/2
    chassis= np.round(chassis)
    chassis = chassis.astype(int)
    GUI.track_figure_handle.clear()
    track = GUI.track_figure_handle.add_subplot(111)
    plt.imshow(GUI.track_img,aspect='auto')
    plt.plot([env.start_line[0,0],env.start_line[1,0]], [env.start_line[0,1],
             env.start_line[1,1]], color='g', linestyle='-', linewidth=1)
    plt.plot([chassis[0,0],chassis[0,1]], [chassis[1,0],chassis[1,1]],
             color='k', linestyle='-', linewidth=2)
    plt.plot([chassis[0,1],chassis[0,2]], [chassis[1,1],chassis[1,2]],
             color='k', linestyle='-', linewidth=2)
    plt.plot([chassis[0,2],chassis[0,3]], [chassis[1,2],chassis[1,3]],
             color='k', linestyle='-', linewidth=2)
    plt.plot([chassis[0,3],chassis[0,0]], [chassis[1,3],chassis[1,0]],
             color='k', linestyle='-', linewidth=2)
    if GUI.obstacles:
        for obstacle in env.obstacles:
            plt.scatter(obstacle[0], obstacle[1],1,"k")
    GUI.draw_track_callback()
    GUI.update_idletasks()
    GUI.update()
    
    
    
    for i in range(max_simulation_laps):
# =============================================================================
#         TODO Random5ly initialise (switchable) our starting 
# =============================================================================
        env.reset(False)
        car.reset(env.start_position,env.start_speed,env.start_dir)
        over = False
#        tmp=1
        cumulative_r = 0
        
        if GUI.sensor_mode.get()=="LIDAR":
            state = np.hstack([env.get_sensor_data(car.pos, car.dir, LIDAR_RESOLUTION , LIDAR_MAX_RANGE)*2/LIDAR_MAX_RANGE-1,
                               car.speed/40])
        else:
            state = np.hstack([(car.pos[0]*2/GUI.track_img.size[0])-1,
                                  (car.pos[1]*2/GUI.track_img.size[1])-1,
                                  car.speed/30,
                                  car.dir/(math.pi*2)])
        while not over:
# =============================================================================
#           TODO  We get the our action here (acc, steer_angle)
# =============================================================================
            epsilon -= 1.0 / EXPLORE
            a = np.zeros([1,action_dim])
            noise = np.zeros([1,action_dim])
            noise[0][0] = max(epsilon, 0) * np.random.normal(0,0.5)
            noise[0][1] = max(epsilon, 0) * np.random.normal(0,0.5)
            if GUI.sensor_mode.get()=="LIDAR":
                state_1 = np.atleast_2d(state[0:-1])
                state_1 = np.expand_dims(state_1,axis=2)
                state_2 = np.atleast_2d(state[-1])
            else:
                state_1 = np.atleast_2d(state[0:2])
                state_2 = np.atleast_2d(state[2:4])
            a_original = actor.model.predict([state_1,state_2])
            a = a_original[0] + noise[0]
            acc, steer_angle = a*np.array([ACC_SCALE,STEER_ANGLE_SCALE])
            #            acc = np.random.uniform(-5,5)
#            steer_angle = np.random.uniform(-math.pi/2,math.pi/2)
#            steer_angle = math.pi/(4*8)
            prev_pos = deepcopy(car.pos)
            prev_speed = deepcopy(car.speed)
            prev_dir = deepcopy(car.dir)
            env.step(car,acc,steer_angle)
            over,result = env.is_over(car,prev_pos)
            
# =============================================================================
#             TODO Store our data in the pandas database.(s,a,r,s')
#            (prev_pos,prev_dir,prev_speed),(acc,steer_angle),r,(car.pos,car.dir,car.speed)
#            OR local info?
# =============================================================================
            if GUI.sensor_mode.get()=="LIDAR":
                state = np.hstack([env.get_sensor_data(prev_pos, prev_dir, LIDAR_RESOLUTION , LIDAR_MAX_RANGE)*2/LIDAR_MAX_RANGE-1,
                                   prev_speed/40])
            else:
                state = np.hstack([(prev_pos[0]*2/GUI.track_img.size[0])-1,
                                  (prev_pos[1]*2/GUI.track_img.size[1])-1,
                                  prev_speed/30,
                                  prev_dir/(math.pi*2)])
# =============================================================================
#             Observing reward
# =============================================================================
            if over and result:
                r = 1
            elif over and not result:
                r = -1
            elif not over:
                r = float(env.get_reward(car.pos)/18)
                
            cumulative_r = cumulative_r + r
            if GUI.sensor_mode.get()=="LIDAR":
                next_state = np.hstack([env.get_sensor_data(car.pos, car.dir, LIDAR_RESOLUTION , LIDAR_MAX_RANGE)*2/LIDAR_MAX_RANGE-1,
                                        car.speed/40])
            else:
                next_state = np.hstack([(car.pos[0]*2/GUI.track_img.size[0])-1,
                                  (car.pos[1]*2/GUI.track_img.size[1])-1,
                                  car.speed/30,
                                  car.dir/(math.pi*2)])

            experience = [state.reshape(-1),a,r,next_state.reshape(-1),over]
            buff.add_item(experience)
            car_color = 'r' if over and not result else 'k'
# =============================================================================
#             Draw the track and car
# =============================================================================
            if GUI.close_flag:
                return
            if GUI.draw_track.get():
                theta = car.dir
                rot_matrix = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=float)
                chassis = np.array([[0,car.width,car.width,0],[0,0,car.length,car.length]])
                chassis[0,:] -= car.width/2
                chassis[1,:] -= car.length/2
                chassis = rot_matrix.dot(chassis)
                chassis[0,:] = chassis[0,:] + car.pos[0] 
                chassis[1,:] = chassis[1,:] + car.pos[1]
                chassis= np.round(chassis)
                chassis = chassis.astype(int)
                GUI.track_figure_handle.clear()
                track = GUI.track_figure_handle.add_subplot(111)
                plt.imshow(GUI.track_img,aspect='auto')
                plt.plot([env.start_line[0,0],env.start_line[1,0]], [env.start_line[0,1],
                          env.start_line[1,1]], color='g', linestyle='-', linewidth=1)
                plt.plot([chassis[0,0],chassis[0,1]], [chassis[1,0],chassis[1,1]],
                         color=car_color, linestyle='-', linewidth=2)
                plt.plot([chassis[0,1],chassis[0,2]], [chassis[1,1],chassis[1,2]],
                         color=car_color, linestyle='-', linewidth=2)
                plt.plot([chassis[0,2],chassis[0,3]], [chassis[1,2],chassis[1,3]],
                         color=car_color, linestyle='-', linewidth=2)
                plt.plot([chassis[0,3],chassis[0,0]], [chassis[1,3],chassis[1,0]],
                         color=car_color, linestyle='-', linewidth=2)
                if GUI.obstacles:
                    for obstacle in env.obstacles:
                        plt.scatter(obstacle[0], obstacle[1],1,"k")
                GUI.draw_track_callback()
            GUI.update_idletasks()
            GUI.update()
# =============================================================================
#   TODO We tran the network with the train data. (R only first then R+max(Q')).
#   Then at validation we stop when the error is the lowest
# =============================================================================
            batch = buff.get_batch(BATCH_SIZE)
            states = np.asarray(np.atleast_2d([e[0] for e in batch]))
            actions = np.asarray(np.atleast_2d([e[1] for e in batch]))
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray(np.atleast_2d([e[3] for e in batch]))
            overs = np.asarray([e[4] for e in batch])
            y = np.asarray([e[1] for e in batch])
            
            if GUI.sensor_mode.get()=="LIDAR":
                states_1 = states[:,0:-1]
                states_1 = np.expand_dims(states_1,axis=2)
                states_2 = states[:,-1]
                new_states_1 = new_states[:,0:-1]
                new_states_1 = np.expand_dims(new_states_1,axis=2)
                new_states_2 = new_states[:,-1]
            else:
                states_1 = states[:,0:2]
                states_2 = states[:,2:4]
                new_states_1 = new_states[:,0:2]
                new_states_2 = new_states[:,2:4]
            target_q_values = critic.target_model.predict([new_states_1,new_states_2,
                                                           actor.target_model.predict([new_states_1,new_states_2])])  
           
            for k in range(len(batch)):
                if overs[k]:
                    y[k] = rewards[k]
                else:
                    y[k] = rewards[k] + GAMMA*target_q_values[k]
                    
            #            TODO LIDAR
            critic.model.train_on_batch([states_1,states_2,actions], y)
            #            TODO LIDAR
            a_for_grad = actor.model.predict([states_1,states_2])
            grads = critic.gradients(states, a_for_grad)
            actor.train(states, grads)
            actor.target_train()
            critic.target_train()
            
            
        print("Episode "+str(i)+"\tCumulative reward:"+str(cumulative_r))
# =============================================================================
#   TODO With the trained network we test our algorythm
# =============================================================================
    if (GUI.save_nn.get()):
        print("Now we save model")
        
        actor.model.save_weights("actormodel.h5", overwrite=True)
        with open("actormodel.json", "w") as outfile:
            json.dump(actor.model.to_json(), outfile)
        
        critic.model.save_weights("criticmodel.h5", overwrite=True)
        with open("criticmodel.json", "w") as outfile:
            json.dump(critic.model.to_json(), outfile)




# =============================================================================
# Main program
# =============================================================================

gui = GUI(handler=start_simulation)
# =============================================================================
# while True:
#     gui.update_idletasks()
#     gui.update()
#     time.sleep(0.1)
# =============================================================================
gui.mainloop()





