# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:33:26 2017

@author: Gergo
"""

from OU import OU
import hashlib
import time
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
from frontend import *

from keras import backend as K
from copy import deepcopy

def softmax(a):
    x = np.asarray(a,dtype=float)
    x -= np.max(x)
    exps = np.exp(x)
    return  exps/exps.sum()

#@profile
def start_simulation(GUI):
    np.random.seed(123)
# =============================================================================
#   Car params  
# =============================================================================
    LIDAR_RESOLUTION = GUI.lidar_res.get()
    LIDAR_MAX_RANGE = GUI.lidar_range.get()
    
    ACC_SCALE = GUI.max_acc.get()
    ACC_RESOLUTION = GUI.acc_res.get()
    
    STEER_ANGLE_SCALE = math.radians(GUI.max_steering.get())
    STEER_ANGLE_RESOLUTION = GUI.steering_res.get()
# =============================================================================
#     Track params
# =============================================================================
    MAX_SIMULATION_LAPS = GUI.episodes.get()
    MAX_STEPS_PER_LAP = 200
    obstacles = []
    if GUI.obstacles:
        obstacles = [(0,10),(0,1)]
# =============================================================================
#   Hyper parameters (NN)
# =============================================================================
    BUFFER_SIZE = GUI.memory_size.get()
    BATCH_SIZE = GUI.batch_size.get()
    GAMMA = GUI.gamma.get()
    TAU = 0.002     #Target Network HyperParameters
    TARGET_UPDATE_FREQ = 10
    TARGET_UPDATE_FREQ_EXP = 1.2
    LRA = 0.001    #Learning rate for Actor
    LRC = GUI.learning_rate.get()    
    EXPLORE = GUI.exploration_decay.get() 
    
    action_dim = 2  #Steering/Acceleration
    if GUI.sensor_mode.get()=="LIDAR":
        state_dim = LIDAR_RESOLUTION+3
    else:
        state_dim = 4  
    
    sess = tf.Session()
    K.set_session(sess)

#    actor = Actor(sess,GUI.sensor_mode.get(), state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    sampled_actions=np.array([[(-ACC_SCALE + 2*ACC_SCALE/ACC_RESOLUTION*x)/ACC_SCALE,
                               (-STEER_ANGLE_SCALE + 2*STEER_ANGLE_SCALE/STEER_ANGLE_RESOLUTION*y)/STEER_ANGLE_SCALE] \
                              for x in range(ACC_RESOLUTION+1) \
                              for y in range(STEER_ANGLE_RESOLUTION+1)])
    critic = Critic(sess,GUI.sensor_mode.get(), state_dim, action_dim, BATCH_SIZE, TAU, LRC,(ACC_RESOLUTION+1)*(STEER_ANGLE_RESOLUTION+1))
    critic.target_train()
    buff = Buffer(BUFFER_SIZE)
    if GUI.load_nn.get():
        try:
#            actor.model.load_weights("actormodel.h5")
            critic.model.load_weights(GUI.nn_path.get())
#            actor.target_model.load_weights("actormodel.h5")
#            critic.target_model.load_weights("criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")
    
    env = Environment(GUI.track_path.get(),np.array([[252,23],[252,82]]),
                np.array([[251,30],[251,80]]),obstacles,np.array([255,255,255]),time_step=1)
    car = Car(GUI.width.get(),GUI.length.get(),env.start_position,env.start_speed,env.start_dir)
    ou = OU()
    theta = car.dir
    rot_matrix = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=float)
# =============================================================================
#   Draw the track and car
# =============================================================================
    car_color = 'k'
    chassis = np.array([[0,car.width,car.width,0],[0,0,car.length,car.length]])
    chassis[0,:] -= car.width/2
    chassis[1,:] -= car.length/2
    chassis = rot_matrix.dot(chassis)
    chassis[0,:] = chassis[0,:] + car.pos[0] 
    chassis[1,:] = chassis[1,:] + car.pos[1]
    chassis= np.round(chassis)
    chassis = chassis.astype(int)
    GUI.track_figure_handle.clear()
    track_view = GUI.track_figure_handle.add_axes([0,0,1,1])
    track_view.imshow(GUI.track_img,aspect='auto')
    track_view.plot([env.start_line[0,0],env.start_line[1,0]], [env.start_line[0,1],
              env.start_line[1,1]], color='g', linestyle='-', linewidth=1)
    track_view.plot([chassis[0,0],chassis[0,1]], [chassis[1,0],chassis[1,1]],
              color=car_color, linestyle='-', linewidth=2)
    track_view.plot([chassis[0,1],chassis[0,2]], [chassis[1,1],chassis[1,2]],
              color=car_color, linestyle='-', linewidth=2)
    track_view.plot([chassis[0,2],chassis[0,3]], [chassis[1,2],chassis[1,3]],
              color=car_color, linestyle='-', linewidth=2)
    track_view.plot([chassis[0,3],chassis[0,0]], [chassis[1,3],chassis[1,0]],
              color=car_color, linestyle='-', linewidth=2)
    if GUI.obstacles:
              for obstacle in env.obstacles:
                  plt.scatter(obstacle[0], obstacle[1],1,"k")
    GUI.draw_track_callback()
    GUI.update_idletasks()
    GUI.update()
    
    log = [-20]
    exploration = np.ones(MAX_STEPS_PER_LAP)*3
    start_time = time.time()
    total_steps = 0
    best_reward = -100
    best_test_index = 0
    for i in range(MAX_SIMULATION_LAPS):

        env.reset(False)
        car.reset(env.start_position,env.start_speed,env.start_dir)
        over = False
        cumulative_r = 0
        experience_batch = pd.DataFrame([[np.array([]),np.array([]),0.,np.array([]),False,"",0.]],columns=['s', 'a', 'r', "s'", 'over','id','p'])
        experience_batch = experience_batch.drop(experience_batch.index[0])
        
        if (int(i/100) > 0 and i % 100 == 0) or GUI.test_flag:
            train = False
            GUI.test_flag = False
        else:
            train = GUI.enable_training.get()
# =============================================================================
#         Draw the track
# =============================================================================
        GUI.progress.set(int(i/MAX_SIMULATION_LAPS*100))
        GUI.progress_label.set(str(int(i/MAX_SIMULATION_LAPS*100))+"%")
        if GUI.draw_track.get():
            car_color = 'k'
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
            track_view = GUI.track_figure_handle.add_axes([0,0,1,1])
            track_view.imshow(GUI.track_img,aspect='auto')
            track_view.plot([env.start_line[0,0],env.start_line[1,0]], [env.start_line[0,1],
                      env.start_line[1,1]], color='g', linestyle='-', linewidth=1)
            track_view.plot([chassis[0,0],chassis[0,1]], [chassis[1,0],chassis[1,1]],
                     color=car_color, linestyle='-', linewidth=2)
            track_view.plot([chassis[0,1],chassis[0,2]], [chassis[1,1],chassis[1,2]],
                     color=car_color, linestyle='-', linewidth=2)
            track_view.plot([chassis[0,2],chassis[0,3]], [chassis[1,2],chassis[1,3]],
                     color=car_color, linestyle='-', linewidth=2)
            track_view.plot([chassis[0,3],chassis[0,0]], [chassis[1,3],chassis[1,0]],
                     color=car_color, linestyle='-', linewidth=2)
            if GUI.obstacles:
                for obstacle in env.obstacles:
                    plt.scatter(obstacle[0], obstacle[1],1,"k")
            GUI.draw_track_callback()
            GUI.update_idletasks()
            GUI.update()
            
        if GUI.sensor_mode.get()=="LIDAR":
            angle_to_axis = math.atan((env.lateral_dist-env.prev_lateral_dist)/max(abs(env.longit_dist-env.prev_longit_dist),0.001))
            sensor_data = env.get_sensor_data(car.pos, car.dir,car.speed, LIDAR_RESOLUTION , LIDAR_MAX_RANGE)
#            state = np.hstack([sensor_data[0:-1]*2/LIDAR_MAX_RANGE-1,
#                               sensor_data[-1]*2-1,
#                               env.longit_dist*2/1000-1,
#                               env.lateral_dist/env.track_width-0.5,
#                               math.sin(angle_to_axis)*car.speed/15,
#                               math.cos(angle_to_axis)*car.speed/30])
            state = np.hstack([sensor_data[0:-1]*2/LIDAR_MAX_RANGE-1,
                               sensor_data[-1]*2-1,
                               car.speed/30])
        else:
            state = np.hstack([(car.pos[0]*2/GUI.track_img.size[0])-1,
                                  (car.pos[1]*2/GUI.track_img.size[1])-1,
                                  car.speed/30,
                                  car.dir/(math.pi*2)])
        step = -1
        trials = 0
        
        while not over and step < MAX_STEPS_PER_LAP-1:
            step += 1
            total_steps +=1
            if  (step == 0 or exploration[step-1]<=2.5) and buff.num_items > 2000:
                exploration[step]  = exploration[step] - 1/EXPLORE 
#            epsilon = 0 
            a = np.zeros([1,action_dim])
            noise = np.zeros([1,action_dim])
            
# =============================================================================
#           We get the our actions here (acc, steer_angle)
# =============================================================================            
            
            if GUI.sensor_mode.get()=="LIDAR":
                state_1 = np.atleast_2d(state[0:-2])
                state_1 = np.expand_dims(state_1,axis=2)
                
                state_2 = np.atleast_2d(state[-2:])
            else:
                state_1 = np.atleast_2d(state[0:2])
                state_2 = np.atleast_2d(state[2:4])
            
            
            q_preds = critic.model.predict_on_batch([state_1,state_2])
#            a_index = np.argmax(q_preds,axis=0)
            p = softmax(q_preds.reshape(-1)/(max(exploration[step]*train,0.0001)))
            a_index = np.random.choice(range((ACC_RESOLUTION+1)*(STEER_ANGLE_RESOLUTION+1)),1,p=p)
            a_original = sampled_actions[a_index,:]
            
            
#            noise[0][0] = max(epsilon, 0) * ou.function(a_original[0][0],  0.5 , 1.00, 0.10)
#            noise[0][1] = max(epsilon, 0) * ou.function(a_original[0][1],  0.0 , 0.60, 0.10)
            a = a_original[0] + noise[0]
            a[0] = np.clip(a[0],-1,1)
            a[1] = np.clip(a[1],-1,1)
            acc, steer_angle = a*np.array([ACC_SCALE,STEER_ANGLE_SCALE])
#            steer_angle = 0
#            acc = 8
            prev_pos = deepcopy(car.pos)
            prev_speed = deepcopy(car.speed)
            prev_dir = deepcopy(car.dir)
            env.step(car,acc,steer_angle)
            over,result = env.is_over(car,prev_pos)
# =============================================================================
#             DEBUG
# =============================================================================
            if GUI.debug_active.get():
                print("###########ENTER DEBUG MODE##############")
                print("BUFFER: ")
                print(buff.buffer.describe())
                print(buff.buffer.tail(30))
                print("EXPLORATION: ")
                print(exploration)
                print("Q prediction: ")
                print(q_preds.reshape(ACC_RESOLUTION+1,STEER_ANGLE_RESOLUTION+1))
                critic.model.save_weights("criticmodel.h5", overwrite=True)
                GUI.debug_active.set(False)      
# =============================================================================
#(prev_pos,prev_dir,prev_speed),(acc,steer_angle),r,(car.pos,car.dir,car.speed) OR
#(prev sensor information, prev orientation flag,prev_speed),(acc,steer_angle),r,(prev sensor information, prev orientation flag,prev_speed)
# =============================================================================
            if GUI.sensor_mode.get()=="LIDAR":
                angle_to_axis = math.atan((env.lateral_dist-env.prev_lateral_dist)/max(abs(env.longit_dist-env.prev_longit_dist),0.001))
                sensor_data = env.get_sensor_data(prev_pos, prev_dir,prev_speed, LIDAR_RESOLUTION , LIDAR_MAX_RANGE)
#                state = np.hstack([sensor_data[0:-1]*2/LIDAR_MAX_RANGE-1,
#                               sensor_data[-1]*2-1,
#                               env.longit_dist*2/1000-1,
#                               env.lateral_dist/env.track_width-0.5,
#                               math.sin(angle_to_axis)*prev_speed/15,
#                               math.cos(angle_to_axis)*prev_speed/30])
                state = np.hstack([sensor_data[0:-1]*2/LIDAR_MAX_RANGE-1,
                               sensor_data[-1]*2-1,
                               prev_speed/30])
            else:
                state = np.hstack([(prev_pos[0]*2/GUI.track_img.size[0])-1,
                                  (prev_pos[1]*2/GUI.track_img.size[1])-1,
                                  prev_speed/30,
                                  prev_dir/(math.pi*2)])
# =============================================================================
#             Observing reward
# =============================================================================
            if over and not result:
                trials +=1
                r = -10 \
#                    -(abs(a[0]-car.prev_acc)/(2*ACC_SCALE))**2 \
#                    -(abs(a[1]-car.prev_steering)/(2*STEER_ANGLE_SCALE))*abs(prev_speed)/10
            elif not over:
                trials = 0
                r = float(env.get_reward(car.pos)/5)
            r = np.clip(r, -10, 40)    
            
            car.prev_acc = a[0]
            car.prev_steering = a[1]
            cumulative_r = cumulative_r + r
            if GUI.sensor_mode.get()=="LIDAR":
                angle_to_axis = math.atan((env.lateral_dist-env.prev_lateral_dist)/max(abs(env.longit_dist-env.prev_longit_dist),0.001))
                next_sensor_data = env.get_sensor_data(car.pos, car.dir,car.speed, LIDAR_RESOLUTION , LIDAR_MAX_RANGE)
#                next_state = np.hstack([next_sensor_data[0:-1]*2/LIDAR_MAX_RANGE-1,
#                               next_sensor_data[-1]*2-1,
#                               env.longit_dist*2/1000-1,
#                               env.lateral_dist/env.track_width-0.5,
#                               math.sin(angle_to_axis)*car.speed/15,
#                               math.cos(angle_to_axis)*car.speed/30])
                next_state = np.hstack([next_sensor_data[0:-1]*2/LIDAR_MAX_RANGE-1,
                               next_sensor_data[-1]*2-1,
                               car.speed/30])
            else:
                next_state = np.hstack([(car.pos[0]*2/GUI.track_img.size[0])-1,
                                  (car.pos[1]*2/GUI.track_img.size[1])-1,
                                  car.speed/30,
                                  car.dir/(math.pi*2)])
# =============================================================================
#           Store experience in repaly memory
# =============================================================================
            string_to_hash = str(state)+str(a)+str(r)+str(next_state)+str(over)
            hashed_string = hashlib.md5(string_to_hash.encode('utf-8')).hexdigest()
            experience =  pd.DataFrame([[state.reshape(-1),a,r,next_state.reshape(-1),bool(over),hashed_string,0]],columns=['s', 'a', 'r', "s'", 'over','id','p'],copy = True)

            if not(over and result):
                if step > 0:
                    r_trace = np.asarray([GAMMA**(j+1) for j in range(step)])*r
                    r_trace = r_trace[-1::-1]
                    experience_batch.loc[:,'r'] = np.clip(experience_batch.loc[:,'r']+r_trace,-10*GAMMA,100)
                    experience_batch = pd.concat([experience_batch,experience],ignore_index=True)
                else:
                    experience_batch = experience
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
                track_view = GUI.track_figure_handle.add_axes([0,0,1,1])
                track_view.imshow(GUI.track_img,aspect='auto')
                track_view.plot([env.start_line[0,0],env.start_line[1,0]], [env.start_line[0,1],
                          env.start_line[1,1]], color='g', linestyle='-', linewidth=1)
                track_view.plot([chassis[0,0],chassis[0,1]], [chassis[1,0],chassis[1,1]],
                         color=car_color, linestyle='-', linewidth=2)
                track_view.plot([chassis[0,1],chassis[0,2]], [chassis[1,1],chassis[1,2]],
                         color=car_color, linestyle='-', linewidth=2)
                track_view.plot([chassis[0,2],chassis[0,3]], [chassis[1,2],chassis[1,3]],
                         color=car_color, linestyle='-', linewidth=2)
                track_view.plot([chassis[0,3],chassis[0,0]], [chassis[1,3],chassis[1,0]],
                         color=car_color, linestyle='-', linewidth=2)
                if GUI.obstacles:
                    for obstacle in env.obstacles:
                        plt.scatter(obstacle[0], obstacle[1],1,"k")
                GUI.draw_track_callback()
            GUI.update_idletasks()
            GUI.update()
# =============================================================================
#           For generating data  
# =============================================================================
#            if over and not result and trials <= 15:
#                over=False
#                car_color = 'k'
#                car.speed = deepcopy(prev_speed)
#                car.dir = deepcopy(prev_dir)
#                car.pos = deepcopy(prev_pos)
#                if GUI.draw_track.get():
#                    theta = car.dir
#                    rot_matrix = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=float)
#                    chassis = np.array([[0,car.width,car.width,0],[0,0,car.length,car.length]])
#                    chassis[0,:] -= car.width/2
#                    chassis[1,:] -= car.length/2
#                    chassis = rot_matrix.dot(chassis)
#                    chassis[0,:] = chassis[0,:] + car.pos[0] 
#                    chassis[1,:] = chassis[1,:] + car.pos[1]
#                    chassis= np.round(chassis)
#                    chassis = chassis.astype(int)
#                    GUI.track_figure_handle.clear()
#                    track = GUI.track_figure_handle.add_subplot(111)
#                    plt.imshow(GUI.track_img,aspect='auto')
#                    plt.plot([env.start_line[0,0],env.start_line[1,0]], [env.start_line[0,1],
#                              env.start_line[1,1]], color='g', linestyle='-', linewidth=1)
#                    plt.plot([chassis[0,0],chassis[0,1]], [chassis[1,0],chassis[1,1]],
#                             color=car_color, linestyle='-', linewidth=2)
#                    plt.plot([chassis[0,1],chassis[0,2]], [chassis[1,1],chassis[1,2]],
#                             color=car_color, linestyle='-', linewidth=2)
#                    plt.plot([chassis[0,2],chassis[0,3]], [chassis[1,2],chassis[1,3]],
#                             color=car_color, linestyle='-', linewidth=2)
#                    plt.plot([chassis[0,3],chassis[0,0]], [chassis[1,3],chassis[1,0]],
#                             color=car_color, linestyle='-', linewidth=2)
#                    if GUI.obstacles:
#                        for obstacle in env.obstacles:
#                            plt.scatter(obstacle[0], obstacle[1],1,"k")
#                    GUI.draw_track_callback()
#                    GUI.update_idletasks()
#                    GUI.update()
# =============================================================================
#           Train the networks with a mini batch.
# =============================================================================
            if buff.num_items > 2000 and train:
                batch, indeces = buff.get_batch(BATCH_SIZE)
                states = np.asarray(np.atleast_2d([e[0] for e in batch]))
                actions = np.asarray(np.atleast_2d([e[1] for e in batch]))
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray(np.atleast_2d([e[3] for e in batch]))
                overs = np.asarray([e[4] for e in batch])
                
                if GUI.sensor_mode.get()=="LIDAR":
                    states_1 = states[:,0:-2]
                    states_1 = np.expand_dims(states_1,axis=2)
                    states_2 = states[:,-2:]
                    new_states_1 = new_states[:,0:-2]
                    new_states_1 = np.expand_dims(new_states_1,axis=2)
                    new_states_2 = new_states[:,-2:]
                else:
                    states_1 = states[:,0:2]
                    states_2 = states[:,2:4]
                    new_states_1 = new_states[:,0:2]
                    new_states_2 = new_states[:,2:4]
    #                target_q_values = critic.target_model.predict([new_states_1,new_states_2,
    #                                                                   actor.target_model.predict([new_states_1,new_states_2])])  
                y = critic.model.predict_on_batch([states_1,states_2])
#                q_preds = critic.model.predict_on_batch([new_states_1,new_states_2])
#                next_a_indeces = np.argmax(q_preds,axis=1)
#                maxQs = critic.target_model.predict_on_batch([new_states_1,new_states_2])
#                maxQs = maxQs[range(len(batch)),next_a_indeces]
                targets = rewards  #+ GAMMA*maxQs*(1-overs)
                a_indeces = []
                for row in actions:
                    a_indeces.append(int(np.where(np.all(sampled_actions==row,axis=1))[0]))
#                errors = (y[range(len(batch)),a_indeces] - targets)
#                buff.update_priorities(indeces,errors)
                y[range(len(batch)),a_indeces] = targets
#                    critic.model.fit([states_1,states_2,actions], y.reshape(-1,1),batch_size=32)        
                critic.model.train_on_batch([states_1,states_2], y)
    #                a_for_grad = actor.model.predict([states_1,states_2])
    #                
    #                grads = critic.gradients(states, a_for_grad)
    #                if GUI.draw_track.get():
    #                    a[1]=math.degrees(a[1])
    #                    print("Reward: "+str(r)+"\tActions: "+str(a))
    #                    print("Grads:"+"\n\tMax:"+str(np.max(grads[:,:]))
    #                                +"\n\tMin:"+str(np.min(grads[:,:]))
    #                                +"\n\tMean:"+str(np.mean(grads[:,:]))
    #                                +"\n\tStd:"+str(np.std(grads[:,:])))
    #                    
    #                actor.train(states, grads)
    #                actor.target_train()
#                if total_steps % TARGET_UPDATE_FREQ == 0:     
#                    critic.target_train()
#                    TARGET_UPDATE_FREQ = int(TARGET_UPDATE_FREQ * TARGET_UPDATE_FREQ_EXP) 
            
        
        elapsed_time = time.time() - start_time
        buff.add_item(experience_batch)
        log.append(cumulative_r)
        if int(i/100) > 0 and i % 100 == 0:
            if best_reward < cumulative_r:
                try:
                    critic.model.save_weights("criticmodel_best.h5", overwrite=True)
                    print("Weights saved successfully")
                except:
                    print("Cannot save the weights")
                best_reward = deepcopy(cumulative_r)
                best_test_index = deepcopy(i)
            print("Episodes "+str(i-100)+"-"+str(i)+"\tAverage reward:"+str(np.mean(np.asarray(log[-100:])))+"\tET: "+str(elapsed_time))
            print("Best at: "+str(best_test_index)+"\t Best reward:"+str(best_reward)+"\tStep: "+str(step))
# =============================================================================
#   Save the weights
# =============================================================================
    buff.buffer.to_json('data_x.json')
    if (GUI.save_nn.get()):
        print("Now we save model")
        
#        actor.model.save_weights("actormodel.h5", overwrite=True)
#        with open("actormodel.json", "w") as outfile:
#            json.dump(actor.model.to_json(), outfile)
        
        critic.model.save_weights("criticmodel.h5", overwrite=True)
        with open("criticmodel.json", "w") as outfile:
            json.dump(critic.model.to_json(), outfile)




# =============================================================================
# Main program
# =============================================================================

gui = GUI(handler=start_simulation)
gui.mainloop()





