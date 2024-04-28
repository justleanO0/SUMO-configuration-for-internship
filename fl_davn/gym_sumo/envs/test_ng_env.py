# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:33:26 2022

@author: Jialin
"""

import gym
import traci
import sumolib
import numpy as np
import random
from collections import deque
import joblib

class NgSumoEnv(gym.Env):
	def __init__(self):
		self.gui = False
		self.numVehicles = 0
		self.lane_ids = []
		self.total_step = 0
		self.vid_list = []
		self.state_dict_left = {}
		self.state_dict_right = {}
		self.lc_threshold = [0.21, 0.26, 0.24, 0.18, 0.1]
		self.name = 'ego_vehicle'
		self.radius = 637
		self.step_length = 0.4 
		self.acc_history = deque([0, 0], maxlen=2)
		self.lane_his = deque([0, 0], maxlen=2)
		self.grid_state_dim = 3
		self.state_dim = (4*self.grid_state_dim*self.grid_state_dim)+1 # 5 info for the agent, 4 for everybody else
		self.pos = (0, 0)
		self.curr_edge = ''
		self.curr_lane = -1
		self.target_speed = 0
		self.speed = 0
		self.acc = 0
		self.angle = 0
		self.gui = False
		self.numVehicles = 0
		self.lane_ids = []
		self.edge_ids = []
		self.curr_step = 0
		self.total_step = 0
		self.collision = False
		self.total_collision = 0
		self.done = False
		self.nb_lc = 0
		self.nb_neighbor = 0
		self.nv_segment = np.zeros(4)
		self.emergency_brake = 0
		self.collision_step = []
		self.collision_num = []
		self.action_his = []
		self.lc_flag = False

		# dynamic collision reward
		self.r = 0
		self.free = 0
        
		self.risky_edge_x = 0 # start point of the risky lane, length of 200m
		self.risky_lane_index = 0
        
		self.risky_time = 0
		self.block_time = 0
		self.rl_flag = False
		self.am_flag = False
		self.model = ""
		self.his_a = [None]*5


	def start(self, gui=True, numVehicles=25, network_conf="E:/hjl_simulations/networks/Ngsim/straight4kmway.sumocfg", network_xml='E:/hjl_simulations/networks/Ngsim/straight4kmway.net.xml'):
		self.gui = gui
		self.numVehicles = numVehicles
		self.network_conf = network_conf
		self.net = sumolib.net.readNet(network_xml)
		self.load_path = "E:\\hjl_simulations\\xgboost_rl.model"
		self.model = joblib.load(self.load_path)
		self.gui = gui

		self.curr_step = 0
		self.total_collision = 0
		self.collision = False
		self.nb_lc = 0
		self.state_dict_left = {}
		self.state_dict_right = {}
		self.done = False
		self.nv_segment = np.zeros(4)
		self.rl_flag = False
		self.am_flag = False
		self.lc_flag = False
		self.risky_time = 0
		self.block_time = 0
		self.r = 0
		self.free = 0
		self.done = False
		self.emergency_brake = 0
		self.action_his = []
		for i in range(numVehicles):
			self.his_a.append(deque(np.zeros(40),maxlen=40))

		home = "D:"
		if self.gui:
			#sumoBinary = home + "/sumo/bin/sumo-gui"
			sumoBinary = home + "/sumo-1.8.0/bin/sumo-gui"
		else:
			sumoBinary = home + "/sumo-1.8.0/bin/sumo"
		sumoCmd = [sumoBinary, "-c", self.network_conf, "--no-step-log", "true", "-W"]
		traci.start(sumoCmd)

        # generate risky lane
		self.risky_edge_x = random.randint(0,3800)
		self.risky_lane_index = traci.vehicle.getLaneIndex()
        
        # populating vehicles
		for i in range(numVehicles):
			veh_name = 'vehicle_' + str(i)
			r = random.random()
            # add vehicle
			self.add_vehicle(r,veh_name)
			traci.simulationStep()
            
        # add ego vehicle
		traci.vehicle.add(self.name, routeID='route_0', typeID='ego')
		traci.simulationStep()
		traci.vehicle.setLaneChangeMode(self.name, 0)
		self.curr_lane = traci.vehicle.getLaneIndex(self.name)
		self.update_params()


	def add_vehicle(self, r, veh_name):
		if r>0 and r<=0.1:
			traci.vehicle.add(veh_name, routeID='route_0', typeID="car_abnormal",departSpeed="random",departLane="random", departPos="random" )
            # departPos: offset in meters from the start of the lane where the vehicle should be added
		elif r>0.1 and r<=0.2:
			traci.vehicle.add(veh_name, routeID='route_0', typeID="car_normal", departSpeed="random",departLane="random",departPos="random" )
		elif r>0.2 and r<=0.3:
			traci.vehicle.add(veh_name, routeID='route_0', typeID="car1", departSpeed="random",departLane="random",departPos="random" )
		elif r>0.3 and r<=0.4:
			traci.vehicle.add(veh_name, routeID='route_0', typeID="car2", departSpeed="random",departLane="random",departPos="random" )
		elif r>0.4 and r<=0.5:
			traci.vehicle.add(veh_name, routeID='route_0', typeID="car3", departSpeed="random",departLane="random",departPos="random" )
		elif r>0.5 and r<=0.6:
			traci.vehicle.add(veh_name, routeID='route_0', typeID="car4", departSpeed="random",departLane="random",departPos="random" )
		elif r>0.6 and r<=0.7:
			traci.vehicle.add(veh_name, routeID='route_0', typeID="car5", departSpeed="random",departLane="random",departPos="random" )
		elif r>0.7 and r<=0.8:
			traci.vehicle.add(veh_name, routeID='route_0', typeID="car6", departSpeed="random",departLane="random",departPos="random" )
		elif r>0.8 and r<=0.9:
			traci.vehicle.add(veh_name, routeID='route_0', typeID="car7", departSpeed="random",departLane="random",departPos="random" )
		else:
 			#r in [0.9,1-np.exp(-10)]:
			traci.vehicle.add(veh_name, routeID='route_0', typeID="car8", departSpeed="random",departLane="random",departPos="random" )
		traci.vehicle.setLaneChangeMode(veh_name, 256)
		self.state_dict_left[veh_name]=[ [0 for col in range (9)] for row in range (40)]
		self.state_dict_right[veh_name]=[ [0 for col in range (9)] for row in range (40)]


	def get_v_state(self, vid):
		# get neighbor
		arr_left = []
		arr_right = []
		x0 =  traci.vehicle.getPosition(vid)[0]
		v0 = traci.vehicle.getSpeed(vid)
		arr_left.append(v0)
		arr_right.append(v0)
        
		# neighbors' IDs
		leader = traci.vehicle.getLeader(vid)
		leftleader = traci.vehicle.getLeftLeaders(vid)
		if len(leftleader)!=0:
			leftleader = leftleader[0]
		rightleader = traci.vehicle.getRightLeaders(vid)
		if len(rightleader)!=0:
			rightleader = rightleader[0]
		follower = traci.vehicle.getFollower(vid)
		leftfollower = traci.vehicle.getLeftFollowers(vid)
		if len(leftfollower)!=0:
			leftfollower = leftfollower[0]
		rightfollower = traci.vehicle.getRightFollowers(vid)
		if len(rightfollower)!=0:
			rightfollower = rightfollower[0]
		neigh_left = [leader,leftleader,follower,leftfollower]
		neigh_right = [leader,rightleader,follower,rightfollower]
        
		# neighbors info
		# x,speed
		collisions = traci.simulation.getCollidingVehiclesIDList()
		lv = []
		lx = []
		rv = []
		rx = []
		for n in neigh_left:
			if type(n)!=tuple or len(n)==0 or n[0]=='' or (n[0] in collisions) :
				lv.append(0)
				lx.append(0)

			elif len(n)!=0:
				vn = v0 - traci.vehicle.getSpeed(n[0])
				xn = x0 - traci.vehicle.getPosition(n[0])[0]
				lv.append(vn)
				lx.append(xn)
		arr_left.extend(lv)
		arr_left.extend(lx)
        
		for n in neigh_right:
			if type(n)!=tuple or len(n)==0 or n[0]=='' or (n[0] in collisions) :
				rv.append(0)
				rx.append(0)

			elif len(n)!=0:
				vn = v0 - traci.vehicle.getSpeed(n[0])
				xn = x0 - traci.vehicle.getPosition(n[0])[0]
				rv.append(vn)
				rx.append(xn)
		arr_right.extend(rv)
		arr_right.extend(rx)
		return arr_left, arr_right


	def update_v_state_dict(self, vid):
		# update v state
		state_l, state_r = self.get_v_state(vid)
		self.state_dict_left[vid].append(state_l)
		self.state_dict_right[vid].append(state_r)
		state_len = len(self.state_dict_right[vid])
		if state_len>40:
			self.state_dict_right[vid] = self.state_dict_right[vid][state_len-40:]
			self.state_dict_left[vid]= self.state_dict_left[vid][state_len-40:]


	def ng_action_predict(self,state40):
		mean = np.mean(state40,axis=0)
		maax = np.max(state40,axis=0)
		miin = np.min(state40,axis=0)
		noorm = (state40-mean)/(maax-miin+np.exp(-5))
		action = self.model.predict(noorm[-1:])
		return action


	def update_params(self):
		# initialize params
		self.pos = traci.vehicle.getPosition(self.name)
		self.curr_edge = traci.vehicle.getLaneID(self.name)
		if self.curr_edge == '':
			assert self.collision
			while self.name in traci.simulation.getStartingTeleportIDList() or traci.vehicle.getLaneID(self.name) == '':
				traci.simulationStep()
			self.curr_edge = traci.vehicle.getLaneID(self.name)
		self.curr_lane = traci.vehicle.getLaneIndex(self.name)
		self.lane_his.append(self.curr_lane)
		if self.lane_his[0] != self.lane_his[1]:
			self.nb_lc += 1
			self.lc_flag = True
    
		if self.curr_edge==self.risk_edge:
			self.risky_time += 1
			self.rl_flag = True
		f = traci.vehicle.getFollower(self.name)
		if len(f)!=0 and f[0]!='':
			if traci.vehicle.getTypeID(f[0])=="ambulance" and f[1]<=100:
				self.block_time += 1
				self.am_flag = True
		self.target_speed = traci.vehicle.getAllowedSpeed(self.name)
		self.speed = traci.vehicle.getSpeed(self.name)
		self.acc = traci.vehicle.getAcceleration(self.name)
		self.acc_history.append(self.acc)
		self.angle = traci.vehicle.getAngle(self.name) # Returns the angle of the named vehicle within the last step [°]
		self.update_nv_segment()

	def get_ego_state(self):
		# Define a state as a vector of vehicles information
		# get neighbor
		arr = []
		# self info x,y,speed,acc,lane_id
		info = self.get_vehicle_info(self.name)
        
        # on risky lane?
		if traci.vehicle.getLaneIndex(self.name)==self.risky_lane_index:
			if traci.vehicle.getPosition(self.name)[0] in range(self.risky_edge_x, self.risky_edge_x+200):
				arr.append(1)
			else:
				arr.append(0)
		for i in info:
			arr.append(i)
		# neighbors' IDs
		l = traci.vehicle.getLeader(self.name)
		ll = traci.vehicle.getLeftLeaders(self.name)
		if len(ll)!=0:
			ll = ll[0]
		rl = traci.vehicle.getRightLeaders(self.name)
		if len(rl)!=0:
			rl = rl[0]
		f = traci.vehicle.getFollower(self.name)
		lf = traci.vehicle.getLeftFollowers(self.name)
		if len(lf)!=0:
			lf = lf[0]
		rf = traci.vehicle.getRightFollowers(self.name)
		if len(rf)!=0:
			rf = rf[0]
		neigh = [l,ll,rl,f,lf,rf]
     
		# neighbors info
		# x,y,laneID,speed,acc,distance to ego
		collisions = traci.simulation.getCollidingVehiclesIDList()
		for n in neigh:
			if type(n)!=tuple or len(n)==0 or n[0]=='' or n[1]>100 or (n[0] in collisions) :
				for j in range(7):# 7 feature if each neighbor
					arr.append(0)
			elif len(n)!=0:
				n_info = self.get_vehicle_info(n[0])
				for i in n_info:
					arr.append(i)
				arr.append(n[1]) # dist to ego
				if traci.vehicle.getTypeID(n[0])=='ambulance':
					arr.append(1)
				else:
					arr.append(0)

		state=np.array(list(map(float,arr)))
		return state # 1*48


	def reset(self, gui=False, numVehicles=150):
		traci.close()
		self.start(gui, numVehicles)
		return self.get_ego_state()

	def close(self):
		traci.close()

	def compute_jerk(self):
		return self.acc_history[1] - self.acc_history[0]

	def detect_collision(self):
		collisions = traci.simulation.getCollidingVehiclesIDList()
		if self.name in collisions:
			self.collision = True
			return True
		self.collision = False
		return False

		
	def compute_reward(self, collision, action):
		w_eff = 1
		w_comf = 1
		w_safe = 1

		# Comfort reward 
		jerk = self.compute_jerk()
		if np.abs(jerk)>8:
			self.emergency_brake += 1
		R_comf = - jerk**2
		
		#Efficiency reward
		try:
			lane_width = traci.lane.getWidth(traci.vehicle.getLaneID(self.name))
		except:
			print(traci.vehicle.getLaneID(self.name))
			lane_width = 3.2

		# Speed
		R_speed = -np.abs(self.speed - self.target_speed)
		# Penalty for changing lane
		if self.lc_flag:
			R_change = -1
		else:
			R_change = 0
		self.lc_flag = False
		# Rewards Parameters
		w_lane = 1
		w_speed = 1
		w_change = 1

		R_eff = w_eff*(w_speed*R_speed + w_change*R_change)
		
        # Safety Reward
		# Vehicle density
		# Dynamic R 
		if self.pos[0]>0:
			if self.pos[1]>0:
				R_den = -self.nv_segment[0]
			elif self.pos[1]<0:
				R_den = -self.nv_segment[3]
		elif self.pos[0]<0:
			if self.pos[1]>0:
				R_den = -self.nv_segment[1]
			elif self.pos[1]<0:
				R_den = -self.nv_segment[2]

        #Penalize collision
		if collision:
			self.r -= 1
		else:
			if self.free >= 10:
				self.r = 0
		R_colli = self.r
		if self.rl_flag==True:
			R_risk = -1
		else:
			R_risk = 0
		if self.am_flag==True:
			R_block = -1
		else:
			R_block = 0

		R_safe = (R_colli + R_den + R_risk + R_block)
		self.rl_flag = False
		self.am_flag = False

		# drive in risky lane
		# ...
		# total reward
		R_tot =  w_comf*R_comf + w_eff*R_eff +  w_safe*R_safe
		return [R_tot, R_comf, R_eff, R_safe]
		
    
    
	def step(self, action):
		#This will :
		#- send action, namely change lane to right/left or stay 
		#- do a simulation step
		#- compute reward
		#- update agent params 
		#- compute nextstate
		#- return nextstate, reward and done

        # update vlist
		vlist = traci.vehicle.getIDList()
		delv = []
		for v in self.state_dict_left:
			if v in vlist:
				self.update_v_state_dict(v)
			else:
				delv.append(v)
            # delete key=vid from dictionary
		for j in delv:
			del self.state_dict_left[j]
			del self.state_dict_right[j]
            
        # update surrounding vehicles' position by NGSIM model
		for v in vlist:
			if v in self.state_dict_right:
				act_r = self.ng_action_predict(self.state_dict_right[v])
				act_l = self.ng_action_predict(self.state_dict_left[v])
				if act_r[0]==2:
					if act_l[0]==1:
						print("conflit")
					elif act_l[0]!=1:
						if traci.vehicle.getLaneIndex(v)-1>=0:
							if random.random()<self.lc_threshold[traci.vehicle.getLaneIndex(v)]:
								traci.vehicle.changeLane(v,traci.vehicle.getLaneIndex(v)-1,2)
								print(v, " turn right. Now on lane ", traci.vehicle.getLaneIndex(v))
				if act_l[0]==1:
						if act_r[0]==2:
							print("conflit")
                    # turn left
						elif act_r[0]!=2:
							if traci.vehicle.getLaneIndex(v)+1<=4:
								if random.random()<self.lc_threshold[traci.vehicle.getLaneIndex(v)]:
									traci.vehicle.changeLane(v,traci.vehicle.getLaneIndex(v)+1,2)
									print(v, " turn left. Now on lane ", traci.vehicle.getLaneIndex(v))

        # update ego state
		if self.lc_flag:
			self.action_his.append(action)
		collision = False
		# Action legend : 0 stay, 1 change to right, 2 change to left
		if action != 0:
		# if at edge lane and action is change lane, then collision = True
			if action == 1: # turn left
				if self.curr_lane == 0:
					traci.vehicle.changeLane(self.name, 1, 2)
				elif self.curr_lane == 1:
					traci.vehicle.changeLane(self.name, 2, 2)
				elif self.curr_lane == 2:
					traci.vehicle.changeLane(self.name, 3, 2)
				elif self.curr_lane == 3:
					traci.vehicle.changeLane(self.name, 4, 2)
				elif self.curr_lane == 4:
					collision = True
			if action == 2: # turn right
				if self.curr_lane == 1:
					traci.vehicle.changeLane(self.name, 0,2)
				elif self.curr_lane == 2:
					traci.vehicle.changeLane(self.name, 1, 2)
				elif self.curr_lane == 3:
					traci.vehicle.changeLane(self.name, 2, 2)
				elif self.curr_lane == 4:
					traci.vehicle.changeLane(self.name, 3, 2)
				elif self.curr_lane == 0:
					collision = True

		traci.simulationStep()
		collisions = traci.simulation.getCollidingVehiclesIDList()
		print(collisions)
		self.curr_step += 1

		# Check road collision
		collision = self.detect_collision() + collision
		if not collision:
			self.free += 1

		#if collision:
		if self.detect_collision():
			self.total_collision += 1
			self.collision_step.append(self.total_step)
			self.collision_num.append(self.total_collision)

        # compute reward
		reward = self.compute_reward(collision, action)

		# Update agent params 
		self.update_params()

		# State 
		next_state = self.get_ego_state()
		# Update curr state
		self.curr_step += 1
		self.total_step += 1
		
		# Return
		done = collision

		return next_state, reward, done, collision
