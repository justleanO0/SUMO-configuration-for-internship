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

class TestEnv(gym.Env):
	def __init__(self):
		self.name = 'ego_vehicle'
		self.radius = 637
		self.acc_history = deque([0, 0], maxlen=2)
		self.lane_his = deque([0, 0], maxlen=2)
		self.pos = (0, 0)
		self.curr_edge = ''
		self.curr_lane = -1
		self.target_speed = 0
		self.speed = 0
		self.maxall_speed = 0
		self.acc = 0
		self.angle = 0
		self.gui = False
		self.numVehicles = 0
		self.lane_ids = []
		self.curr_step = 0
		self.total_step = 0
		self.collision = False
		self.total_collision = 0
		self.done = False
		self.nb_lc = 0
		self.nv_segment = np.zeros(4)
		self.collision_step = []
		self.collision_num = []
		self.action_his = []
		self.lc_flag = False
		self.global_count = 0
		self.w_eff = 1
		self.w_comf = 1
		self.w_safe = 3
		# dynamic collision reward
		self.r = 0
		self.free = 0
		self.action_local=0
		self.action_gobal=0
		self.action = 0

        # urgent lc request for global control
		self.t_ULCR=10 # 10s to change lane
		self.thre_changemode=5
		self.nb_neigh = 0

		self.risk_edge = ''
		self.risky_time = 0
		self.block_time = 0
		self.rl_flag = False
		self.am_flag = False

		self.model = ""
		self.his_a = []

		self.vid_list = []
		self.state_dict_left = {}
		self.state_dict_right = {}
        
		self.risky_edge_x = 0 # start point of the risky lane, length of 200m
		self.risky_lane_index = 0


	def start(self, gui=True, numVehicles=150, network_conf="networks/user_highway/circles.sumocfg", network_xml='networks/user_highway/circless.net.xml', thresh=3, tulcr=10):
		#print("start()")
		self.gui = gui
		self.numVehicles = numVehicles
		self.network_conf = network_conf
		self.net = sumolib.net.readNet(network_xml)
		
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
		self.action_local=0
		self.action_global=0
		self.action = 0

        # urgent lc request for global control
		self.t_ULCR=tulcr
		self.global_count = 0
		self.w_eff = 1
		self.w_comf = 1
		self.w_safe = 3
		self.thre_changemode = thresh
		self.nb_neigh = 0

		self.action_his = []
        
		self.load_path = "E:/hjl_simulations/xgboost_rl.model"
		self.model = joblib.load(self.load_path)
        
		for i in range(numVehicles):
			self.his_a.append(deque(np.zeros(40),maxlen=40))

		home = "D:"

		if self.gui:
			#sumoBinary = home + "/sumo/bin/sumo-gui"
			sumoBinary = home + "/sumo-1.8.0/bin/sumo"
		else:
			sumoBinary = home + "/sumo-1.8.0/bin/sumo"
		sumoCmd = [sumoBinary, "-c", self.network_conf, "--no-step-log", "true", "-W"]
		traci.start(sumoCmd)

		self.lane_ids = traci.lane.getIDList()
		re = random.randint(0,13) # random edge sumo network of 13 egdes
		rl = random.randint(0,2) # random lane three lanes
		self.risk_edge = "edge" + str(re) + "_" + str(rl) 
        
	# Do some random step to distribute the vehicles
		for i in range(numVehicles):
			if i%5 == 0:
				ambulance_id = int(i/5)
				a_name = "ambulance_"+ str(ambulance_id)
				traci.vehicle.add(a_name, routeID='route_0', typeID="ambulance",departSpeed="random",departLane="random", departPos="random" )
				traci.vehicle.setLaneChangeMode(a_name, 256)
			else:
				veh_name = 'vehicle_' + str(i+1)
				r = random.random()
				if r>0 and r<=0.03:
					traci.vehicle.add(veh_name, routeID='route_0', typeID="car_abnormal",departSpeed="random",departLane="random", departPos="random" )
				elif r>0.03 and r<=0.06:
					traci.vehicle.add(veh_name, routeID='route_0', typeID="bus", departSpeed="random",departLane="random",departPos="random" )
				elif r>0.06 and r<=0.09:
					traci.vehicle.add(veh_name, routeID='route_0', typeID="track", departSpeed="random",departLane="random",departPos="random" )
				elif r>0.09 and r<=0.2:
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
					traci.vehicle.add(veh_name, routeID='route_0', typeID="car8", departSpeed="random",departLane="random",departPos="random" )

				traci.vehicle.setLaneChangeMode(veh_name, 256)
				self.state_dict_left[veh_name]=[ [0 for col in range (9)] for row in range (40)]
				self.state_dict_right[veh_name]=[ [0 for col in range (9)] for row in range (40)]
                
        # add ego vehicle
		traci.vehicle.add(self.name, routeID='route_0', typeID='ego') 
        
		# Do some random step to distribute the vehicles
		for step in range(self.numVehicles*2):
			traci.simulationStep()
		# Setting the lane change mode to 0 meaning that we disable any autonomous lane change and collision avoidance of ego vehicle
		traci.vehicle.setLaneChangeMode(self.name, 0)

		# Setting up useful parameters
		self.curr_lane = traci.vehicle.getLaneIndex(self.name)
		self.maxall_speed = traci.vehicle.getAllowedSpeed(self.name)
		self.update_params()


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
		self.vid_list = traci.vehicle.getIDList()


	def update_nv_segment(self):
		v_id = traci.vehicle.getIDList()
		self.nv_segment = np.zeros(4)
		for v in v_id:
			pos_v = traci.vehicle.getPosition(v)
			if pos_v[0]>0:
				if pos_v[1]>0:
					self.nv_segment[0] += 1
				elif pos_v[1]<0:
					self.nv_segment[3] += 1
			elif pos_v[0]<0:
				if pos_v[1]>0:
					self.nv_segment[1] += 1
				elif pos_v[1]<0:
					self.nv_segment[2] += 1


	def get_vehicle_info(self,vehicle_name):
		# X position, Y position, lane ID, velocity, acceleration
		[x,y] = traci.vehicle.getPosition(vehicle_name)
		lane = traci.vehicle.getLaneID(vehicle_name)
		lane_id = lane.split("_")[-1]
		speed = traci.vehicle.getSpeed(vehicle_name)
		acc = traci.vehicle.getAcceleration(vehicle_name)
		return x,y,speed,acc,lane_id


	def get_state(self):
		# Define a state as a vector of vehicles information
		# get neighbor
		arr = []
		count_neigh = 0
		# self info x,y,speed,acc,lane_id
		info = self.get_vehicle_info(self.name)
		if self.curr_edge==self.risk_edge:
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
				count_neigh += 1
				n_info = self.get_vehicle_info(n[0])
				for i in n_info:
					arr.append(i)
				arr.append(n[1]) # dist to ego
				if traci.vehicle.getTypeID(n[0])=='ambulance':
					arr.append(1)
				else:
					arr.append(0)

		state=np.array(list(map(float,arr)))
		return state, count_neigh # 1*48


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
			# Reward function is made of three elements:
			# - Comfort 
			 #- Efficiency
			# - Safety

		# Comfort reward 
		jerk = self.compute_jerk()

		R_comf = - jerk**2

		#Efficiency reward
		# Speed
		R_speed = -np.abs(self.speed - self.target_speed)
		# Penalty for changing lane
		if self.lc_flag:
			R_change = -1
		else:
			R_change = 0
		self.lc_flag = False
		# Rewards Parameters
		R_eff = R_speed + R_change

        # Safety Reward
		# Vehicle density
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
		
		'''# Define ego driving mode: speed/safe
		if self.nb_neigh<=self.thre_changemode:
            # speed mode
			#print("speed mode")
			self.w_eff=1
			self.w_comf=1
			self.w_safe=1
		elif self.nb_neigh>self.thre_changemode:
            # safe mode
			#print("safety mode")
			self.w_eff=1
			self.w_comf=1
			self.w_safe=3'''

		# Dynamic r
		if collision:
			self.r -= 1
		else:
			self.free += 1
			if self.free >= 10:
				self.r = 0
				self.free = 0
		R_colli = self.r

        # Penalize driving on risky lane
		if self.rl_flag==True:
			R_risk = -1
		else:
			R_risk = 0

        # Penalize blocking an emergency vehicle
		if self.am_flag==True:
			R_block = -1
		else:
			R_block = 0

		R_safe = R_colli + R_den + R_risk + R_block

		self.rl_flag = False
		self.am_flag = False

		# total reward
		R_tot =  self.w_comf*R_comf + self.w_eff*R_eff + self.w_safe*R_safe
		return [R_tot, R_comf, R_eff, R_safe]


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


	def step(self, a):
		#This will :
		#- send action, namely change lane to right/left or stay 
		#- do a simulation step
		#- compute reward
		#- update agent params 
		#- compute nextstate
		#- return nextstate, reward and done

		if self.curr_step%20==0:
        # update surrounding vehicles' position by NGSIM model
			for v in self.vid_list:
				if v in self.state_dict_right:
					self.update_v_state_dict(v)
					act_r = self.ng_action_predict(self.state_dict_right[v])
					act_l = self.ng_action_predict(self.state_dict_left[v])
					#print(v, " act_r: ", act_r,"act_l: ", act_l)
					if act_r[0]==2: # turn right
						if act_l[0]!=1:
							if traci.vehicle.getLaneIndex(v)==2:
								traci.vehicle.changeLane(v,1,2)
							elif traci.vehicle.getLaneIndex(v)==1:
								traci.vehicle.changeLane(v,0,2)
								
					if act_l[0]==1: # turn left
							if act_r[0]!=2:
								if traci.vehicle.getLaneIndex(v)==0:
									traci.vehicle.changeLane(v,1,2)
								elif traci.vehicle.getLaneIndex(v)==1:
									traci.vehicle.changeLane(v,2,2)

		action = a
		self.action = action

		collision = False

        # update ego vehicle's lane
		if self.lc_flag:
			self.action_his.append(action)
		collision = False
		# Action legend : 0 stay, 1 change to right, 2 change to left
		if action != 0:
		# if at edge lane and action is change lane, then collision = True
			if action == 1:
				if self.curr_lane == 1:
					traci.vehicle.changeLane(self.name, 0, 2)
				elif self.curr_lane == 2:
					traci.vehicle.changeLane(self.name, 1, 2)
				elif self.curr_lane == 0:
					collision = True
			if action == 2:
				if self.curr_lane == 0:
					traci.vehicle.changeLane(self.name, 1,2)
				elif self.curr_lane == 1:
					traci.vehicle.changeLane(self.name, 2, 2)
				elif self.curr_lane == 2:
					collision = True

		# Sim step
		traci.simulationStep()

		# Check road collision
		self.detect_collision()
		collision = self.collision + collision

		reward = self.compute_reward(collision, action)

		# Update agent params 
		self.update_params()
		# State 
		next_state, nb_neigh = self.get_state()
		self.nb_neigh = nb_neigh
		# Update curr state
		self.curr_step += 1
		self.total_step += 1
		done = collision
		return next_state, reward, done, collision
		
	def render(self, mode='human', close=False):
		pass

	def reset(self, gui=False, numVehicles=150, thr=3, ttulcr=10):
		self.start(gui, numVehicles, thresh = thr, tulcr=ttulcr)
		return self.get_state()[0]


	def close(self):
		print("Total step:", self.total_step)
		print("Total vehicle number:", traci.vehicle.getIDCount())
		traci.close()