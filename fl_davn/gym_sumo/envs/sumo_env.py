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
import math

def d2v_achivabel_datarate(POS_uav=[400,600,150],POS_veh=[200,300],
                             FC=2.4*math.pow(10,9), ETA_LoSDB=1,ETA_NLoSDB=20,
                             C=3*math.pow(10,8),N_0DB=-174, P_TRANS=0.28, BW=100*10^6):
    xu_i,yu_i,zu_i = POS_uav
    xv_i,yv_i = POS_veh
    
    a = 14.39
    b = 0.13
    h_i = zu_i
    dhor_ij = np.sqrt((xu_i-xv_i)*(xu_i-xv_i)+(yu_i-yv_i)*(yu_i-yv_i))
    theta_ij = np.arctan(h_i/dhor_ij)
    prob_LoS_ij = 1/(1+a*np.exp(-b*(theta_ij-a)))
    prob_NLoS_ij = 1-prob_LoS_ij

    fc = FC  #Hz
    eta_LoSdB = ETA_LoSDB #dB
    eta_NLoSdB = ETA_NLoSDB #dB
    eta_LoS = math.pow(10,eta_LoSdB/10) #W
    eta_NLoS = math.pow(10,eta_NLoSdB/10) #W
    c = 3*math.pow(10,8) #m/s
    deuc_ij = np.sqrt((h_i)*(h_i)+(dhor_ij)*(dhor_ij))
    pl_LoS_ij = 20*np.log10(4*np.pi*fc*deuc_ij/c) + eta_LoS
    pl_NLoS_ij = 20*np.log10(4*np.pi*fc*deuc_ij/c) + eta_NLoS
    pl_ij = prob_LoS_ij*pl_LoS_ij + prob_NLoS_ij*pl_NLoS_ij

    G_ij = 1/pl_ij

    N_0dB = N_0DB #dB/Hz
    N_0 = math.pow(10,N_0dB/10) #W
    p_trans_i = P_TRANS #W

    SNR_ij = p_trans_i*G_ij/N_0

    B = BW #Hz
    C_ij = B*np.log2(1+SNR_ij)
    return C_ij

def get_d2v_communication_energy(P_uav=[400,600,150],P_veh=[200,300],SI=639104, P_TRANS=0.28):
    C_ij = d2v_achivabel_datarate(POS_uav=P_uav,POS_veh=P_veh)
    S_i = SI #bytes
    p_trans_i = P_TRANS
    E_ij = S_i*p_trans_i/C_ij
    return E_ij


def get_d2d_communication_energy(POS_uav1=[400,600,150],POS_uav2=[200,300,100],
                                 POS_uav3=[455,650,170],POS_uav4=[280,370,150],
                             FC=2.4*math.pow(10,9), ETA_LoSDB=1, C=3*math.pow(10,8),
                             N_0DB=-174, P_TRANS=0.28, BW=100*10^6, SI=639104):
    xu1,yu1,zu1 = POS_uav1
    xu2,yu2,zu2 = POS_uav2
    xu3,yu3,zu3 = POS_uav3
    xu4,yu4,zu4 = POS_uav4
    
    deuc_nm12 = np.sqrt((xu1-xu2)*(xu1-xu2)+(yu1-yu2)*(yu1-yu2))
    deuc_nm13 = np.sqrt((xu1-xu3)*(xu1-xu3)+(yu1-yu3)*(yu1-yu3))
    deuc_nm14 = np.sqrt((xu1-xu4)*(xu1-xu4)+(yu1-yu4)*(yu1-yu4))
    deuc_nm = [deuc_nm12,deuc_nm13,deuc_nm14]
    E_d2d = 0
    
    for d in deuc_nm: 
        if d!=0:
            eta_LoSdB = ETA_LoSDB
            c = C
            pld2d_mn = 20*np.log10(4*np.pi*FC*d/c) + eta_LoSdB
            Gd2d_nm = 1/pld2d_mn
            N_0dB = N_0DB #dB/Hz
            N_0 = math.pow(10,N_0dB/10) #W
            p_trans_i = P_TRANS
            SNRd2d_nm = p_trans_i*Gd2d_nm/N_0
            B = BW
            S_i = SI
            Cd2d_nm = B*np.log2(1+SNRd2d_nm)
            Ed2d_nm = S_i*p_trans_i/Cd2d_nm
            E_d2d += Ed2d_nm
    return E_d2d
    
def uav_mobility_energy(POS1=[0,0,150],POS2=[1,1,151], Vh=1, Vv=1, tau=0.4):
    # tau: hovering time
    #xu1,yu1,zu1 = POS1
    #xu2,yu2,zu2 = POS2

    # Mobility part
    V_i = np.sqrt(Vh*Vh+Vv*Vv)*1000/3600 #m/s
    P_1 = 88.63
    P_0 = 84.14
    U_tip = 120
    v_0 = 4.03
    d_0 = 0.6
    s = 0.05
    rho = 1.225
    A = 0.503
    #d_MM = 0.1 # drones are static and the movement of each simulation step is 0.1m

    P_i = P_0*(1+3*V_i*V_i/(U_tip*U_tip))+P_1*pow((np.sqrt(1+V_i*V_i*V_i*V_i/(4*v_0*v_0*v_0*v_0))-V_i*V_i/(2*v_0*v_0)),0.5)+0.5*d_0*rho*s*A*V_i*V_i*V_i
    #d_MM = np.sqrt((xu1-xu2)*(xu1-xu2)+(yu1-yu2)*(yu1-yu2)+(zu1-zu2)*(zu1-zu2))
    E_MM = tau*P_i
    #E_MM = d_MM/V_i*P_i
    return E_MM


def uav_computation_energy(k=1e-28,C=1e3,phi=1e9,sw=639104,n_client=1):
    # cite art-118: Zeng, Tengchan, et al. "Federated learning in the sky: Joint power allocation and scheduling with UAV swarms." ICC 2020-2020 IEEE International Conference on Communications (ICC). IEEE, 2020.
    
    return n_client*(k*C*phi*phi*sw)


class Agent_vehicle:
	def __init__(self,ind=1,r_edge=''):
		self.index = ind
		self.name = "agent_"+str(self.index)
		self.acc_history = deque([0, 0], maxlen=2)
		self.lane_his = deque([0, 0], maxlen=2)
		self.pos = (0, 0)
		self.curr_edge = ''
		self.curr_lane = -1
		self.speed = 0
		self.target_speed = 0
		self.acc = 0
		self.angle = 0
		self.nb_lc = 0
		self.collision = False
		self.total_collision = 0
		self.collision_step = []
		self.collision_num = []
		self.action_his = []
		self.lc_flag = False
		self.w_eff = 1
		self.w_comf = 1
		self.w_safe = 1
		self.r = 0
		self.free = 0
		self.action_local=0
		self.action_gobal=0
		self.action = 0
		self.risk_edge = r_edge
        
        # urgent lc request for global control
		self.t_ULCR=10 # 10s to change lane
		self.thre_changemode=5
		self.nb_neigh = 0
        
		self.risky_time = 0
		self.block_time = 0
		self.rl_flag = False
		self.am_flag = False
		self.nv_segment = np.zeros(4)

        
        
	def update_params(self):
		# initialize params
		self.pos = traci.vehicle.getPosition(self.name)
		self.curr_edge = traci.vehicle.getLaneID(self.name)
		if self.curr_edge == '':
			#assert self.collision
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
		
        
	def compute_jerk(self):
		return self.acc_history[1] - self.acc_history[0]



        

class SumoEnv(gym.Env):
	def __init__(self,nb_agent=8):
		self.agents = []
		self.agent_number=nb_agent
		self.radius = 637
		self.maxall_speed = 0
		self.gui = False
		self.numVehicles = 0
		self.lane_ids = []
		self.curr_step = 0
		self.total_step = 0
		self.done = False
		self.nv_segment = np.zeros(4)
		self.global_count = 0
		self.risk_edge = ''

		self.model = ""
		self.his_a = []

        # for ordinary vehicles
		self.vid_list = []
		self.state_dict_left = {}
		self.state_dict_right = {}
        
		self.risky_edge_x = 0 # start point of the risky lane, length of 200m
		self.risky_lane_index = 0
        
		self.r_comf_norm = 100
		self.r_eff_norm = 30
		self.r_safe_norm = 70
        

	def start(self, gui=True, numVehicles=150, network_conf="networks/user_highway/circles.sumocfg", network_xml='networks/user_highway/circless.net.xml', thresh=3, tulcr=10):
		#print("start()")
		self.gui = gui
		self.numVehicles = numVehicles
		self.network_conf = network_conf
		self.net = sumolib.net.readNet(network_xml)
		
		self.curr_step = 0
		self.state_dict_left = {}
		self.state_dict_right = {}
		self.done = False
		self.nv_segment = np.zeros(4)

        # urgent lc request for global control
		#self.t_ULCR=tulcr
		self.global_count = 0
		self.r_comf_norm = 100
		self.r_eff_norm = 30
		self.r_safe_norm = 70
        
        # NGSIM LC model for ordinary vehicles
		self.load_path = "E:/hjl_simulations/xgboost_rl.model"
		self.model = joblib.load(self.load_path)


        # for ordinary vehicles
		self.vid_list = []
		self.state_dict_left = {}
		self.state_dict_right = {}
        
		self.risky_edge_x = 0 # start point of the risky lane, length of 200m
		self.risky_lane_index = 0
        
        
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
        
		for i in range(self.agent_number):
			self.agents.append(Agent_vehicle(ind=i+1,r_edge=self.risk_edge))
        
	# Do some random step to distribute the vehicles
		for i in range(numVehicles):
			if i%10 == 0:
				ambulance_id = int(i/10)
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

		# Do some random step to distribute the vehicles
		for step in range(self.numVehicles*10):
			traci.simulationStep()

		
        # add ego vehicle
		for i in range(self.agent_number):
			traci.vehicle.add("agent_"+str(i+1), routeID='route_0', typeID='ego') 
			traci.vehicle.setLaneChangeMode("agent_"+str(i+1), 0)
			# Setting up useful parameters
			self.agents[i].curr_lane = traci.vehicle.getLaneIndex(self.agents[i].name)
			self.agents[i].update_params()
			for step in range(10):
				traci.simulationStep()
                
		self.maxall_speed = traci.vehicle.getAllowedSpeed(self.agents[i].name)
		self.update_nv_segment()
		self.vid_list = traci.vehicle.getIDList()
        
        
	'''def compute_reputation(self,r_local,r_tot,c_local,c_tot):
		#if client.t_send<0:
			#rep = -10 # straggler
			#else:
		rep = self.w_r*r_local/r_tot + self.w_c*c_local/c_tot
			#if rep>1: # rep should <=1
			#print("Error value!")
		return rep'''
       
   
	def update_params(self):
		self.update_nv_segment()
		self.vid_list = traci.vehicle.getIDList()
		for i in range(self.agent_number):
			self.agents[i].update_params()
			self.agents[i].nv_segment = self.nv_segment
		

        
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
		state = []
		for i in range(self.agent_number):
			arr = []
			count_neigh = 0
			agent = self.agents[0]
			# self info x,y,speed,acc,lane_id
			info = self.get_vehicle_info(agent.name)
			if agent.curr_edge==self.risk_edge:
				arr.append(1)
			else:
				arr.append(0)
			for i in info:
				arr.append(i)
                
			# neighbors' IDs
			l = traci.vehicle.getLeader(agent.name)
			ll = traci.vehicle.getLeftLeaders(agent.name)
			if len(ll)!=0:
				ll = ll[0]
			rl = traci.vehicle.getRightLeaders(agent.name)
			if len(rl)!=0:
				rl = rl[0]
			f = traci.vehicle.getFollower(agent.name)
			lf = traci.vehicle.getLeftFollowers(agent.name)
			if len(lf)!=0:
				lf = lf[0]
			rf = traci.vehicle.getRightFollowers(agent.name)
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
	
			state_=np.array(list(map(float,arr)))
			#state.append(state_) 
			state.append([state_,count_neigh]) 
		return state # 1*48*n_agents


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


	def detect_collision(self):
		collisions = traci.simulation.getCollidingVehiclesIDList()
		collision_list = []
		for i in range(self.agent_number):
			agent = self.agents[i]
			if agent.name in collisions:
				agent.collision = True
				collision_list.append(True)
			agent.collision = False
			collision_list.append(False)
		return collision_list

		
	def compute_reward(self,agent):
			# Reward function is made of three elements:
			# - Comfort 
			 #- Efficiency
			# - Safety
		#agent = self.agents[0]
		# Comfort reward 
		#print(str(agent.name)+" compute reward")
		jerk = agent.compute_jerk()

		R_comf = - jerk**2/self.r_comf_norm

		#Efficiency reward
		# Speed
		R_speed = -np.abs(agent.speed - agent.target_speed)
		# Penalty for changing lane
		if agent.lc_flag:
			R_change = -1
		else:
			R_change = 0
		agent.lc_flag = False
		# Rewards Parameters
		R_eff = (R_speed + R_change)/self.r_eff_norm

        # Safety Reward
		# Vehicle density
		if agent.pos[0]>0:
			if agent.pos[1]>0:
				R_den = -agent.nv_segment[0]
			elif agent.pos[1]<0:
				R_den = -agent.nv_segment[3]
		elif agent.pos[0]<0:
			if agent.pos[1]>0:
				R_den = -agent.nv_segment[1]
			elif agent.pos[1]<0:
				R_den = -agent.nv_segment[2]
		
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
		if agent.collision:
			agent.r = -100
		else:
			agent.r = 0
			agent.free += 1
			if agent.free >= 10:
				agent.r = 1
				agent.free = 0
		R_colli = agent.r

        # Penalize driving on risky lane
		if agent.rl_flag==True:
			R_risk = -1
		else:
			R_risk = 0

        # Penalize blocking an emergency vehicle
		if agent.am_flag==True:
			R_block = -1
		else:
			R_block = 0

		R_safe = (R_colli + R_den + R_risk + R_block)/self.r_safe_norm

		agent.rl_flag = False
		agent.am_flag = False

		# total reward
		R_tot =  agent.w_comf*R_comf + agent.w_eff*R_eff + agent.w_safe*R_safe
		#print("R_comf: ",R_comf," R_eff: ",R_eff," R_safe: ",R_safe)
		return [R_tot, R_comf, R_eff, R_safe]

	def total_energy(self,p1_uav=[0,0,150],p2_uav=[0,0,150],p_veh=[200,300,0],STEP_PER_EPOCH=64, G_UPDATE_FREQ=10,N_AGENT=4):
		#p1_uav=[0,0,150],p2_uav=[0,0,150],p_veh=[200,300,0],EPOCH=1000, UPDATE_FREQ=10):
		# mobility for duration UPDATE_FREQ*0.4s
		e_m = uav_mobility_energy(tau=G_UPDATE_FREQ*STEP_PER_EPOCH*0.4)
		# e_comp
		e_comp = uav_computation_energy(n_client=N_AGENT)
		e_comm = 0
		for i in range(self.agent_number):
			agent = self.agents[i]
			# e_d2v 
			e_comm += get_d2v_communication_energy(P_uav=p1_uav,P_veh=agent.pos)
			#e_d2d = get_d2d_communication_energy(POS_uav1=p1_uav,POS_uav2=p_uav2,POS_uav3=p_uav3,POS_uav4=p_uav4)
		e_total = e_m + e_comp + e_comm
		return e_m, e_comp , e_comm,e_total
    
    
    # Queuing Delay 
    # between u_i and v_j
    # at time t1 and t2
	def total_delay(self,POS1_uav=[0,0,150],POS1_veh=[500,300],VEH_NUM=150,N_AGENT=4):
		delay_total = []
		VEH_NUM = self.numVehicles
		for i in range(self.agent_number):
			agent = self.agents[i]
			xu1, yu1, zu1 = POS1_uav
			xv1, yv1 = agent.pos
			zv1 = 0

			dd2v_ij = np.sqrt((xu1-xv1)*(xu1-xv1)+(yu1-yv1)*(yu1-yv1)+(zu1-zv1)*(zu1-zv1))
			#dv2d_ij = np.sqrt((xu2-xv2)*(xu2-xv2)+(yu2-yv2)*(yu2-yv2)+(zu2-zv2)*(zu2-zv2))
			dv2d_ij = dd2v_ij
			C1_ij = d2v_achivabel_datarate(POS_uav=POS1_uav,POS_veh=POS1_veh)
			#C2_ij = d2v_achivabel_datarate(POS_uav=POS2_uav,POS_veh=POS2_veh)
			C2_ij = C1_ij
			Wv2d_ij = dv2d_ij/C1_ij
			Wd2v_ij = dd2v_ij/C2_ij
            
            # high priority safety message 
			lambda_1 = 2.5
            #mu = 250
			E_B1 = 4.763*pow(10,-7)
			E_B12 = 2.269*pow(10,-13)
            #E_R1 = 3
			rho_1 = lambda_1*E_B1
			E_S_safe = rho_1/(2*(1-rho_1))*E_B12/E_B1 + E_B1
        
            # low priority vehicle state information
            # length = 
			E_B2 = 2.977*pow(10,-8)
			E_B22 = 0
            #E_R2 = 3
			lambda_2 = N_AGENT 
			rho_2 = lambda_2*E_B2
			sum_rho = rho_1*E_B12/(2*E_B1)+rho_2*E_B22/(2*E_B2)
			E_S_state = 1/((1-(rho_1+rho_2))*(1-rho_1))*sum_rho + E_B2

			#d_agent = Wd2v_ij + Wv2d_ij + E_S_safe + E_S_state
			d_agent = Wd2v_ij + E_S_safe + E_S_state
			delay_total.append(d_agent)
		return delay_total


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

        #agent = self.agents[0]
		if self.curr_step%20==0:
        # update ordinary vehicles' position by NGSIM model
			for v in self.vid_list:
				if v in self.state_dict_right and traci.vehicle.getTypeID(v)!='agent':
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

		step_state = []
		for i in range(self.agent_number):
			agent = self.agents[i]
			action = a[i]
			agent.action = action
    
			collision = False
    
            # update ego vehicle's lane
			if agent.lc_flag:
				agent.action_his.append(action)
			collision = False
			# Action legend : 0 stay, 1 change to right, 2 change to left
			if action != 0:
			# if at edge lane and action is change lane, then collision = True
				if action == 1:
					if agent.curr_lane == 1:
						traci.vehicle.changeLane(agent.name, 0, 2)
					elif agent.curr_lane == 2:
						traci.vehicle.changeLane(agent.name, 1, 2)
					elif agent.curr_lane == 0:
						collision = True
				if action == 2:
					if agent.curr_lane == 0:
						traci.vehicle.changeLane(agent.name, 1,2)
					elif agent.curr_lane == 1:
						traci.vehicle.changeLane(agent.name, 2, 2)
					elif agent.curr_lane == 2:
						collision = True
    
			# Sim step
			traci.simulationStep()
    
			# Check road collision
			collision_list = self.detect_collision()
			collision = agent.collision + collision
    
			reward = self.compute_reward(agent)
    
			# Update agent params 
			self.update_params()
			# State 
			state_ = self.get_state()
			next_state, nb_neigh = state_[i]
			agent.nb_neigh = nb_neigh
			# Update curr state
			self.curr_step += 1
			self.total_step += 1
			done = collision
            
			step_state.append([next_state, reward, done, collision])
            
		return step_state
		
	def render(self, mode='human', close=False):
		pass

	def reset(self, gui=False, numVehicles=150,  network_conf="networks/user_highway/circles.sumocfg", network_xml='networks/user_highway/circless.net.xml', thr=3, ttulcr=10):
		self.start(gui, numVehicles, thresh = thr, tulcr=ttulcr)
		return self.get_state()


	def close(self):
		#print("env.closed")
		#print("Total step:", self.total_step)
		#print("Total vehicle number:", traci.vehicle.getIDCount())
		traci.close()