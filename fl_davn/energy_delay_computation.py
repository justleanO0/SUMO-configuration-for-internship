# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:35:37 2023

@author: Jialin
"""

# compute energy and delay

import numpy as np
from tqdm import tqdm
import math
from scipy import spatial

UPDATE_FREQ = 4 # target network update frequency
MODEL_SAVE_FREQ = 1000
GLOBAL_UPDATE_FREQ = 75
REPLAY_MEMORY_START_SIZE = 33
steps_per_epoch=300
epochs=50
NB_AGENT = 6
NB_VEHICLE = 150

# energy consumption
total_energy_drone = 0
e=0
energy_his = []
energy_his.append(e)
model_bit = 639104
client_tot_delay = np.zeros(NB_AGENT)
total_step = 0
delay_his = []
for i in range(NB_AGENT):
  delay_his.append([])

'''
from keras.models import load_model
network = load_model("DQN_models/global_model/wglobal_init.h5")
g_weights = network.get_weights()
model_size = 0
for item in g_weights:
    model_size += item.size
model_bit = model_size*32 # model size in bit
'''
model_bit = 639104
# 639104


def cosine_distance(weight1,weight2):
    
    cos_sim1 = 1 - spatial.distance.cosine(np.array(weight1[0]).reshape(weight1[0].size,), np.array(weight2[0]).reshape(weight2[0].size,))
    cos_sim2 = 1 - spatial.distance.cosine(np.array(weight1[2]).reshape(weight1[2].size,), np.array(weight2[2]).reshape(weight2[2].size,))
    cos_sim3 = 1 - spatial.distance.cosine(np.array(weight1[4]).reshape(weight1[4].size,), np.array(weight2[4]).reshape(weight2[4].size,))
    cos_sim4 = 1 - spatial.distance.cosine(np.array(weight1[8]).reshape(weight1[8].size,), np.array(weight2[8]).reshape(weight2[8].size,))

    c = cos_sim1+cos_sim2+cos_sim3+cos_sim4

    return c/4

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


def uav_computation_energy(k=1e-28,C=1e3,phi=1e9,sw=model_bit,n_client=4):
    # cite art-118: Zeng, Tengchan, et al. "Federated learning in the sky: Joint power allocation and scheduling with UAV swarms." ICC 2020-2020 IEEE International Conference on Communications (ICC). IEEE, 2020.
    
    return n_client*(k*C*phi*phi*sw)


def total_energy(p1_uav=[0,0,150],p2_uav=[0,0,150],p_veh=[200,300,0],EPOCH=1000, G_UPDATE_FREQ=GLOBAL_UPDATE_FREQ,N_AGENT=NB_AGENT):
    # mobility for duration UPDATE_FREQ*0.4s
    e_total = uav_mobility_energy(tau=G_UPDATE_FREQ*0.4)
    # e_comp
    e_total += uav_computation_energy(n_client= N_AGENT)
    for i in range(N_AGENT):
        apos = [-600.6918013036047, 174.440581435308]
        # e_d2v 
        e_total += get_d2v_communication_energy(P_uav=p1_uav,P_veh=apos)
        #e_d2d = get_d2d_communication_energy(POS_uav1=p1_uav,POS_uav2=p_uav2,POS_uav3=p_uav3,POS_uav4=p_uav4)

    return e_total


def total_delay(POS1_uav=[0,0,150],POS1_veh=[500,300],VEH_NUM=NB_VEHICLE,N_AGENT=NB_AGENT):
    delay_total = []
    #VEH_NUM =  numVehicles
    for i in range(N_AGENT):
        xu1, yu1, zu1 = POS1_uav
        xv1, yv1 = [-600.6918013036047, 174.440581435308]
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
        lambda_1 = 0.2*VEH_NUM
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
        lambda_2 = 2.5
        rho_2 = lambda_2*E_B2
        sum_rho = rho_1*E_B12/(2*E_B1)+rho_2*E_B22/(2*E_B2)
        E_S_state = 1/((1-(rho_1+rho_2))*(1-rho_1))*sum_rho + E_B2

        #d_agent = Wd2v_ij + Wv2d_ij + E_S_safe + E_S_state
        d_agent = Wd2v_ij + E_S_safe + E_S_state
        delay_total.append(d_agent)
    return delay_total


total_step = 0
for epoch in tqdm(range(epochs)):
    #print(epoch)
    for step in range(steps_per_epoch):
        total_step += 1
        if step % GLOBAL_UPDATE_FREQ == 0 and step > REPLAY_MEMORY_START_SIZE:
            # energy consumption
            e = total_energy(EPOCH=steps_per_epoch, G_UPDATE_FREQ=GLOBAL_UPDATE_FREQ)
            total_energy_drone += e
            energy_his.append(e)
            
            
            # delay
            delays = total_delay()
            for i in range(NB_AGENT):
                client_tot_delay[i] += delays[i]
                delay_his[i].append(delays[i])


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(energy_his,label='drone')
ax.set_xlabel('epoch')
ax.set_ylabel('energy')
ax.set_title('Energy consumption of the drone')
ax.legend()
plt.show()

fig, ax = plt.subplots()
for i in range(NB_AGENT):
    ax.plot(delay_his[i],label='agent'+str(i))
ax.set_xlabel('epoch')
ax.set_ylabel('Delay')
ax.set_title('Delay of agents')
ax.legend()
plt.show()


print("Vehicle number is ",NB_VEHICLE)
print("Total delay of client:")
for i in range(NB_AGENT):
    print(client_tot_delay[i])
print("Total energy consumption of the drone:", total_energy_drone)
print("Global update frequency:", GLOBAL_UPDATE_FREQ)
print("Step per epoch:", steps_per_epoch)
print("Number of agents:",NB_AGENT)