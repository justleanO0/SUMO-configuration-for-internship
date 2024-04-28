# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:45:26 2023

@author: Jialin
"""
import numpy as np
import math
# parameters computation
def in_uav_range0(POS_veh=[400,600,0], R=637):
    xv_i,yv_i,zv_i = POS_veh
    print("vehicle position: ",POS_veh)
    r = R
    r_unit = int(np.sqrt(1/2)*r)+1
    if xv_i in range(-r_unit,r_unit) and yv_i in range(r_unit,r):
        uav_id = 1
    elif xv_i in range(r_unit,r) and yv_i in range(-r_unit,r_unit):
        uav_id = 2
    elif xv_i in range(-r_unit,r_unit) and yv_i in range(-r,-r_unit):
        uav_id = 3
    elif xv_i in range(-r,-r_unit) and yv_i in range(-r_unit,r_unit):
        uav_id = 4
    else:
        uav_id = 0
        print("wrong segment!")
    return uav_id

def in_uav_range(VEH_LANE="edge13", R=637):
    uid = 0
    if VEH_LANE in ["edge12","edge13","edge0","edge1"]:
        uid = 1
    if VEH_LANE in ["edge2","edge3","edge4","edge5"]:
        uid = 2
    if VEH_LANE in ["edge6","edge7","edge8"]:
        uid = 3
    if VEH_LANE in ["edge9","edge10","edge11"]:
        uid = 4
    return uid

# Communication energy
def d2v_achivabel_datarate(POS_uav=[400,600,150],POS_veh=[200,300,0],
                             FC=2.4*math.pow(10,9), ETA_LoSDB=1,ETA_NLoSDB=20,
                             C=3*math.pow(10,8),N_0DB=-174, P_TRANS=0.28, BW=100*10^6):
    xu_i,yu_i,zu_i = POS_uav
    xv_i,yv_i,zv_i = POS_veh
    
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

def get_d2v_communication_energy(P_uav=[400,600,150],P_veh=[200,300,0],SI=512, P_TRANS=0.28):
    C_ij = d2v_achivabel_datarate(POS_uav=P_uav,POS_veh=P_veh)
    S_i = SI #bytes
    p_trans_i = P_TRANS
    E_ij = S_i*p_trans_i/C_ij
    return E_ij


def get_d2d_communication_energy(POS_uav1=[400,600,150],POS_uav2=[200,300,100],
                                 POS_uav3=[455,650,170],POS_uav4=[280,370,150],
                             FC=2.4*math.pow(10,9), ETA_LoSDB=1, C=3*math.pow(10,8),
                             N_0DB=-174, P_TRANS=0.28, BW=100*10^6, SI=512):
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
    
def uav_mobility_energy(POS1=[400,600,150],POS2=[200,300,100], Vh=30, Vv=10):
    
    xu1,yu1,zu1 = POS1
    xu2,yu2,zu2 = POS2
    
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

    P_i = P_0*(1+3*V_i*V_i/(U_tip*U_tip))+P_1*pow((np.sqrt(1+V_i*V_i*V_i*V_i/(4*v_0*v_0*v_0*v_0))-V_i*V_i/(2*v_0*v_0)),0.5)+0.5*d_0*rho*s*A*V_i*V_i*V_i
    d_MM = np.sqrt((xu1-xu2)*(xu1-xu2)+(yu1-yu2)*(yu1-yu2)+(zu1-zu2)*(zu1-zu2))
    E_MM = d_MM/V_i*P_i
    return E_MM

def total_energy(p1_uav=[400,600,150],p2_uav=[200,400,130],p_veh=[200,300,0],p_uav2=[400,300,100],p_uav3=[450,350,155],p_uav4=[460,380,170]):
    e_d2v = get_d2v_communication_energy(P_uav=p1_uav,P_veh=p_veh)
    e_d2d = get_d2d_communication_energy(POS_uav1=p1_uav,POS_uav2=p_uav2,POS_uav3=p_uav3,POS_uav4=p_uav4)
    e_m = uav_mobility_energy(POS1=p1_uav,POS2=p2_uav, Vh=30, Vv=10)
    e_total = e_d2v + e_d2d + e_m
    return e_total


# Queuing Delay 
# between u_i and v_j
# at time t1 and t2
def total_delay(POS1_uav=[400,600,150],POS2_uav=[500,400,130],
                          POS1_veh=[500,300,0],POS2_veh=[400,600,0],
                          VEH_NUM=150):
    xu1, yu1, zu1 = POS1_uav
    xu2, yu2, zu2 = POS2_uav
    xv1, yv1, zv1 = POS1_veh
    xv2, yv2, zv2 = POS2_veh
    dd2v_ij = np.sqrt((xu1-xv1)*(xu1-xv1)+(yu1-yv1)*(yu1-yv1)+(zu1-zv1)*(zu1-zv1))
    dv2d_ij = np.sqrt((xu2-xv2)*(xu2-xv2)+(yu2-yv2)*(yu2-yv2)+(zu2-zv2)*(zu2-zv2))

    C1_ij = d2v_achivabel_datarate(POS_uav=POS1_uav,POS_veh=POS1_veh)
    C2_ij = d2v_achivabel_datarate(POS_uav=POS2_uav,POS_veh=POS2_veh)
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
    
    total_delay = Wd2v_ij + Wv2d_ij + E_S_safe + E_S_state

    return total_delay



