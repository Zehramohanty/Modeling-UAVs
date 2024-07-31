import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from matplotlib import animation
from matplotlib import rc

import pandas as pd


Nm = 15                                        
No = 25                                         
Nt = 100                                        

Ai = 1                                          
Cdi = 0.25                                      
mi = 10                                        
va = [-0.2, 0.2, 0.5]                          
ra = 1.225                                      
Fpi_scalar = 200                                        

dt = 0.2                                        
tf = 60                                         


agent_sight = 5                                 
crash_range = 2     


C = 6                                           
P = 6                                           
S = 20                                          
G = 100                                         
minLam = 0                                     
maxLam = 2                                     
num_params = 15

w1 = 70                                         
w2 = 10                                         
w3 = 20  


# Domain Parameters
xmax = 150                                     
ymax = 150                                      
zmax = 60                                      

locx = 100                                     
locy = 100                                      
locz = 10                                       



x_obs = (locx - (-locx))*np.random.rand(No) + (-locx)
y_obs = (locy - (-locy))*np.random.rand(No) + (-locy)
z_obs = (locz - (-locz))*np.random.rand(No) + (-locz)
obs = np.array([x_obs, y_obs, z_obs]).T


x_tar = (locx - (-locx))*np.random.rand(Nt) + (-locx)
y_tar = (locy - (-locy))*np.random.rand(Nt) + (-locy)
z_tar = (locz - (-locz))*np.random.rand(Nt) + (-locz)
tar = np.array([x_tar, y_tar, z_tar]).T



x_pos = (xmax - 0.05*xmax)*np.ones(Nm)
y_pos = np.linspace(-ymax + 0.05*ymax, ymax - 0.05*ymax, Nm)
z_pos = np.zeros(Nm)
pos = np.array([x_pos, y_pos, z_pos]).T



vel = np.zeros([Nm, 3])

pos_initial = pos                              # Initial agent positions
vel_initial = vel                              # Initial agent velocities
tar_initial = tar  



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], color='r')
ax.scatter(tar[:, 0], tar[:, 1], tar[:, 2], color='g')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='k')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.view_init(elev=70., azim=40)
ax.legend(['Obstacles', 'Targets', 'Agents'])



def droneSim(n_m, n_o, n_t, w1, w2, w3, LAM, dt, tf, posM, velM, posTar, posObs):
    
    Nt_initial = n_t  
    Nm_initial = n_m 

   
    W_mt = LAM[0]
    W_mo = LAM[1]
    W_mm = LAM[2]
    w_t1 = LAM[3]
    w_t2 = LAM[4]
    w_o1 = LAM[5]
    w_o2 = LAM[6]
    w_m1 = LAM[7]
    w_m2 = LAM[8]
    a1 = LAM[9]
    a2 = LAM[10]
    b1 = LAM[11]
    b2 = LAM[12]
    c1 = LAM[13]
    c2 = LAM[14]

    time = int(np.ceil(tf / dt))                       
    count = 0                                         
    posData = []                                        
    tarData = []   
    
    posData.append(posM)
    tarData.append(posTar)

    i = 0
    while i < time:
        
        mtDiff = np.zeros((posM.shape[0], posTar.shape[0], 3))               
        mmDiff = np.zeros((posM.shape[0], posM.shape[0], 3))                 
        moDiff = np.zeros((posM.shape[0], posObs.shape[0], 3))               

        
        mtDist = np.zeros((posM.shape[0], posTar.shape[0]))                 
        mmDist = np.zeros((posM.shape[0], posM.shape[0]))                   
        moDist = np.zeros((posM.shape[0], posObs.shape[0]))                  

        # check each agent
        for j in range(posM.shape[0]):
            
            mtDiff[j, :, :] = (posTar - posM[j])                            
            mmDiff[j, :, :] = (posM - posM[j])                              
            moDiff[j, :, :] = (posObs - posM[j])                            
            mmDiff[j, j, :] = np.nan

            mtDist[j, :] = np.linalg.norm(posM[j] - posTar, ord=2, axis=1)  
            mmDist[j, :] = np.linalg.norm(posM[j] - posM, ord=2, axis=1)    
            moDist[j, :] = np.linalg.norm(posM[j] - posObs, ord=2, axis=1)  
            mmDist[j, j] = np.nan
            

        mtHit = np.where(mtDist < agent_sight)                
        moHit = np.where(moDist < crash_range)                
        mmHit = np.where(mmDist < crash_range)                
        # check for lost agents
        
        xLost = np.where(np.abs(posM[:, 0]) > xmax)
        yLost = np.where(np.abs(posM[:, 1]) > ymax)
        zLost = np.where(np.abs(posM[:, 2]) > zmax)

        xy = np.append(xLost[0], yLost[0], axis = 0)
        xyz = np.append(xy, zLost[0], axis = 0) 
        mLost = np.unique(xyz)
       
        tarMapped = np.unique(mtHit[1])                       

        a = np.append(mmHit[0], moHit[0], axis = 0)
        b = np.append(a, mLost)
        mCrash = np.unique(b)

        # remove crashed agents
        posM = np.delete(posM, (mCrash), axis=0)
        velM = np.delete(velM, (mCrash), axis=0)
        n_t = n_t - len(tarMapped)
        n_m = n_m - len(mCrash)

        mtDist = np.delete(mtDist, (mCrash), axis=0)     
        mtDiff = np.delete(mtDiff, (mCrash), axis=0)
        mmDist = np.delete(mmDist, (mCrash), axis=0)     
        mmDist = np.delete(mmDist, (mCrash), axis=1)     
        mmDiff = np.delete(mmDiff, (mCrash), axis=0)
        mmDiff = np.delete(mmDiff, (mCrash), axis=1)
        moDist = np.delete(moDist, (mCrash), axis=0)     
        moDiff = np.delete(moDiff, (mCrash), axis=0)

    
        posTar = np.delete(posTar, (tarMapped), axis=0)  
        mtDist = np.delete(mtDist, (tarMapped), axis=1)  
        mtDiff = np.delete(mtDiff, (tarMapped), axis=1)

    
        if (posTar.size == 0 ) or (posM.size == 0):
            break

    
        n_MT = mtDiff / mtDist[:, :, np.newaxis]                
        n_MO = moDiff / moDist[:, :, np.newaxis]                 
        n_MM = mmDiff / mmDist[:, :, np.newaxis]                 

      
        one = (w_t1 * np.exp(-a1 * mtDist) - w_t2 * np.exp(-a2 * mtDist)) 
        nMT_place = one[:, :, np.newaxis] * n_MT                             

        two = (w_o1 * np.exp(-b1 * moDist) - w_o2 * np.exp(-b2 * moDist))  
        nMO_place = two[:, :, np.newaxis] * n_MO                            

        three = (w_m1 * np.exp(-c1 * mmDist) - w_m2 * np.exp(-c2 * mmDist)) 
        nMM_place = three[:, :, np.newaxis] * n_MM                             

       
        N_mt = np.nansum(nMT_place, axis=1)                                     
        N_mo = np.nansum(nMO_place, axis=1)                                     
        N_mm = np.nansum(nMM_place, axis=1)                                     

        
        N_total = (W_mt * N_mt) + (W_mo * N_mo) + (W_mm * N_mm)                            

        
        pr = N_total / np.linalg.norm(N_total, 2, axis=1)[:, np.newaxis]
        prop_force = Fpi_scalar * pr                                                  

      
        e = np.linalg.norm(va - velM, 2, axis=1)[:, np.newaxis]
        drag_force = 1. / 2. * ra * Cdi * Ai * e * (va - velM)

        total_force = prop_force + drag_force

        velM = velM + dt * total_force / mi
        
        posM = posM + dt * velM

        
        posData.append(posM)
        tarData.append(posTar)   
        count += 1
        
            
        i += 1

        
        
    
    M_star = n_t / Nt_initial
    T_star = (count * dt) / tf
    L_star = ((Nm_initial - n_m) / Nm_initial)
    PI = (w1 * M_star) + (w2 * T_star) + (w3 * L_star)


    
    return (PI, posData, tarData, count, M_star, T_star, L_star)




def myGA(S, G, P, C, minLam, maxLam, num_params, Nm, No, Nt, w1, w2, w3, dt, tf, pos, vel):
   
   
    
    cost = np.zeros(G)                                      
    parent = np.zeros(G)                                        
    pop = np.zeros(G)                                         
    Pi = np.zeros(S)                                          
    M_star = np.zeros(S)                                       
    T_star = np.zeros(S)                                         
    L_star = np.zeros(S)                                       
    M_cost = np.zeros(G)                              
    T_cost = np.zeros(G)                               
    L_cost = np.zeros(G)                                  
    M_parent = np.zeros(G)                           
    T_parent = np.zeros(G)                            
    L_parent = np.zeros(G)                       
    M_pop = np.zeros(G)                          
    T_pop = np.zeros(G)                             
    L_pop = np.zeros(G)                             

    
    lam_param = (maxLam - minLam) * np.random.rand(num_params, S) + minLam

    l = 0

    for i in range(G):  
        
        for j in range(l, S):
            
            Pi[j], _, _, _, M_star[j], T_star[j], L_star[j] = droneSim(Nm, No, Nt, w1, w2, w3, lam_param[:, j], dt, tf, pos, vel, tar, obs)

        
        f = np.argsort(Pi)
        Pi = np.sort(Pi)
        lam_param = lam_param[:, f]
        M_star = M_star[f]
        T_star = T_star[f]
        L_star = L_star[f]

        
        phi = np.random.rand(num_params, C)
        q = range(0, C, 2)
        r = range(1, C, 2)

        lam_param = np.hstack((lam_param[:, 0:P], phi[:, q] * lam_param[:, q] + (1 - phi[:, q]) * lam_param[:, r],
                         phi[:, r] * lam_param[:, r] + (1 - phi[:, r]) * lam_param[:, q],
                         (maxLam - minLam) * np.random.rand(num_params, S - P - C) + minLam))

        
        cost[i] = Pi[0]
        parent[i] = np.mean(Pi[0:P])
        pop[i] = np.mean(Pi)

        M_cost[i] = M_star[0]
        T_cost[i] = T_star[0]
        L_cost[i] = L_star[0]

        M_parent[i] = np.mean(M_star[0:P])
        T_parent[i] = np.mean(T_star[0:P])
        L_parent[i] = np.mean(L_star[0:P])

        M_pop[i] = np.mean(M_star)
        T_pop[i] = np.mean(T_star)
        L_pop[i] = np.mean(L_star)

     
        l = P
        
        print(cost[i])

    return (lam_param, Pi, cost, parent, pop, M_cost, T_cost, L_cost, M_parent, T_parent, L_parent,
            M_pop, T_pop, L_pop)



#Calling the GA Function

lam_param, Pi, cost, parent, pop, M_cost, T_cost, L_cost, M_parent, T_parent, L_parent, M_pop, T_pop, L_pop = myGA(
    S, G, P, C, minLam, maxLam, num_params, Nm, No, Nt, w1, w2, w3, dt, tf, pos, vel)


_, posTot, tarTot, c, _, _, _ = droneSim(Nm, No, Nt, w1, w2, w3, lam_param[:, 0], dt, tf, pos_initial, vel_initial, tar_initial, obs)




fig1 = plt.figure(figsize=(12, 5))
plt.semilogy(range(0, G), cost)
plt.semilogy(range(0, G), parent)
plt.semilogy(range(0, G), pop)
plt.xlabel('Generation')
plt.ylabel('Min or Ave cost')
plt.title('Cost Evolution')
plt.legend(['Best Cost', 'Average Parent Cost', 'Average Cost'])
plt.show()

fig2 = plt.figure(figsize=(12, 5))
plt.plot(range(0, G), M_cost)
plt.plot(range(0, G), T_cost)
plt.plot(range(0, G), L_cost)
plt.xlabel('Generation')
plt.ylabel('Cost Parameter Value')
plt.title('Best Cost Parameter Evolution')
plt.legend(['M*', 'T*', 'L*'])
plt.show()

fig3 = plt.figure(figsize=(12, 5))
plt.plot(range(0, G), M_parent)
plt.plot(range(0, G), T_parent)
plt.plot(range(0, G), L_parent)
plt.xlabel('Generation')
plt.ylabel('Cost Parameter Value')
plt.title('Parent Average Cost Parameter Evolution')
plt.legend(['M*', 'T*', 'L*'])
plt.show()

fig4 = plt.figure(figsize=(12, 5))
plt.plot(range(0, G), M_pop)
plt.plot(range(0, G), T_pop)
plt.plot(range(0, G), L_pop)
plt.xlabel('Generation')
plt.ylabel('Cost Parameter Value')
plt.title('Population Average Cost Parameter Evolution')
plt.legend(['M*', 'T*', 'L*'])
plt.show()



plotting = np.linspace(0, len(posTot) - 1, 5, dtype = int)
plotting



pos = posTot[plotting[0]]
tar = tarTot[plotting[0]]
    
#plot 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], color='r')
ax.scatter(tar[:, 0], tar[:, 1], tar[:, 2], color='g')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='k')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.view_init(elev=70., azim=40)
ax.legend(['Obstacles', 'Targets', 'Agents']) 
plt.title('Simulation at first time step')



pos = posTot[plotting[1]]
tar = tarTot[plotting[1]]
    
#plot 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], color='r')
ax.scatter(tar[:, 0], tar[:, 1], tar[:, 2], color='g')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='k')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.view_init(elev=70., azim=40)
ax.legend(['Obstacles', 'Targets', 'Agents'])  
plt.title('Simulation at second time step')



pos = posTot[plotting[2]]
tar = tarTot[plotting[2]]
    
#plot 3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], color='r')
ax.scatter(tar[:, 0], tar[:, 1], tar[:, 2], color='g')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='k')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.view_init(elev=70., azim=40)
ax.legend(['Obstacles', 'Targets', 'Agents'])  
plt.title('Simulation at third time step')




pos = posTot[plotting[3]]
tar = tarTot[plotting[3]]
    
#plot 4
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], color='r')
ax.scatter(tar[:, 0], tar[:, 1], tar[:, 2], color='g')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='k')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.view_init(elev=70., azim=40)
ax.legend(['Obstacles', 'Targets', 'Agents'])  
plt.title('Simulation at fourth time step')



pos = posTot[plotting[4]]
tar = tarTot[plotting[4]]
    
#plot 5
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], color='r')
ax.scatter(tar[:, 0], tar[:, 1], tar[:, 2], color='g')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='k')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.view_init(elev=70., azim=40)
ax.legend(['Obstacles', 'Targets', 'Agents'])  
plt.title('Simulation at fifth time step')




Wmt = lam_param[:, 0][0]
Wmo = lam_param[:, 0][1]
Wmm = lam_param[:, 0][2]
wt1 = lam_param[:, 0][3]
wt2 = lam_param[:, 0][4]
wo1 = lam_param[:, 0][5]
wo2 = lam_param[:, 0][6]
wm1 = lam_param[:, 0][7]
wm2 = lam_param[:, 0][8]
a1 = lam_param[:, 0][9]
a2 = lam_param[:, 0][10]
b1 = lam_param[:, 0][11]
b2 = lam_param[:, 0][12]
c1 = lam_param[:, 0][13]
c2 = lam_param[:, 0][14]




PI_1 = droneSim(Nm, No, Nt, w1, w2, w3, lam_param[:, 0], dt, tf, pos_initial, vel_initial, tar_initial, obs)[0]
PI_2 = droneSim(Nm, No, Nt, w1, w2, w3, lam_param[:, 1], dt, tf, pos_initial, vel_initial, tar_initial, obs)[0]
PI_3 = droneSim(Nm, No, Nt, w1, w2, w3, lam_param[:, 2], dt, tf, pos_initial, vel_initial, tar_initial, obs)[0]
PI_4 = droneSim(Nm, No, Nt, w1, w2, w3, lam_param[:, 3], dt, tf, pos_initial, vel_initial, tar_initial, obs)[0]




values = [[Wmt, Wmo, Wmm, wt1, wt2, wo1, wo2, wm1, wm2, a1, a2, b1, b2, c1, c2, PI_1], 
           [lam_param[:, 1][0], lam_param[:, 1][1], lam_param[:, 1][2], lam_param[:, 1][3], lam_param[:, 1][4], lam_param[:, 1][5], lam_param[:, 1][6], lam_param[:, 1][7], lam_param[:, 1][8], lam_param[:, 1][9], lam_param[:, 1][10], lam_param[:, 1][11], lam_param[:, 1][12], lam_param[:, 1][13], lam_param[:, 1][14], PI_2], 
           [lam_param[:, 2][0], lam_param[:, 2][1], lam_param[:, 2][2], lam_param[:, 2][3], lam_param[:, 2][4], lam_param[:, 2][5], lam_param[:, 2][6], lam_param[:, 2][7], lam_param[:, 2][8], lam_param[:, 2][9], lam_param[:, 2][10], lam_param[:, 2][11], lam_param[:, 2][12], lam_param[:, 2][13], lam_param[:, 2][14], PI_3], 
           [lam_param[:, 3][0], lam_param[:, 3][1], lam_param[:, 3][2], lam_param[:, 3][3], lam_param[:, 3][4], lam_param[:, 3][5], lam_param[:, 3][6], lam_param[:, 3][7], lam_param[:, 3][8], lam_param[:, 3][9], lam_param[:, 3][10], lam_param[:, 3][11], lam_param[:, 3][12], lam_param[:, 3][13], lam_param[:, 3][14], PI_4]]

df = pd.DataFrame(values,
                  columns=['Wmt', 'Wmo', 'Wmm', 'wt1', 'wt2', 'wo1', 'wo2', 'wm1', 'wm2', 'a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'PI'],
                  index=['1', '2', '3', '4'])
df


