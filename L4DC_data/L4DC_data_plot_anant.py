# from constants_lqg import *
# from errors_doc import *
# from constants_enkf_common import *

import numpy as np
import numpy.linalg as LA
import scipy.linalg as SLA
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['font.size'] = 14

import os
import csv

colour_blue = (0, 0.4470, 0.7410)
colour_orange = (0.8500, 0.3250, 0.0980)
colour_green = (0.4660, 0.6740, 0.1880)
colour_red = (0.6350, 0.0780, 0.1840)
colour_purple = (0.4940, 0.1840, 0.5560)
colours = [colour_blue,colour_orange,colour_green,colour_red,colour_purple,"orchid","grey"]

markers = ['.','^','*']

# #### 1/N plots

# markers = ['.','^','*']

# #NSIM = int(NSIM)
# NVEC = np.ceil(NVEC)

# Np = 3 # number of particles in dimensions plot
# Nind = np.array([0,7,NSIM-1]) # index of particles in NVEC -- 100, 600, 1000


# #DVEC = np.array([2,4,20,50,80])
# #D = 5

# DVEC = np.array([10,40,50,80])
# D=np.size(DVEC)


# mseSN2 = np.zeros((3,D,NSIM))
# mseSN = np.zeros((3,D,NSIM))
# mseSN_mod = np.zeros((3,D,NSIM))
# mseSN_mod2 = np.zeros((3,D,NSIM))
# mseSN_inf2 = np.zeros((3,D,NSIM))
# mseSN_inf = np.zeros((3,D,NSIM))
# mseSN_inv = np.zeros((3,D,NSIM))
# mseSN_inv2 = np.zeros((3,D,NSIM))
# mseSN_inv_inf = np.zeros((3,D,NSIM))
# mseSN_inv_inf2 = np.zeros((3,D,NSIM))
# mseSN_inv_mod = np.zeros((3,D,NSIM))
# mseSN_inv_mod2 = np.zeros((3,D,NSIM))
# # mseSNi = np.zeros((D,NSIM))
# # eSNi = np.zeros((D,NSIM))
# # mseKN = np.zeros((D,NSIM))
# # mseKNi = np.zeros((D,NSIM))
# # eKNi = np.zeros((D,NSIM))

# prob_type = ["LQG","LEQGP","LEQGN"]
# # dict_of_vars = {"Sct":mseSN,"Sct2":mseSN2,"Sctinv": mseSN_inv,"Sctinv2":mseSN_inv2}
# dict_of_vars = ({"Sct":mseSN,"Sct2":mseSN2,"Sctmod":mseSN_mod,"Sctmod2":mseSN_mod2,
#     "Sctinf":mseSN_inf,"Sctinf2":mseSN_inf2,
#     "Sctinvinf":mseSN_inv_inf,"Sctinvinf2":mseSN_inv_inf2,"Sctinv":mseSN_inv,"Sctinv2":mseSN_inv2,
#     "Sctinvmod":mseSN_inv_mod,"Sctinvmod2":mseSN_inv_mod2,})
# plotvars = [mseSN_mod2,mseSN_inv_mod2]

# prob_count = 0
# for prob in prob_type:

# 	dim_count = 0

# 	for i in DVEC:
		
# 		filename_base = (os.path.dirname(os.path.realpath(__file__)) + '/Results/EnKF-' + prob + '/')
# 		# filename_d = filename_base + str(i) + 'D/'	
# 		# filename = filename_d + prob + str(i) + varname

# 		for var in dict_of_vars:	
		
# 			filename = filename_base + str(i) + 'D/' + prob + str(i) + var + '.npy'
# 			# a = np.load(filename)
# 			# print(a)
# 			dict_of_vars[var][prob_count,dim_count,:] = np.load(filename)
# 			# mseSN[prob_count,dim_count,:] = np.load(filename)

# 		dim_count = dim_count + 1

# 	prob_count = prob_count + 1
		
# print(mseSN[0,:,:] - mseSN[1,:,:])

# # q7 = np.power(NVEC,-0.5)
# q7 = np.power(NVEC,-1)



# f2, axes = plt.subplots(nrows=1,ncols=2)
# var_count = 0
# for var in plotvars:
# 	ax = axes[var_count]
# 	ax.plot(NVEC,q7,'--k')
# 	var_count+=1
# for i in range(0,3):	
# 	dim_count = 0
# 	for j in DVEC:
# 		var_count = 0
# 		for var in plotvars:
# 			ax = axes[var_count]
# 			ax.plot(NVEC,var[i,dim_count,:],label=r"$d=$ " + str(DVEC[dim_count]),
# 				c=colours[dim_count],marker=markers[i])		
# 			ax.set_xscale('log')
# 			ax.set_yscale('log')
# 			ax.grid(which="both",lw=0.5)
# 			ax.set_xlabel("Number of particles")
# 			ax.set_ylabel("MSE")
# 			var_count+=1
# 		dim_count+=1
# ax = axes[0]
# ax.legend(ncol=9,loc=(0.0,1.02))
# ax.text(0.4, 0.1, r"$O(1/N)$", size="large", horizontalalignment='center', 
#      verticalalignment='center', transform=ax.transAxes)
# plt.show()





##### SMD energy plots

# T = 5
# STEP = 0.02
# ITER = int(T/STEP)
# TIMEVEC = np.arange(1,ITER+2)
# DVEC = [4,10,20,30,40,50,80]
# D = np.size(DVEC)
# a_energy = np.zeros((3,D,ITER+1))
# a_cost = np.zeros((3,D,ITER+1))
# a_energy_ana = np.zeros((3,D,ITER+1))
# a_cost_ana = np.zeros((3,D,ITER+1))
# dict_of_vars = ({"cost":a_cost,"energy":a_energy,"costana":a_cost,"energyana":a_energy})

# prob_type = ["LQG","LEQGP","LEQGN"]

# plotvars = [[a_cost,a_cost_ana],[a_energy,a_energy_ana]]

# for prob_count, prob in enumerate(prob_type):	

# 	for dim_count, i in enumerate(DVEC):
		
# 		filename_base = (os.path.dirname(os.path.realpath(__file__)) + '/Results/Stable-SMD-' + prob + '/')
# 		# filename_d = filename_base + str(i) + 'D/'	
# 		# filename = filename_d + prob + str(i) + varname

# 		for var in dict_of_vars:	
		
# 			filename = filename_base + str(i) + 'D/' + prob + str(i) + var + '.npy'
# 			# a = np.load(filename)
# 			# print(a)
# 			dict_of_vars[var][prob_count,dim_count,:] = np.load(filename)
# 			# mseSN[prob_count,dim_count,:] = np.load(filename)


# f2, axes = plt.subplots(nrows=2,ncols=3)

# for probcount, prob in enumerate(prob_type):
# 	for row in range(0,2):
# 		ax = axes[row,probcount]
# 		for dimcount, dim in enumerate(DVEC):
# 			for var in range(0,2):
# 				ax.plot(TIMEVEC,plotvars[row][var][probcount,dimcount,:],label=r"$d=$ " + str(DVEC[dimcount]),
# 				c=colours[dimcount],marker = markers[probcount])
# plt.show()	

##### randomsys plots

# def dimx(data):
# 	return data[:,3]
# def Np(data):
# 	return data[:,1]
# def kz(data):
# 	return data[:,2]
# def ecost(data,ind):
# 	return data[ind,7]
# def egain(data,ind):
# 	return data[ind,6]
# def stime(data,ind):
# 	return data[ind,8]

# a_kz_cost = np.zeros((5,3))
# a_kz_gain = np.zeros((4,3))
# a_kz_cost[:,0] = np.array([5000,100,50,25,15])
# a_kz_gain[:,0] = np.array([10,8,7,6])
# data = np.genfromtxt('neurips2024.csv', delimiter=',')

# list_of_vars = [a_kz_cost,a_kz_gain]
# list_of_funcs = [ecost,egain]

# for varcount,varname in enumerate(list_of_vars):
# 	for i in range(varname.shape[0]):
# 		ind = np.where((Np(data) == varname[i,0]) & 
# 			(kz(data) == 1) & (dimx(data) == 3))
# 		varname[i,1] = list_of_funcs[varcount](data,ind[0][0])
# 	print(varname)

# list_of_ylabels = ["rel. error in cost","rel. error in gain"]
# f1, axes = plt.subplots(nrows=1, ncols=2)
# # f1.set_size_inches((18,4.5))
# for varcount,varname in enumerate(list_of_vars):
# 	ax = axes[varcount]
# 	ax.plot(varname[:,1],varname[:,2])
# 	ax.set_ylabel(list_of_ylabels[varcount])
# ax.set_xlabel("simulation time (s)")

# from constants_randomsys import *
# # NVEC = np.array([100,500,1000,5000])
# NSIM = np.size(NVEC)
# DIMX = 3
# a_cov_cost = np.zeros((2,2,NSIM)) # LQG or LEQG, fixed or random, NVEC, 
# m_mean_cost = np.zeros((2,2,NSIM)) 

# a_cov_term = np.zeros((2,2,NSIM))
# m_mean_term = np.zeros((2,2,NSIM))

# a_cov_l1 = np.zeros((2,2,NSIM))
# m_mean_l1 = np.zeros((2,2,NSIM))

# a_cov_time = np.zeros((2,2,NSIM))
# m_mean_time = np.zeros((2,2,NSIM))

# meanvars = {"m_mean_cost":m_mean_cost,"m_mean_term":m_mean_term,"m_mean_l1":m_mean_l1,"m_mean_time":m_mean_time}
# covvars = ["a_cov_cost","a_cov_term","a_cov_l1","a_cov_time"]



# NSYS = 1
# m_error_gain1 = np.zeros((ITER+1,NSIM,NSYS,NAVG))
# m_simtime1 = np.zeros((ITER+1,NSIM,NSYS,NAVG))
# m_cost = np.zeros((NSIM,NSYS,NAVG))
# v_cost_ana = np.zeros((NSIM,NSYS))
# dict_of_vars = {"cost":m_cost, "errorK":m_error_gain1, "simtime":m_simtime1, "costana" : v_cost_ana}
# readvars = ["m_cost","m_error_gain","m_simtime", "v_cost_ana"]

# # for the fixed models
# for i in range(0,2): # LQG or LEQG
# 	filename = basefilenames[i][1] + filename2[i]
# 	for varc,varname in enumerate(dict_of_vars):
# 		filenamenpy = filename + varname + ".npy"
# 		varname2 = readvars[varc]
# 		if varc==0:
# 			m_cost = np.load(filenamenpy)
# 		elif varc == 1:
# 			m_error_gain1 = np.load(filenamenpy)
# 		elif varc == 2:
# 			m_simtime1 = np.load(filenamenpy)
# 		elif varc == 3:
# 			v_cost_ana = np.load(filenamenpy)
		
# 	m_cost_error = 0*m_cost
# 	m_error_l1 = np.sum(m_error_gain1,axis=0)*STEP/T
# 	m_error_gain = m_error_gain1[-1,:,:,:]
# 	m_simtime = m_simtime1[-1,:,:,:]
# 	array_of_vars = ["m_cost_error","m_error_gain","m_error_l1","m_simtime"]
# 	dict_of_vars1 = ({"m_mean_cost":m_cost_error, "m_mean_term":m_error_gain, 
# 		"m_mean_l1":m_error_l1, "m_mean_time" : m_simtime})
# 	for nsim in range(NSIM):
# 		m_cost_error[nsim,0,:] = np.abs(m_cost[nsim,0,:] - v_cost_ana[nsim,0])/v_cost_ana[nsim,0]
# 		# print(i,v_cost_ana)
# 		for statvar in meanvars:
# 			meanvars[statvar][i,0,nsim] = np.mean(dict_of_vars1[statvar][nsim,0,:])
# 			# globals()[meanvars[varcount]][i,0,nsim] = np.mean(globals()[array_of_vars[varcount]][nsim,0,:])
# 			# print(meanvars[varcount][i,0,nsim])
# 			# globals()[covvars[varcount][i,0,nsim]] = np.std(array_of_vars[varcount][nsim,0,:])

# # for the random models

# NSYS = 5
# m_error_gain1 = np.zeros((ITER+1,NSIM,NSYS,NAVG))
# m_simtime1 = np.zeros((ITER+1,NSIM,NSYS,NAVG))
# m_cost = np.zeros((NSIM,NSYS,NAVG))
# v_cost_ana = np.zeros((NSIM,NSYS))
# dict_of_vars = {"cost":m_cost, "errorK":m_error_gain1, "simtime":m_simtime1, "costana" : v_cost_ana}

# for i in range(0,2): # LQG or LEQG
# 	filename = basefilenames[i][1] + filename2[i]
# 	for varname in dict_of_vars:
# 		filenamenpy = filename + varname + ".npy"
# 		dict_of_vars[varname][...] = np.load(filenamenpy)
# 		# globals()[readvars[varc]] = np.load(filenamenpy)
# 		# exec(f"{varname} = np.load(filenamenpy)")
# 	m_cost_error = 0*m_cost
# 	m_error_l1 = np.sum(m_error_gain1,axis=0)*STEP/T
# 	m_error_gain = m_error_gain1[-1,:,:,:]
# 	m_simtime = m_simtime1[-1,:,:,:]
# 	array_of_vars = ["m_cost_error","m_error_gain","m_error_l1","m_simtime"]
# 	dict_of_vars1 = ({"m_mean_cost":m_cost_error, "m_mean_term":m_error_gain, 
# 		"m_mean_l1":m_error_l1, "m_mean_time" : m_simtime})
# 	for nsim in range(NSIM):
# 		for nsys in range(NSYS):
# 			m_cost_error[nsim,nsys,:] = np.abs(m_cost[nsim,nsys,:] - v_cost_ana[nsim,nsys])/v_cost_ana[nsim,nsys]
# 		for statvar in meanvars:
# 			meanvars[statvar][i,1,nsim] = np.mean(dict_of_vars1[statvar][nsim,:,:])
# 			# print(i,nsim,varcount,array_of_vars[varcount])
# 			# print(globals()[array_of_vars[varcount]].shape)
# 			# globals()[meanvars[varcount]][i,1,nsim] = np.mean(globals()[array_of_vars[varcount]][nsim,:,:])
# 			# print(meanvars[varcount][i,1,nsim])
# 			# globals()[covvars[varcount][i,1,nsim]] = np.std(array_of_vars[varcount][nsim,:,:])


### comparison with br and kz, simtime plots
T=1 # [1, 10]
DIMX = 3
NSIM = 10 
a_stats = np.zeros((2,2,NSIM,4,2)) # lqg or leqg, deterministic or random, 
#  no of particles, cost terminal_error_gain(lqg) l1_gain(lqeg) time, 
# mean or std

basefilenames = ([[(os.path.dirname(os.path.realpath(__file__)) + '/Results/BRSys-LQG/'),
	(os.path.dirname(os.path.realpath(__file__)) + '/Results/RandomSys-LQG/')],
	[(os.path.dirname(os.path.realpath(__file__)) + '/Results/KZSys-LEQGP/'),
	(os.path.dirname(os.path.realpath(__file__)) + '/Results/RandomSys-LEQGP/')]])
filename2 = [str(3) + "D/"+ 'LQG' + str(DIMX) + "D",str(3) + "D/" + 'LEQGP' + str(DIMX) + "D"]

for i in range(0,2): # LQG or LEQG
	for j in range(0,1): # deterministic or random
		filename = basefilenames[i][j] + filename2[i]  + str(T) + "T"
		filenamenpy = filename + "mean" + ".npy"
		print(filenamenpy)
		meanfile = np.load(filenamenpy)
		# print(meanfile.shape)
		a_stats[i,j,:,:,0] = meanfile
		filenamenpy = filename + "cov" + ".npy"
		print(filenamenpy)
		covfile = np.load(filenamenpy)
		a_stats[i,j,:,:,1] = covfile


# np.save('3dstats10lyap.npy',a_stats)

f2, axes = plt.subplots(nrows=2,ncols=2) # kz and br
for i in range(0,2): # lqg or leqg
	ax = axes[i,0]
	ax.plot(a_stats[i,0,:,3,0],a_stats[i,0,:,0,0],color='black')
	# ax.semilogy(a_stats[i,1,:,0,0],a_stats[i,1,:,3,0],color='blue')
	# ax.errorbar(a_stats[i,0,:,0,0],a_stats[i,0,:,3,0],a_stats[i,0,:,3,1])
	# ax.set_yscale('log')
	ax.set_ylabel("cost")
ax = axes[0,1]
ax.errorbar(a_stats[0,0,:,3,0],a_stats[0,0,:,1,0],color='black')
# ax.semilogy(a_stats[0,1,:,1,0],a_stats[0,1,:,3,0],color='blue')
ax.set_ylabel("BR term")
ax = axes[1,1]
ax.errorbar(a_stats[1,0,:,3,0],a_stats[1,0,:,2,0],color='black')
# ax.semilogy(a_stats[1,1,:,2,0],a_stats[1,1,:,3,0],color='blue')
ax.set_ylabel("KZ l1")
# plt.show()

m_stats_mean = a_stats[0,0,:,:,0]

dataname = ["cost","gain term","gain l1"]
f1, axs= plt.subplots(3,1)
for i in range(0,3):
	ax = axs[i]
	ax.errorbar(m_stats_mean[:,3],m_stats_mean[:,i])
	ax.set_ylabel(dataname[i])
	ax.set_xlabel("sim time")
f1.suptitle("T = " + str(T) + " " + "LQG")
plt.show()


# # #### smd simtime plots

# DVEC= [20]
# D = np.max(DVEC)
# NSIM = 4
# a_stats_smd = np.zeros((D,NSIM,3,4,2)) # dimension, no of particles, lqg or leqg,  cost term l1 time, mean or std

# basefilename = (os.path.dirname(os.path.realpath(__file__)) + '/Results/SMDTime/')
# # filename2 = [str(dim) + "D/"+ str(dim) + "D"]

# for dimcount, dim in enumerate(DVEC):
# 	filename = basefilename + str(dim) + "D/"+ str(dim) + "D" + str(int(T)) + "T"
# 	filenamenpy = filename + "mean" + ".npy"
# 	meanfile = np.load(filenamenpy)
# 	a_stats_smd[dimcount,:,:,:,0] = meanfile
# 	filenamenpy = filename + "cov" + ".npy"
# 	covfile = np.load(filenamenpy)
# 	a_stats_smd[dimcount,:,:,:,1] = covfile


# f2, axes = plt.subplots(nrows=3,ncols=3) # kz and br
# for i in range(0,3): # lqg or leqg(+,-)
# 	for j in range(0,3): # cost term l
# 		ax = axes[i,j]
# 		for dimcount, dim in enumerate(DVEC):		
# 			ax.semilogy(a_stats_smd[dimcount,:,i,j,0],a_stats_smd[dimcount,:,i,3,0],color='black',label=str(dim)+"D",marker="*")
		
# # 	ax.semilogy(a_stats[i,1,:,0,0],a_stats[i,1,:,3,0],color='blue')
# # 	# ax.errorbar(a_stats[i,0,:,0,0],a_stats[i,0,:,3,0],a_stats[i,0,:,3,1])
# # 	ax.set_yscale('log')
# # ax = axes[0,1]
# # ax.semilogy(a_stats[0,0,:,1,0],a_stats[0,0,:,3,0],color='black')
# # ax.semilogy(a_stats[0,1,:,1,0],a_stats[0,1,:,3,0],color='blue')
# # ax = axes[1,1]
# # ax.semilogy(a_stats[1,0,:,2,0],a_stats[1,0,:,3,0],color='black')
# # ax.semilogy(a_stats[1,1,:,2,0],a_stats[1,1,:,3,0],color='blue')
# plt.show()

# np.save('smdstats.npy',a_stats_smd)

