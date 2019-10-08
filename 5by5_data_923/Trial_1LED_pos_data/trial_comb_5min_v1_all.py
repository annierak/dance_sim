# Small circle configuration  numbered 
# 5 minute trials
# 6 trials
# 1, 15 sec
# 1 LED center
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import math
import pandas as pd
#
from custom_trial_v1_numbered import find_nearest
from custom_trial_v1_numbered import cus_atan
from custom_trial_v1_numbered import rev_from_angle
from custom_trial_v1_numbered import adjust_spines
from custom_trial_v1_numbered import rev_adjust_1F
#from custom_trial_v1_numbered import r01_finder_1F


no = 24
af1_d = 2

No = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]


r0_1F = np.transpose(np.array([[0.0 for x in range(len(No))] for y in range(6)]))
r1_1F = np.transpose(np.array([[0.0 for x in range(len(No))] for y in range(6)]))

t_af1_all = np.transpose(np.array([[0.0 for x in range(6*len(No))] for y in range(50)]))
RE_af1_all = np.transpose(np.array([[0.0 for x in range(6*len(No))] for y in range(50)]))

no_tr = 6 # number of trials
dur = 5 # duration of each period: 5 mins

#
order = 71 
dis = 30 
wid = 200
##########################################
for kk in range(len(No)):
	Nam = 'Gr5a_1LED_trial_'+str(No[kk]) # Name of the directory

	################ loading the time 
	t = np.loadtxt(Nam+'/time.txt')
	t_sh = t[0]
	t = t - t[0]
	t = np.divide(t,60) # to minutes

	######
	tt = pd.read_csv(Nam+'/data.txt', header=None, usecols=[2])
	tt_v = tt.values

	tt_f = np.zeros(len(tt_v))

	for ii in range(len(tt_v)-1):
		tt_f[ii] = float(tt_v[ii+1][0])


	t_adj = np.divide(tt_f, 1e9)
	t_adj = t_adj - t_sh
	t_adj = np.divide(t_adj, 60)
	#############################################################

	xx = np.loadtxt(Nam+'/pos_x.txt')
	yy = np.loadtxt(Nam+'/pos_y.txt')

	xx = xx - 180
	yy = yy - 120
	xx = np.divide(xx,46*0.95)
	yy = np.divide(yy,46*0.95)


	th = 0.1
	for ii in range(len(xx)-2):
		if abs(xx[ii+1] - xx[ii]) > th:
			xx[ii+1] = np.mean([xx[ii], xx[ii+2]])
		#	
		if abs(yy[ii+1] - yy[ii]) > th:
			yy[ii+1] = np.mean([yy[ii], yy[ii+2]])

	plt.figure(1)
	plt.plot(xx,yy,'.r')

	# LED pin
	pin = pd.read_csv(Nam+'/data.txt', header=None, usecols=[6])
	pin_v = pin.values

	pin_f = np.zeros(len(pin_v))

	for ii in range(len(pin_v)-1):
		pin_f[ii] = float(pin_v[ii+1][0])
	# LED value
	val = pd.read_csv(Nam+'/data.txt', header=None, usecols=[7])
	val_v = val.values

	val_f = np.zeros(len(val_v))

	for ii in range(len(val_v)-1):
		val_f[ii] = float(val_v[ii+1][0])

	#
	LL = np.zeros(len(val_f))
	TT = np.zeros(len(val_f))

	for ii in range(len(val_f)):
		if val_f[ii] == 100 and pin_f[ii] == 10:
			LL[ii] = 1
			TT[ii] = t_adj[ii]

	LL_nz = LL[np.nonzero(LL)]
	TT_nz = TT[np.nonzero(TT)]

	if LL_nz[0] == 1:
		ind_10 = np.where(pin_f == 10)
		ind_L = ind_10

	fig = plt.figure(2)
	plt.plot(TT_nz,LL_nz,'.g')
	plt.plot(np.ones(5)*5, np.linspace(0.8,1.2,5),'r')
	for ii in range(no_tr-1):
		plt.plot(np.ones(5)*(ii+1)*10, np.linspace(0.8,1.2,5),'b')
		plt.plot(np.ones(5)*(ii+1)*10+5, np.linspace(0.8,1.2,5),'r')
	plt.ylim([0.9,1.1])	
	fig.savefig('Sum_plots_numbered' + '/LED_'+str(No[kk]))

	#plt.show()
	#exit()
	#############################################################
	ind_be_e = find_nearest(t, dur)
	ind_be = range(0, ind_be_e, 1)
	ind_du_e = find_nearest(t, dur+1*dur+0*dur)
	ind_af_e = find_nearest(t, dur+1*dur+1*dur)
	ind_af1_e = find_nearest(t, dur+1*dur+2)

	ind_be = range(0, ind_be_e, 1)
	ind_du = range(ind_be_e  + 1, ind_du_e, 1)
	ind_af = range(ind_du_e  + 1, ind_af_e, 1)
	ind_af1 = range(ind_du_e  + 1, ind_af1_e, 1)

	t_1 = t[ind_be[0]:ind_af[-1]]
	xx_1 = xx[ind_be[0]:ind_af[-1]]
	yy_1 = yy[ind_be[0]:ind_af[-1]]

	mx = min(xx_1)
	Mx = max(xx_1)
	my = min(yy_1)
	My = max(yy_1)

	# adjusting the center
	xx_1 = xx_1 - 0.5*(mx + Mx)
	yy_1 = yy_1 - 0.5*(my + My)

	plt.figure(11)
	plt.plot(xx_1,yy_1,'.b')

	t_be = t_1[ind_be[0]:ind_be[-1]]
	xx_be = xx_1[ind_be[0]:ind_be[-1]]
	yy_be = yy_1[ind_be[0]:ind_be[-1]]

	t_du = t_1[ind_du[0]:ind_du[-1]] 
	xx_du = xx_1[ind_du[0]:ind_du[-1]]
	yy_du = yy_1[ind_du[0]:ind_du[-1]]

	t_af = t_1[ind_af[0]:ind_af[-1]] 
	xx_af = xx_1[ind_af[0]:ind_af[-1]]
	yy_af = yy_1[ind_af[0]:ind_af[-1]]

	t_af1 = t_1[ind_af1[0]:ind_af1[-1]] 
	xx_af1 = xx_1[ind_af1[0]:ind_af1[-1]]
	yy_af1 = yy_1[ind_af1[0]:ind_af1[-1]]



	fig = plt.figure(301)
	plt.subplot(211)
	plt.plot(t_be,xx_be,'.b')
	plt.subplot(212)
	plt.plot(t_be,yy_be,'.b')
	#
	plt.subplot(211)
	plt.plot(t_du,xx_du,'.k')
	plt.subplot(212)
	plt.plot(t_du,yy_du,'.k')
	#
	plt.subplot(211)
	plt.plot(t_af1,xx_af1,'.r')
	plt.subplot(212)
	plt.plot(t_af1,yy_af1,'.r')
	fig.savefig('Sum_plots_numbered/xy_plots' + '/xy_'+str(No[kk])+'_'+str(1))

	e_be = np.zeros(len(t_be))
	e_du = np.zeros(len(t_du))
	e_duL5 = np.zeros(len(t_du))
	e_af1 = np.zeros(len(t_af1))
	e_af = np.zeros(len(t_af))
	e_af1L5 = np.zeros(len(t_af1))

	aa = 0.5
	for ii in range(len(t_be)):
		e_be[ii] = math.exp(0.5*aa*t_be[ii])
	for ii in range(len(t_du)):
		e_du[ii] = math.exp(0.2*0.5*aa*(t_du[ii] - dur))
	for ii in range(len(t_du)):
		e_duL5[ii] = math.exp(10*0.2*0.5*aa*(t_du[ii] - dur))	
	for ii in range(len(t_af1)):
		e_af1[ii] = math.exp(aa*t_af1[ii])
	for ii in range(len(t_af1)):
		e_af1L5[ii] = math.exp(10*0.2*0.5*aa*(t_af1[ii] - dur))	

	t_last = TT_nz[find_nearest(TT_nz, dur+dur)]
	e_t_last = 	math.exp(10*0.2*0.5*aa*(t_last - dur))	

	theta_be = np.zeros(len(xx_be))
	theta_du = np.zeros(len(xx_du))
	theta_af = np.zeros(len(xx_af))
	theta_af1 = np.zeros(len(xx_af1))

	#####  before
	for ii in range(len(xx_be)):
		theta_be[ii] = cus_atan(yy_be[ii], xx_be[ii])

	###  during
	for ii in range(len(xx_du)):
		theta_du[ii] = cus_atan(yy_du[ii], xx_du[ii])

	###  after 
	for ii in range(len(xx_af)):
		theta_af[ii] = cus_atan(yy_af[ii], xx_af[ii])

	###  after 1
	for ii in range(len(xx_af1)):
		theta_af1[ii] = cus_atan(yy_af1[ii], xx_af1[ii])

	r_be = np.ones(len(t_be))
	r_du = np.ones(len(t_du))
	r_af = np.ones(len(t_af))
	r_af1 = np.ones(len(t_af1))

	##################################### save data
	np.savetxt('Annie_Data_trial_1food/t_1F_be_'+str(No[kk])+'_'+'1'+'.txt', t_be)
	np.savetxt('Annie_Data_trial_1food/t_1F_du_'+str(No[kk])+'_'+'1'+'.txt', t_du)
	np.savetxt('Annie_Data_trial_1food/t_1F_af_'+str(No[kk])+'_'+'1'+'.txt', t_af)
	np.savetxt('Annie_Data_trial_1food/t_1F_af1_'+str(No[kk])+'_'+'1'+'.txt', t_af1)
	#
	np.savetxt('Annie_Data_trial_1food/theta_1F_be_'+str(No[kk])+'_'+'1'+'.txt', theta_be)
	np.savetxt('Annie_Data_trial_1food/theta_1F_du_'+str(No[kk])+'_'+'1'+'.txt', theta_du)
	np.savetxt('Annie_Data_trial_1food/theta_1F_af_'+str(No[kk])+'_'+'1'+'.txt', theta_af)
	np.savetxt('Annie_Data_trial_1food/theta_1F_af1_'+str(No[kk])+'_'+'1'+'.txt', theta_af1)
	#####################
	[t_du_co, RE_du_co, t_af1_co, RE_af1_co] = rev_from_angle(t_du, theta_du, t_af1, theta_af1, order, dis, wid)
	##################################### 
	print('@@@@@@@@@@@@')
	print(RE_af1_co*(180/np.pi))
	[t_af1_co, RE_af1_co] = rev_adjust_1F(No[kk], t_af1, theta_af1, t_du_co, RE_du_co, t_af1_co, RE_af1_co,1)
	r0_1F[kk,0] = RE_af1_co[0]
	r1_1F[kk,0] = RE_af1_co[1] - RE_af1_co[0]

	####### storing reversal values and time 
	if len(t_af1_co) < 50:
		t_af1_all[6*kk,0:len(t_af1_co)] = t_af1_co
		RE_af1_all[6*kk,0:len(RE_af1_co)] = RE_af1_co
	#
	#print('@@@@@@@@@@@@@@', t_af1_co)
	r_du_peaks = np.ones(len(t_du_co))
	r_af1_peaks = np.ones(len(t_af1_co))
	e_du_peaks = np.zeros(len(t_du_co))
	e_af1_peaks = np.zeros(len(t_af1_co))
	#
	for ii in range(len(t_du_co)):
		e_du_peaks[ii] = math.exp(10*0.2*0.5*aa*(t_du_co[ii] - dur))	
	for ii in range(len(t_af1_co)):
		e_af1_peaks[ii] = math.exp(aa*(t_af1_co[ii] - dur))
	#####################	

	fig = plt.figure(201+10*kk)
	fig.set_size_inches(12, 9)
	ax = fig.add_subplot(111, polar=True)
	ax.grid(False)
	plt.setp(ax.yaxis.get_ticklabels(), visible=False)
	#ax.plot(theta_du[7*int(len(theta_du)/8):] - np.pi/2, r_du[7*int(len(theta_du)/8):]*e_duL5[7*int(len(theta_du)/8):], 'b')
	#ax.plot(theta_af1[0:1*int(len(theta_af1)/2)] - np.pi/2, r_af1[0:1*int(len(theta_af1)/2)]*e_af1L5[0:1*int(len(theta_af1)/2)], 'r')
	ax.plot(theta_af1 - np.pi/2, r_af1*e_af1L5, 'r')
	ax.plot(theta_du - np.pi/2, r_du*e_duL5, 'b')
	#ax.plot(theta_af1 - np.pi/2, r_af1*e_af1L5, 'r')
	#
	ax.plot(0 - np.pi/2, 1*e_t_last, 'sk')
	#ax.plot(RE_du_co - np.pi/2, r_du_peaks*e_du_peaks, 'om')
	ax.plot(RE_af1_co - np.pi/2, r_af1_peaks*e_af1_peaks, 'oc')
	#
	#ax.plot(np.ones(10)*(0 - np.pi/2), np.linspace(0.3*min(r_du[7*int(len(theta_du)/8):]*e_duL5[7*int(len(theta_du)/8):]), 1.1*max(r_af1[0:1*int(len(theta_af1)/2)]*e_af1L5[0:1*int(len(theta_af1)/2)]),10), 'g')
	#ax.plot(np.ones(10)*(0 - np.pi/2), np.linspace(0.3*min(r_du*e_duL5), 1.1*max(r_af1[0:1*int(len(theta_af1)/2)]*e_af1L5[0:1*int(len(theta_af1)/2)]),10), 'g')
	ax.plot(np.ones(10)*(0 - np.pi/2), np.linspace(0.3*min(r_du*e_duL5), 1.1*max(r_af1*e_af1L5),10), 'g')
	fig.savefig('Sum_plots_numbered/polar_plots' + '/t1L5_'+str(No[kk])+'_'+str(1))

	###################
	sc1 = 2.4*(360/(40*np.pi))
	sc = sc1*(np.pi/180)

	cu_du = 400
	#
	fig = plt.figure(1201+10*kk)
	fig.set_size_inches(12, 9)
	#
	ax = fig.add_subplot(1, 1, 1)
	ax.spines['left'].set_position(('axes', 1.6))
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_position(('axes', 0.1))
	ax.spines['top'].set_color('none')
	#
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	plt.setp(ax.yaxis.get_ticklabels(), visible=False)
	ax.plot(t_du[cu_du:]-dur, theta_du[cu_du:]/sc,'k')
	ax.plot(t_af1-dur, theta_af1/sc,'k')
	plt.title(str(No[kk])+'_'+str(1))
	#
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	plt.ylabel('position (body length)', fontsize = 20)
	plt.xlabel('time (minutes)', fontsize = 20)
	adjust_spines(ax, ['left', 'bottom'])
	plt.ylim([-25,25])
	fig.savefig('Sum_plots_numbered/lin_plots' + '/h_all_'+str(No[kk])+'_'+str(1))
	#plt.show()
	#exit

	for jj in range(no_tr-1):
		print('#############')
		print(jj+1)
		st = dur+(jj+1)*dur+(jj+1)*dur
		du = st + dur
		af = du + dur
		af1 = du + 2

		#print(st, du, af, af1)

		ind_st = find_nearest(t, st)
		ind_du_e = find_nearest(t, du)
		ind_af_e = find_nearest(t, af)
		ind_af1_e = find_nearest(t, af1)

		ind_du = range(ind_st  + 1, ind_du_e, 1)
		ind_af = range(ind_du_e  + 1, ind_af_e, 1)
		ind_af1 = range(ind_du_e   + 1, ind_af1_e, 1)

		#print(ind_st)
		#print(ind_du_e)
		#print(ind_du[0])
		#print(t[ind_du[0]])
		#print(t[ind_af[0]])

		t_1 = t[ind_du[0]:ind_af[-1]] - st
		xx_1 = xx[ind_du[0]:ind_af[-1]]
		yy_1 = yy[ind_du[0]:ind_af[-1]]

		mx = min(xx_1)
		Mx = max(xx_1)
		my = min(yy_1)
		My = max(yy_1)

		# adjusting the center
		xx_1 = xx_1 - 0.5*(mx + Mx)
		yy_1 = yy_1 - 0.5*(my + My)

		ind_du = range(1, ind_du_e - ind_st, 1)
		ind_af = range(ind_du_e  + 1 - ind_st, ind_af_e - ind_st, 1)
		ind_af1 = range(ind_du_e   + 1 - ind_st, ind_af1_e - ind_st, 1)

		t_du = t_1[ind_du[0]:ind_du[-1]]
		xx_du = xx_1[ind_du[0]:ind_du[-1]]
		yy_du = yy_1[ind_du[0]:ind_du[-1]]

		t_af = t_1[ind_af[0]:ind_af[-1]]
		xx_af = xx_1[ind_af[0]:ind_af[-1]]
		yy_af = yy_1[ind_af[0]:ind_af[-1]]

		t_af1 = t_1[ind_af1[0]:ind_af1[-1]]
		xx_af1 = xx_1[ind_af1[0]:ind_af1[-1]]
		yy_af1 = yy_1[ind_af1[0]:ind_af1[-1]]


		fig = plt.figure(302+jj)
		plt.subplot(211)
		plt.plot(t_du,xx_du,'.k')
		plt.subplot(212)
		plt.plot(t_du,yy_du,'.k')
		#
		plt.subplot(211)
		plt.plot(t_af1,xx_af1,'.r')
		plt.subplot(212)
		plt.plot(t_af1,yy_af1,'.r')
		fig.savefig('Sum_plots_numbered/xy_plots' + '/xy_'+str(No[kk])+'_'+str(jj+2))

		e_be = np.zeros(len(t_be))
		e_du = np.zeros(len(t_du))
		e_duL5 = np.zeros(len(t_du))
		e_af1 = np.zeros(len(t_af1))
		e_af = np.zeros(len(t_af))
		e_af1L5 = np.zeros(len(t_af1))

		aa = 0.5
		for ii in range(len(t_be)):
			e_be[ii] = math.exp(0.5*aa*t_be[ii])
		for ii in range(len(t_du)):
			e_du[ii] = math.exp(0.2*0.5*aa*t_du[ii])
		for ii in range(len(t_du)):
			e_duL5[ii] = math.exp(10*0.2*0.5*aa*t_du[ii])	
		for ii in range(len(t_af1)):
			e_af1[ii] = math.exp(aa*t_af1[ii])
		for ii in range(len(t_af1)):
			e_af1L5[ii] = math.exp(10*0.2*0.5*aa*t_af1[ii])	

		t_last = TT_nz[find_nearest(TT_nz, 2*dur + 2*(jj+1)*dur)] - 2*(jj+1)*dur 
		#print('!!!!!!!!!!!!!!!!')
		#print(t_last)
		e_t_last = 	math.exp(10*0.2*0.5*aa*(t_last - dur))


		theta_be = np.zeros(len(xx_be))
		theta_du = np.zeros(len(xx_du))
		theta_af = np.zeros(len(xx_af))
		theta_af1 = np.zeros(len(xx_af1))

		#####  before
		for ii in range(len(xx_be)):
			theta_be[ii] = cus_atan(yy_be[ii], xx_be[ii])

		###  during
		for ii in range(len(xx_du)):
			theta_du[ii] = cus_atan(yy_du[ii], xx_du[ii])

		###  after 
		for ii in range(len(xx_af)):
				theta_af[ii] = cus_atan(yy_af[ii], xx_af[ii])

		###  after 1
		for ii in range(len(xx_af1)):
			theta_af1[ii] = cus_atan(yy_af1[ii], xx_af1[ii])

		r_be = np.ones(len(t_be))
		r_du = np.ones(len(t_du))
		r_af = np.ones(len(t_af))
		r_af1 = np.ones(len(t_af1))

		##################################### save data
		np.savetxt('Annie_Data_trial_1food/t_1F_be_'+str(No[kk])+'_'+str(jj+2)+'.txt', t_be)
		np.savetxt('Annie_Data_trial_1food/t_1F_du_'+str(No[kk])+'_'+str(jj+2)+'.txt', t_du)
		np.savetxt('Annie_Data_trial_1food/t_1F_af_'+str(No[kk])+'_'+str(jj+2)+'.txt', t_af)
		np.savetxt('Annie_Data_trial_1food/t_1F_af1_'+str(No[kk])+'_'+str(jj+2)+'.txt', t_af1)
		#
		np.savetxt('Annie_Data_trial_1food/theta_1F_be_'+str(No[kk])+'_'+str(jj+2)+'.txt', theta_be)
		np.savetxt('Annie_Data_trial_1food/theta_1F_du_'+str(No[kk])+'_'+str(jj+2)+'.txt', theta_du)
		np.savetxt('Annie_Data_trial_1food/theta_1F_af_'+str(No[kk])+'_'+str(jj+2)+'.txt', theta_af)
		np.savetxt('Annie_Data_trial_1food/theta_1F_af1_'+str(No[kk])+'_'+str(jj+2)+'.txt', theta_af1)
		###########################
		[t_du_co, RE_du_co, t_af1_co, RE_af1_co] = rev_from_angle(t_du, theta_du, t_af1, theta_af1, order, dis, wid)
		#
		[t_af1_co, RE_af1_co] = rev_adjust_1F(No[kk], t_af1, theta_af1, t_du_co, RE_du_co, t_af1_co, RE_af1_co,jj+2)
		r0_1F[kk,jj+1] = RE_af1_co[0]
		r1_1F[kk,jj+1] = RE_af1_co[1] - RE_af1_co[0]

		####### storing reversal values and time 
		if len(t_af1_co) < 50:
			t_af1_all[6*kk+jj+1,0:len(t_af1_co)] = t_af1_co
			RE_af1_all[6*kk+jj+1,0:len(RE_af1_co)] = RE_af1_co
		##################################### reversal adjustments
		#if No_1F[kk] == 1 and tr_1F[kk] == 1:
		#	t_af1_co = np.hstack([t_af1_co])
		#	RE_af1_co = np.hstack([RE_af1_co])


		##################################### end of reversal adjustments

		#print('@@@@@@@@@@@@@@', t_af1_co)
		r_du_peaks = np.ones(len(t_du_co))
		r_af1_peaks = np.ones(len(t_af1_co))
		e_du_peaks = np.zeros(len(t_du_co))
		e_af1_peaks = np.zeros(len(t_af1_co))
		#
		for ii in range(len(t_du_co)):
			e_du_peaks[ii] = math.exp(10*0.2*0.5*aa*(t_du_co[ii]))	
		for ii in range(len(t_af1_co)):
			e_af1_peaks[ii] = math.exp(aa*(t_af1_co[ii]))
		###########################
			
		fig = plt.figure(202+10*kk+jj)
		fig.set_size_inches(12, 9)
		ax = fig.add_subplot(111, polar=True)
		ax.grid(False)
		plt.setp(ax.yaxis.get_ticklabels(), visible=False)
		#ax.plot(theta_du[7*int(len(theta_du)/8):] - np.pi/2, r_du[7*int(len(theta_du)/8):]*e_duL5[7*int(len(theta_du)/8):], 'b')
		#ax.plot(theta_af1[0:1*int(len(theta_af1)/2)] - np.pi/2, r_af1[0:1*int(len(theta_af1)/2)]*e_af1L5[0:1*int(len(theta_af1)/2)], 'r')
		ax.plot(theta_af1 - np.pi/2, r_af1*e_af1L5, 'r')
		ax.plot(theta_du - np.pi/2, r_du*e_duL5, 'b')
		#ax.plot(theta_af1 - np.pi/2, r_af1*e_af1L5, 'r')
		#
		ax.plot(0 - np.pi/2, 1*e_t_last, 'sk')
		#ax.plot(RE_du_co - np.pi/2, r_du_peaks*e_du_peaks, 'om')
		ax.plot(RE_af1_co - np.pi/2, r_af1_peaks*e_af1_peaks, 'oc')
		#
		#ax.plot(np.ones(10)*(0 - np.pi/2), np.linspace(0.3*min(r_du[7*int(len(theta_du)/8):]*e_duL5[7*int(len(theta_du)/8):]), 1.1*max(r_af1[0:1*int(len(theta_af1)/2)]*e_af1L5[0:1*int(len(theta_af1)/2)]),10), 'g')
		#ax.plot(np.ones(10)*(0 - np.pi/2), np.linspace(0.3*min(r_du*e_duL5), 1.1*max(r_af1[0:1*int(len(theta_af1)/2)]*e_af1L5[0:1*int(len(theta_af1)/2)]),10), 'g')
		ax.plot(np.ones(10)*(0 - np.pi/2), np.linspace(0.3*min(r_du*e_duL5), 1.1*max(r_af1*e_af1L5),10), 'g')
		fig.savefig('Sum_plots_numbered/polar_plots' + '/t1L5_'+str(No[kk])+'_'+str(jj+2))
		
		#
		fig = plt.figure(1202+10*kk+jj)
		fig.set_size_inches(12, 9)
		#
		ax = fig.add_subplot(1, 1, 1)
		ax.spines['left'].set_position(('axes', 1.6))
		ax.spines['right'].set_color('none')
		ax.spines['bottom'].set_position(('axes', 0.1))
		ax.spines['top'].set_color('none')
		#
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		plt.setp(ax.yaxis.get_ticklabels(), visible=False)
		ax.plot(t_du[cu_du:], theta_du[cu_du:]/sc,'k')
		ax.plot(t_af1, theta_af1/sc,'k')
		plt.title(str(No[kk])+'_'+str(jj+2))
		#
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')
		plt.ylabel('position (body length)', fontsize = 20)
		plt.xlabel('time (minutes)', fontsize = 20)
		adjust_spines(ax, ['left', 'bottom'])
		plt.ylim([-25,25])
		fig.savefig('Sum_plots_numbered/lin_plots' + '/h_all_'+str(No[kk])+'_'+str(jj+2))


np.savetxt('trial_1F_data/r0_1F.txt', r0_1F)
np.savetxt('trial_1F_data/r1_1F.txt', r1_1F)

np.savetxt('trial_1F_data/t_af1_all_1F.txt', t_af1_all)
np.savetxt('trial_1F_data/RE_af1_all_1F.txt', RE_af1_all)