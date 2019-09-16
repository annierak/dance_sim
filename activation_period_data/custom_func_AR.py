################## custom functions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import math
import pandas as pd
from scipy.stats import sem
import string
import scipy.signal
from scipy.signal import find_peaks


############# find nearest
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

############# safe div
def safe_div(x,y):
    return 1.0*x/y if y else 0


############# cus atan
def cus_atan(y,x):
    if x >= 0 and y >= 0:
    	return math.atan2(abs(y), abs(x)) + 1*np.pi/2
    #
    if x >= 0 and y < 0 :
    	return -math.atan2(abs(y), abs(x)) + 1*np.pi/2
    #
    if x < 0 and y < 0 :
    	return math.atan2(abs(y), abs(x)) - 1*np.pi/2
    #
    if x < 0 and y > 0 :
    	return -math.atan2(abs(y), abs(x)) - 1*np.pi/2


########## finding reversals
def find_rev(case, fly, flip, af1_d, order, dis, wid, no):
 	################ set the directory
	if case == 1: # 1 food
		Nam = '1LED/1LED_'+str(fly)
	if case == 2: # 2 LED 60
		Nam = '2LED60/2LED60_'+str(fly)
	if case == 3: # 2 LED 90
		Nam = '2LED90/2LED90_'+str(fly)
	if case == 4: # 2 LED 120
		Nam = '2LED120/2LED120_'+str(fly)
	if case == 5: # 3 LED 0, 30, 90
		Nam = '3LED03090/3LED03090_'+str(fly)
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

	#############################################################
	############################################################# finding the last led
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

	print('last LED:', pin_f[len(val_f)-2])
	if pin_f[len(val_f)-2] == 3:
		print('x_not flipped')
	if pin_f[len(val_f)-2] == 10 or pin_f[len(val_f)-2] == 11:
		if flip == 0:
			print('flipping not activated,', 'x_not flipped')
			xx = 1.0*xx
		if flip == 1:
			print('flipping activated,', 'x_flipped')
			xx = -1.0*xx
	############################################################# end of finding the last led
	last_LED = pin_f[len(val_f)-2]
	############################################################# finding the first led
	#
	LL = np.zeros(len(val_f))
	TT = np.zeros(len(val_f))

	for ii in range(len(val_f)):
		if val_f[ii] == 100 and pin_f[ii] == 10:
			LL[ii] = 300
			TT[ii] = t_adj[ii]
		if val_f[ii] == 100 and pin_f[ii] == 3:
			LL[ii] = -400
			TT[ii] = t_adj[ii]

	LL_nz = LL[np.nonzero(LL)]
	TT_nz = TT[np.nonzero(TT)]

	if LL_nz[0] == 300:
		ind_10 = np.where(pin_f == 10)
		ind_L = ind_10


	if LL_nz[0] == -400:
		ind_3 = np.where(pin_f == 3)
		ind_L = ind_3

	############# angle shift
	if case == 1 and fly >= 6 and fly < 22:
		ang_sh = 0
	if case == 1 and fly < 6:
		ang_sh = np.pi/4
	if case == 1 and fly >= 22:
		ang_sh = np.pi/4

	#print('ind_L:', ind_L[0][0])
	#print('t:', TT[ind_L[0][0]])
	ind_du_2 = find_nearest(t, TT[ind_L[0][0]])

	if case < 5:
		C_LED = 0
	C = np.zeros(3)
	############################################################# end of finding the first led

	#############################################################
	ind_be_e = find_nearest(t, 2)
	ind_du_e = find_nearest(t, 50)
	ind_af_e = find_nearest(t, 60)
	ind_af1_e = find_nearest(t, 50+af1_d)
	ind_af2_e = find_nearest(t, 50+2*af1_d)
	ind_af3_e = find_nearest(t, 50+3*af1_d)
	ind_af4_e = find_nearest(t, 50+4*af1_d)
	ind_af5_e = find_nearest(t, 50+5*af1_d)
	ind_af_e = find_nearest(t, 60)

	ind_due_e = find_nearest(t, 50-0.25*af1_d)


	ind_be = range(0, ind_be_e, 1)
	ind_du = range(ind_be_e  + 1, ind_du_e, 1)
	ind_af = range(ind_du_e  + 1, ind_af_e, 1)
	ind_af1 = range(ind_du_e  + 1, ind_af1_e, 1)
	ind_af2 = range(ind_af1_e  + 1, ind_af2_e, 1)
	ind_af3 = range(ind_af2_e  + 1, ind_af3_e, 1)
	ind_af4 = range(ind_af3_e  + 1, ind_af4_e, 1)
	ind_af5 = range(ind_af4_e  + 1, ind_af5_e, 1)

	ind_t1 = range(ind_due_e  + 1, ind_du_e, 1)

	t = t[ind_be[0]:ind_af[-1]]
	xx = xx[ind_be[0]:ind_af[-1]]
	yy = yy[ind_be[0]:ind_af[-1]]

	mx = min(xx)
	Mx = max(xx)
	my = min(yy)
	My = max(yy)

	# adjusting the center
	xx = xx - 0.5*(mx + Mx)
	yy = yy - 0.5*(my + My)

	############ breaking the regions
	t_be = t[ind_be[0]:ind_be[-1]]
	xx_be = xx[ind_be[0]:ind_be[-1]]
	yy_be = yy[ind_be[0]:ind_be[-1]]

	t_du = t[ind_du[0]:ind_du[-1]]
	xx_du = xx[ind_du[0]:ind_du[-1]]
	yy_du = yy[ind_du[0]:ind_du[-1]]

	t_af1 = t[ind_af1[0]:ind_af1[-1]]
	xx_af1 = xx[ind_af1[0]:ind_af1[-1]]
	yy_af1 = yy[ind_af1[0]:ind_af1[-1]]

	t_af2 = t[ind_af2[0]:ind_af2[-1]]
	xx_af2 = xx[ind_af2[0]:ind_af2[-1]]
	yy_af2 = yy[ind_af2[0]:ind_af2[-1]]

	t_af3 = t[ind_af3[0]:ind_af3[-1]]
	xx_af3 = xx[ind_af3[0]:ind_af3[-1]]
	yy_af3 = yy[ind_af3[0]:ind_af3[-1]]

	t_af4 = t[ind_af4[0]:ind_af4[-1]]
	xx_af4 = xx[ind_af4[0]:ind_af4[-1]]
	yy_af4 = yy[ind_af4[0]:ind_af4[-1]]

	t_af5 = t[ind_af5[0]:ind_af5[-1]]
	xx_af5 = xx[ind_af5[0]:ind_af5[-1]]
	yy_af5 = yy[ind_af5[0]:ind_af5[-1]]

	t_af = t[ind_af[0]:ind_af[-1]]
	xx_af = xx[ind_af[0]:ind_af[-1]]
	yy_af = yy[ind_af[0]:ind_af[-1]]

	t_t1 = t[ind_t1[0]:ind_t1[-1]]
	xx_t1 = xx[ind_t1[0]:ind_t1[-1]]
	yy_t1 = yy[ind_t1[0]:ind_t1[-1]]
	############################ adjusted by the LED
	ind_be_e1 = find_nearest(t, t_adj[0])

	ind_be1 = range(0, ind_be_e1, 1)
	ind_du1 = range(ind_be_e1  + 1, ind_du_e, 1)


	t_be1 = t[ind_be1[0]:ind_be1[-1]]
	xx_be1 = xx[ind_be1[0]:ind_be1[-1]]
	yy_be1 = yy[ind_be1[0]:ind_be1[-1]]
	#print(t_be1[-1])

	t_du1 = t[ind_du1[0]:ind_du1[-1]]
	xx_du1 = xx[ind_du1[0]:ind_du1[-1]]
	yy_du1 = yy[ind_du1[0]:ind_du1[-1]]
	#print(t_du1[-1])

	############################ adjusted for the second LED
	ind_du2 = range(ind_du_2  + 1, ind_du_e, 1)

	t_du2 = t[ind_du2[0]:ind_du2[-1]]
	xx_du2 = xx[ind_du2[0]:ind_du2[-1]]
	yy_du2 = yy[ind_du2[0]:ind_du2[-1]]
	#print('t_du2_start:', t_du2[0])

	################################
	#th = np.linspace(-np.pi, np.pi, num = no)
	#print(th)

	theta_be = np.zeros(len(xx_be))
	theta_du = np.zeros(len(xx_du))
	theta_du1 = np.zeros(len(xx_du1))
	theta_du2 = np.zeros(len(xx_du2))
	theta_af = np.zeros(len(xx_af))
	theta_af1 = np.zeros(len(xx_af1))
	theta_af2 = np.zeros(len(xx_af2))
	theta_af3 = np.zeros(len(xx_af3))
	theta_af4 = np.zeros(len(xx_af4))
	theta_af5 = np.zeros(len(xx_af5))

	theta_t1 = np.zeros(len(xx_t1))

	#####  before
	for ii in range(len(xx_be)):
		theta_be[ii] = cus_atan(yy_be[ii], xx_be[ii])


	###  during
	for ii in range(len(xx_du)):
		theta_du[ii] = cus_atan(yy_du[ii], xx_du[ii])

	###  during 1
	for ii in range(len(xx_du1)):
		theta_du1[ii] = cus_atan(yy_du1[ii], xx_du1[ii])

	###  during 2 after the second LED
	for ii in range(len(xx_du2)):
		theta_du2[ii] = cus_atan(yy_du2[ii], xx_du2[ii])

	###  after 1
	for ii in range(len(xx_af1)):
		theta_af1[ii] = cus_atan(yy_af1[ii], xx_af1[ii])

	###  after 2
	for ii in range(len(xx_af2)):
		theta_af2[ii] = cus_atan(yy_af2[ii], xx_af2[ii])

	###  after 3
	for ii in range(len(xx_af3)):
		theta_af3[ii] = cus_atan(yy_af3[ii], xx_af3[ii])

	###  after 3
	for ii in range(len(xx_af4)):
		theta_af4[ii] = cus_atan(yy_af4[ii], xx_af4[ii])

	###  after 5
	for ii in range(len(xx_af5)):
		theta_af5[ii] = cus_atan(yy_af5[ii], xx_af5[ii])

	###  transition 1
	for ii in range(len(xx_t1)):
		theta_t1[ii] = cus_atan(yy_t1[ii], xx_t1[ii])

	#### ADJUSTING
	#####  before
	theta_be1 = np.zeros(len(xx_be1))
	theta_du1 = np.zeros(len(xx_du1))
	theta_du2 = np.zeros(len(xx_du2))


	for ii in range(len(xx_be1)):
		theta_be1[ii] = cus_atan(yy_be1[ii], xx_be1[ii])

	###  during
	for ii in range(len(xx_du1)):
		theta_du1[ii] = cus_atan(yy_du1[ii], xx_du1[ii])

	###  during 2 after the second LED
	for ii in range(len(xx_du2)):
		theta_du2[ii] = cus_atan(yy_du2[ii], xx_du2[ii])

	if case == 1:
		theta_be = theta_be + ang_sh
		theta_be1 = theta_be1 + ang_sh
		theta_du1 = theta_du1 + ang_sh
		theta_du2 = theta_du2 + ang_sh
		theta_af1 = theta_af1 + ang_sh
		theta_af2 = theta_af2 + ang_sh
		theta_af3 = theta_af3 + ang_sh
		theta_af4 = theta_af4 + ang_sh
		theta_af5 = theta_af5 + ang_sh

		theta_t1 = theta_t1 + ang_sh
	##########################################
	e_be = np.zeros(len(t_be))
	e_be1 = np.zeros(len(t_be1))
	e_du2 = np.zeros(len(t_du2))
	e_du2L5 = np.zeros(len(t_du2))
	e_af1 = np.zeros(len(t_af1))
	e_af1L5 = np.zeros(len(t_af1))

	e_t1 = np.zeros(len(t_t1))

	aa = 0.5
	for ii in range(len(t_be)):
		e_be[ii] = math.exp(0.5*aa*t_be[ii])
	for ii in range(len(t_be1)):
		e_be1[ii] = math.exp(0.5*aa*t_be1[ii])
	for ii in range(len(t_du2)):
		e_du2[ii] = math.exp(0.2*0.5*aa*t_du2[ii])
	for ii in range(len(t_du2)):
		e_du2L5[ii] = math.exp(10*0.2*0.5*aa*t_du2[ii])
	for ii in range(len(t_af1)):
		e_af1[ii] = math.exp(aa*t_af1[ii])
	for ii in range(len(t_af1)):
		e_af1L5[ii] = math.exp(10*0.2*0.5*aa*t_af1[ii])

	for ii in range(len(t_t1)):
		e_t1[ii] = math.exp(aa*t_t1[ii])
	##########################################
	################ before
	th_filter_be = scipy.signal.savgol_filter(np.unwrap(theta_be),3*order,1)

	peaks_p, properties = find_peaks(th_filter_be, height=None, threshold=None, distance=dis*1, width = wid)
	peaks_n, properties = find_peaks(-th_filter_be, height=None, threshold=None, distance=dis*1, width = wid)

	#RE_be_p[0:len(peaks_p)] = th_filter_be[peaks_p] - np.pi/2
	#RE_be_n[0:len(peaks_n)] = th_filter_be[peaks_n] - np.pi/2

	#print(t_be[peaks_p])
	t_peaks_be = list(list(t_be[peaks_p]) + list(t_be[peaks_n]))
	t_peaks_be.sort()
	#print(t_peaks_be)

	ang_be = np.zeros(len(t_peaks_be))
	for mm in range(len(ang_be)):
		ind_temp = find_nearest(t_be, t_peaks_be[mm])
		ang_be[mm] = theta_be[ind_temp]

	ang_be_diff = np.diff(np.array(ang_be))

	peaks_p_be = peaks_p
	peaks_n_be = peaks_n
	#
	r_be = np.ones(len(t_be))
	R_be = np.linspace(0, e_be[-1], 10)
	th1 = (-np.pi/2)*np.ones(len(R_be))
	####################
	################ during
	th_filter_du2 = scipy.signal.savgol_filter(np.unwrap(theta_du2),order,1)

	peaks_p, properties = find_peaks(th_filter_du2, height=None, threshold=None, distance=dis)
	peaks_n, properties = find_peaks(-th_filter_du2, height=None, threshold=None, distance=dis)

	peaks_p_du2 = peaks_p
	peaks_n_du2 = peaks_n

	#RE_du2_p[0:len(peaks_p)] = th_filter_du2[peaks_p] - np.pi/2
	#RE_du2_n[0:len(peaks_n)] = th_filter_du2[peaks_n] - np.pi/2

	#print(t_du2[peaks_p])
	t_peaks_du2 = list(list(t_du2[peaks_p]) + list(t_du2[peaks_n]))
	t_peaks_du2.sort()

	ang_du2 = np.zeros(len(t_peaks_du2))
	for mm in range(len(ang_du2)):
		ind_temp = find_nearest(t_du2, t_peaks_du2[mm])
		ang_du2[mm] = theta_du2[ind_temp]

	ang_du2_diff = np.diff(np.array(ang_du2))

	###
	r_du2 = np.ones(len(t_du2))
	R_du2 = np.linspace(0, e_du2[-1], 10)
	th1 = (-np.pi/2)*np.ones(len(R_du2))
	#
	th_time = 0.1
	for ii in range(len(peaks_p)-1):
		if t_du2[peaks_p[ii+1]] - t_du2[peaks_p[ii]] < th_time:
			peaks_p[ii] = 0
	for ii in range(len(peaks_n)-1):
		if t_du2[peaks_n[ii+1]] - t_du2[peaks_n[ii]] < th_time:
			peaks_n[ii] = 0


	################ after 1
	th_filter_af1 = scipy.signal.savgol_filter(np.unwrap(theta_af1),order,1)

	peaks_p, properties = find_peaks(th_filter_af1, height=None, threshold=None, distance=dis)
	peaks_n, properties = find_peaks(-th_filter_af1, height=None, threshold=None, distance=dis)

	RE_af1_p = th_filter_af1[peaks_p] - np.pi/2
	RE_af1_n = th_filter_af1[peaks_n] - np.pi/2

	peaks_p_af1 = peaks_p
	peaks_n_af1 = peaks_n
	########################
	peaks_temp = np.hstack([peaks_p, peaks_n])
	peaks_temp = np.sort(peaks_temp)
	#print('######### time')
	#print(t_af1[peaks_temp])
	#print('######### angle')
	#print((th_filter_af1[peaks_temp] - np.pi/2)*(180/np.pi))

	t_af1_co = t_af1[peaks_temp]
	RE_af1_co = theta_af1[peaks_temp]

	################ transition 1
	th_filter_t1 = scipy.signal.savgol_filter(np.unwrap(theta_t1),order,1)

	peaks_p, properties = find_peaks(th_filter_t1, height=None, threshold=None, distance=dis)
	peaks_n, properties = find_peaks(-th_filter_t1, height=None, threshold=None, distance=dis)

	RE_t1_p = th_filter_t1[peaks_p] - np.pi/2
	RE_t1_n = th_filter_t1[peaks_n] - np.pi/2

	peaks_p_t1 = peaks_p
	peaks_n_t1 = peaks_n
	########################
	peaks_temp = np.hstack([peaks_p, peaks_n])
	peaks_temp = np.sort(peaks_temp)

	t_t1_co = t_t1[peaks_temp]
	RE_t1_co = theta_t1[peaks_temp]

	return t_af1, xx_af1, yy_af1, e_af1, e_af1L5, theta_af1, t_af1_co, RE_af1_co, theta_du2, t_du2, e_du2, e_du2L5, t_t1, xx_t1, yy_t1, e_t1, theta_t1, t_t1_co, RE_t1_co, C_LED, last_LED
