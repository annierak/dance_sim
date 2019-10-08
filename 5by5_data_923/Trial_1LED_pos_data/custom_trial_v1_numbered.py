################## custom functions for trial structure analysis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import math
import pandas as pd
import scipy.stats 
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
def rev_from_angle(t_du, theta_du, t_af1, theta_af1, order, dis, wid):
	################ during
	th_filter_du = scipy.signal.savgol_filter(np.unwrap(theta_du),order,1)

	peaks_p, properties = find_peaks(th_filter_du, height=None, threshold=None, distance=dis)
	peaks_n, properties = find_peaks(-th_filter_du, height=None, threshold=None, distance=dis)

	RE_du_p = th_filter_du[peaks_p] - np.pi/2
	RE_du_n = th_filter_du[peaks_n] - np.pi/2

	peaks_p_du = peaks_p
	peaks_n_du = peaks_n
	########################
	peaks_temp = np.hstack([peaks_p, peaks_n])
	peaks_temp = np.sort(peaks_temp)

	t_du_co = t_du[peaks_temp]
	RE_du_co = theta_du[peaks_temp]


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
	
	t_af1_co = t_af1[peaks_temp]
	RE_af1_co = theta_af1[peaks_temp]
	

	return t_du_co, RE_du_co, t_af1_co, RE_af1_co 

########################## adjust axis
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(direction='in', length=6, width=2, colors='k',grid_color='k', grid_alpha=0.5)
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def rev_adjust_1F(No, t_af1, theta_af1, t_du_co, RE_du_co, t_af1_co, RE_af1_co, jj):
	######################################### No = 1
	if No == 1 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[2:4], t_af1_co[6], t_af1_co[7], t_af1_co[17:20], t_af1_co[-1]])
		RE_af1_co = np.hstack([RE_af1_co[2:4], RE_af1_co[6], RE_af1_co[7], RE_af1_co[17:20], RE_af1_co[-1]])
	#
	if No == 1 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[9], t_af1_co[12], t_af1_co[18], t_af1_co[19]])
		RE_af1_co = np.hstack([RE_af1_co[9], RE_af1_co[12], RE_af1_co[18], RE_af1_co[19]])
	#
	if No == 1 and jj == 3:
		t_af1_co = np.hstack([t_af1_co[3:5], t_af1_co[21], t_af1_co[-2], t_af1_co[-1]])
		RE_af1_co = np.hstack([RE_af1_co[3:5], RE_af1_co[21], RE_af1_co[-2], RE_af1_co[-1]])
	#
	if No == 1 and jj == 4:
		t_af1_co = np.hstack([t_af1_co[10], t_af1_co[21]])
		RE_af1_co = np.hstack([RE_af1_co[10], RE_af1_co[21]])		
	#
	if No == 1 and jj == 5:
		t_af1_co = np.hstack([t_af1_co[1], t_af1_co[3], t_af1_co[4:]])
		RE_af1_co = np.hstack([RE_af1_co[1], RE_af1_co[3], RE_af1_co[4:]])		
	#
	######################################### No = 5
	if No == 2 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[0:2], t_af1_co[4:]])
		RE_af1_co = np.hstack([RE_af1_co[0:2], RE_af1_co[4:]])
	#
	if No == 2 and jj == 2:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:2], t_af1_co[6:8], t_af1_co[-1]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:2], RE_af1_co[6:8], RE_af1_co[-1]])
	#
	######################################### No = 7
	if No == 3 and jj == 3:
		t_af1_co = np.hstack([t_af1[0], t_af1_co[0:]])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co[0:]])
	#
	if No == 3 and jj in [4,6]:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:]])			
   	#
	if No == 3 and jj == 5:
		t_af1_co = np.hstack([t_af1_co[6:]])
		RE_af1_co = np.hstack([RE_af1_co[6:]])
   	#
	if No == 4 and jj in [1]:
		t_af1_co = np.hstack([t_af1[0], t_af1_co[0:]])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co[0:]])
	#
	if No == 4 and jj in [2]:
		t_af1_co = np.hstack([t_du_co[-1], t_af1[0], t_af1_co[0:7], t_af1_co[9:12], t_af1_co[14:16]])
		RE_af1_co = np.hstack([RE_du_co[-1], theta_af1[0], RE_af1_co[0:7], RE_af1_co[9:12], RE_af1_co[14:16]])	
	#
	if No == 4 and jj in [3]:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:10], t_af1_co[-5:]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:10], RE_af1_co[-5:]])
	#
	if No == 4 and jj == 5:
		t_af1_co = np.hstack([t_af1_co[2:]])
		RE_af1_co = np.hstack([RE_af1_co[2:]])	
	#
	######################################### No = 8
	if No == 5 and jj in [1]:
		t_af1_co = np.hstack([t_af1[0], t_af1_co[0:]])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co[0:]])
	#	
	if No == 5 and jj in [3]:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:]])
	#
	######################################### No = 9
	if No == 6 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[4:]])
		RE_af1_co = np.hstack([RE_af1_co[4:]])		
	#
	if No == 6 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[6:]])
		RE_af1_co = np.hstack([RE_af1_co[6:]])
	#
	if No == 6 and jj == 3:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:2], t_af1_co[-2:]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:2], RE_af1_co[-2:]])
	#
	if No == 6 and jj == 4:
		t_af1_co = np.hstack([t_af1_co[2], t_af1_co[6:8]])
		RE_af1_co = np.hstack([RE_af1_co[2], RE_af1_co[6:8]])		
	#
	if No == 6 and jj == 5:
		t_af1_co = np.hstack([t_af1_co[4], t_af1_co[7:]])
		RE_af1_co = np.hstack([RE_af1_co[4], RE_af1_co[7:]])	
	#
	######################################### No = 10
	if No == 7 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[5:]])
		RE_af1_co = np.hstack([RE_af1_co[5:]])	
	#
	if No == 7 and jj == 3:
		t_af1_co = np.hstack([t_af1_co[6], t_af1_co[8:14], t_af1_co[16:]])
		RE_af1_co = np.hstack([RE_af1_co[6], RE_af1_co[8:14], RE_af1_co[16:]])
	#
	if No == 7 and jj == 4:
		t_af1_co = np.hstack([t_af1_co[6], t_af1_co[9], t_af1_co[14], t_af1_co[17], t_af1_co[-1]])
		RE_af1_co = np.hstack([RE_af1_co[6], RE_af1_co[9], RE_af1_co[14], RE_af1_co[17], RE_af1_co[-1]])
	#
	if No == 7 and jj == 5:
		t_af1_co = np.hstack([t_af1_co[6], t_af1_co[8],t_af1_co[9], t_af1_co[12:14], t_af1_co[14], t_af1_co[15], t_af1_co[-1]])
		RE_af1_co = np.hstack([RE_af1_co[6], RE_af1_co[8], RE_af1_co[9], RE_af1_co[12:14], RE_af1_co[14], RE_af1_co[15], RE_af1_co[-1]])	
	#
	if No == 7 and jj == 6:
		t_af1_co = np.hstack([t_af1_co[11], t_af1_co[13], t_af1_co[-4:]])
		RE_af1_co = np.hstack([RE_af1_co[11], RE_af1_co[13], RE_af1_co[-4:]])	
	#
	if No == 8 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[5:12], t_af1_co[14], t_af1_co[-4:]])
		RE_af1_co = np.hstack([RE_af1_co[5:12], RE_af1_co[14], RE_af1_co[-4:]])	
	#
	if No == 8 and jj == 3:
		t_af1_co = np.hstack([t_af1_co[7:]])
		RE_af1_co = np.hstack([RE_af1_co[7:]])	
	#
	if No == 8 and jj == 5:
		t_af1_co = np.hstack([t_af1_co[0], t_af1_co[4], t_af1_co[-2:]])
		RE_af1_co = np.hstack([RE_af1_co[0], RE_af1_co[4], RE_af1_co[-2:]])	
	#
	if No == 8 and jj == 6:
		t_af1_co = np.hstack([t_af1_co[0], t_af1_co[-3:]])
		RE_af1_co = np.hstack([RE_af1_co[0], RE_af1_co[-3:]])
	#	
	######################################### No = 11
	if No == 9 and jj in [1]:
		t_af1_co = np.hstack([t_du_co[-1], t_af1[0], t_af1_co[0:]])
		RE_af1_co = np.hstack([RE_du_co[-1], theta_af1[0], RE_af1_co[0:]])
	#	
	if No == 9 and jj in [3]:
		t_af1_co = np.hstack([t_du_co[-6], t_af1_co[0:]])
		RE_af1_co = np.hstack([RE_du_co[-6], RE_af1_co[0:]])
	#	
	if No == 9 and jj in [4]:
		t_af1_co = np.hstack([t_af1[0], t_af1_co[0:]])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co[0:]])
	#	
	if No == 9 and jj in [5]:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:2], t_af1_co[-5:]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:2], RE_af1_co[-5:]])	
	#	
	######################################### No = 12
	if No == 10 and jj in [1]:
		t_af1_co = np.hstack([t_af1[0], t_af1_co[0:2], t_af1_co[4], t_af1_co[-4:]])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co[0:2], RE_af1_co[4], RE_af1_co[-4:]])	
	#	
	if No == 10 and jj in [3]:
		t_af1_co = np.hstack([t_af1_co[0:4], t_af1_co[4], t_af1_co[7], t_af1_co[8], t_af1_co[-4:]])
		RE_af1_co = np.hstack([RE_af1_co[0:4], RE_af1_co[4], RE_af1_co[7], RE_af1_co[8], RE_af1_co[-4:]])
	#	
	if No == 10 and jj in [4]:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:2], t_af1_co[4:7]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:2], RE_af1_co[4:7]])
	#	
	if No == 10 and jj in [5]:
		t_af1_co = np.hstack([t_af1[0], t_af1_co[0:]])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co[0:]])
	#
	######################################### No = 13
	if No == 11 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[7:]])
		RE_af1_co = np.hstack([RE_af1_co[7:]])	
	#
	if No == 11 and jj == 3:
		t_af1_co = np.hstack([t_af1_co[2:]])
		RE_af1_co = np.hstack([RE_af1_co[2:]])
	#
	if No == 11 and jj == 5:
		t_af1_co = np.hstack([t_af1_co[0:5], t_af1_co[-1]])
		RE_af1_co = np.hstack([RE_af1_co[0:5], RE_af1_co[-1]])
	#
	if No == 11 and jj == 6:
		t_af1_co = np.hstack([t_af1_co[21:]])
		RE_af1_co = np.hstack([RE_af1_co[21:]])	
	#	
	######################################### No = 14
	if No == 12 and jj in [1]:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:]])
	#	
	if No == 12 and jj in [2]:
		t_af1_co = np.hstack([t_af1[0], t_af1_co[0:]])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co[0:]])
	#	
	if No == 12 and jj in [3]:
		t_af1_co = np.hstack([t_af1[0], t_af1_co[0:8], t_af1_co[-5:]])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co[0:8], RE_af1_co[-5:]])	
	#	
	if No == 12 and jj in [4]:
		t_af1_co = np.hstack([t_af1_co[7], t_af1_co[10:12], t_af1_co[16:18], t_af1_co[18]])
		RE_af1_co = np.hstack([RE_af1_co[7], RE_af1_co[10:12], RE_af1_co[16:18], RE_af1_co[18]])		
	#	
	if No == 12 and jj in [5]:
		t_af1_co = np.hstack([t_af1[0], t_af1_co[0:5], t_af1_co[14], t_af1_co[15], t_af1_co[25]])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co[0:5], RE_af1_co[14], RE_af1_co[15], RE_af1_co[25]])
	#
	if No == 12 and jj == 6:
		t_af1_co = np.hstack([t_af1_co[14:18], t_af1_co[25:33], t_af1_co[42]])
		RE_af1_co = np.hstack([RE_af1_co[14:18], RE_af1_co[25:33], RE_af1_co[42]])	
	#
	######################################### No = 15
	if No == 13 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[0], t_af1_co[7:10], t_af1_co[11], t_af1_co[-2:]])
		RE_af1_co = np.hstack([RE_af1_co[0], RE_af1_co[7:10], RE_af1_co[11], RE_af1_co[-2:]])	
	#
	if No == 13 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[4], t_af1_co[7], t_af1_co[10], t_af1_co[11]])
		RE_af1_co = np.hstack([RE_af1_co[4], RE_af1_co[7], RE_af1_co[10], RE_af1_co[11]])
	#
	if No == 13 and jj == 5:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:]])
	#		
	######################################### No = 18
	if No == 14 and jj == 1:
		t_af1_co = np.hstack([t_du_co[-2],t_af1[20], t_af1_co])
		RE_af1_co = np.hstack([RE_du_co[-2],theta_af1[20], RE_af1_co])
	#
	if No == 14 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[4:]])
		RE_af1_co = np.hstack([RE_af1_co[4:]])
	#
	if No == 14 and jj == 4:
		t_af1_co = np.hstack([t_af1[0] -0.05, t_af1_co])
		RE_af1_co = np.hstack([theta_af1[0] - np.radians(10), RE_af1_co])
	#
	if No == 14 and jj == 5:
		t_af1_co = np.hstack([t_af1[30], t_af1_co])
		RE_af1_co = np.hstack([theta_af1[30], RE_af1_co])
	#
	if No == 15 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[2], t_af1_co[5:]])
		RE_af1_co = np.hstack([RE_af1_co[2], RE_af1_co[5:]])
	#
	if No == 15 and jj == 2:
		t_af1_co = np.hstack([t_af1[0], t_af1_co])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co])
	#	
	if No == 15 and jj in [3]:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:]])
	#
	if No == 15 and jj == 4:
		t_af1_co = np.hstack([t_af1_co[1:]])
		RE_af1_co = np.hstack([RE_af1_co[1:]])	
	#	
	######################################### No = 19
	if No == 16 and jj == 1:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:-9], t_af1_co[-2:]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:-9], RE_af1_co[-2:]])
	#
	if No == 16 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[0], t_af1_co[1], t_af1_co[6:8], t_af1_co[18:20], t_af1_co[-14], t_af1_co[-1]])
		RE_af1_co = np.hstack([RE_af1_co[0], RE_af1_co[1], RE_af1_co[6:8], RE_af1_co[18:20], RE_af1_co[-14], RE_af1_co[-1]])		
	#
	if No == 16 and jj == 3:
		t_af1_co = np.hstack([t_af1[0], t_af1_co[0], t_af1_co[4], t_af1_co[21], t_af1_co[33], t_af1_co[-1]])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co[0], RE_af1_co[4], RE_af1_co[21], RE_af1_co[33], RE_af1_co[-1]])
	#
	if No == 16 and jj == 4:
		t_af1_co = np.hstack([t_af1_co[14], t_af1_co[25], t_af1_co[49]])
		RE_af1_co = np.hstack([RE_af1_co[14], RE_af1_co[25], RE_af1_co[49]])	
	#
	######################################### No = 20
	if No == 17 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[0], t_af1_co[2], t_af1_co[6], t_af1_co[9:11], t_af1_co[14:]])
		RE_af1_co = np.hstack([RE_af1_co[0], RE_af1_co[2], RE_af1_co[6], RE_af1_co[9:11], RE_af1_co[14:]])	
	#
	if No == 17 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[21], t_af1_co[27]])
		RE_af1_co = np.hstack([RE_af1_co[21], RE_af1_co[27]])
	#	
	######################################### No = 21
	if No == 18 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[2:]])
		RE_af1_co = np.hstack([RE_af1_co[2:]])	
	#
	if No == 18 and jj in [2,3,5,6]:
		t_af1_co = np.hstack([t_af1[0], t_af1_co])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co])	
	#	
	if No == 18 and jj in [4]:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[0:]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[0:]])
	#
	if No == 19 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[1:]])
		RE_af1_co = np.hstack([RE_af1_co[1:]])	
	#
	if No == 19 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[4:]])
		RE_af1_co = np.hstack([RE_af1_co[4:]])
	#
	if No == 19 and jj == 3:
		t_af1_co = np.hstack([t_af1_co[0:-4]])
		RE_af1_co = np.hstack([RE_af1_co[0:-4]])
	#
	if No == 19 and jj in [4,5]:
		t_af1_co = np.hstack([t_af1[0], t_af1_co])
		RE_af1_co = np.hstack([theta_af1[0], RE_af1_co])
	#
	######################################### No = 22
	if No == 20 and jj == 4:
		t_af1_co = np.hstack([t_af1_co[7], t_af1_co[11], t_af1_co[17], t_af1_co[18], t_af1_co[-12:-4]])
		RE_af1_co = np.hstack([RE_af1_co[7], RE_af1_co[11], RE_af1_co[17], RE_af1_co[18], RE_af1_co[-12:-4]])	
	#
	######################################### No = 23
	if No == 21 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[1:]])
		RE_af1_co = np.hstack([RE_af1_co[1:]])
	#	
	if No == 21 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[1:]])
		RE_af1_co = np.hstack([RE_af1_co[1:]])
	#	
	if No == 21 and jj == 3:
		t_af1_co = np.hstack([t_af1_co[0], t_af1_co[5], t_af1_co[17], t_af1_co[-5:]])
		RE_af1_co = np.hstack([RE_af1_co[0], RE_af1_co[5], RE_af1_co[17], RE_af1_co[-5:]])
	#	
	if No == 21 and jj == 4:
		t_af1_co = np.hstack([t_af1_co[0], t_af1_co[5], t_af1_co[11], t_af1_co[14:16], t_af1_co[18:20]])
		RE_af1_co = np.hstack([RE_af1_co[0], RE_af1_co[5], RE_af1_co[11], RE_af1_co[14:16], RE_af1_co[18:20]])
	#	
	if No == 21 and jj == 6:
		t_af1_co = np.hstack([t_af1_co[10], t_af1_co[20], t_af1_co[25], t_af1_co[-5:-2]])
		RE_af1_co = np.hstack([RE_af1_co[10], RE_af1_co[20], RE_af1_co[25], RE_af1_co[-5:-2]])
	#	
	if No == 22 and jj == 1:
		t_af1_co = np.hstack([t_du_co[-1], t_af1_co[1:-8]])
		RE_af1_co = np.hstack([RE_du_co[-1], RE_af1_co[1:-8]])
	#	
	if No == 22 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[15], t_af1_co[21:23]])
		RE_af1_co = np.hstack([RE_af1_co[15], RE_af1_co[21:23]])
	#	
	if No == 22 and jj == 3:
		t_af1_co = np.hstack([t_af1_co[5], t_af1_co[14], t_af1_co[-4:]])
		RE_af1_co = np.hstack([RE_af1_co[5], RE_af1_co[14], RE_af1_co[-4:]])
	#	
	if No == 22 and jj == 4:
		t_af1_co = np.hstack([t_af1_co[20], t_af1_co[40], t_af1_co[42], t_af1_co[-2:]])
		RE_af1_co = np.hstack([RE_af1_co[20], RE_af1_co[40], RE_af1_co[42], RE_af1_co[-2:]])
	#	
	if No == 22 and jj == 6:
		t_af1_co = np.hstack([t_af1_co[5], t_af1_co[6], t_af1_co[-5:]])
		RE_af1_co = np.hstack([RE_af1_co[5], RE_af1_co[6], RE_af1_co[-5:]])				
	#
	######################################### No = 24
	if No == 23 and jj == 1:
		t_af1_co = np.hstack([t_af1_co[27], t_af1_co[-3:]])
		RE_af1_co = np.hstack([RE_af1_co[27], RE_af1_co[-3:]])	
	#												
	if No == 23 and jj == 2:
		t_af1_co = np.hstack([t_af1_co[2:]])
		RE_af1_co = np.hstack([RE_af1_co[2:]])
	#												
	if No == 23 and jj == 4:
		t_af1_co = np.hstack([t_af1_co[0], t_af1_co[12], t_af1_co[18], t_af1_co[19], t_af1_co[21], t_af1_co[-2:]])
		RE_af1_co = np.hstack([RE_af1_co[0], RE_af1_co[12], RE_af1_co[18], RE_af1_co[19], RE_af1_co[21], RE_af1_co[-2:]])
	#												
	if No == 23 and jj == 5:
		t_af1_co = np.hstack([t_af1_co[2], t_af1_co[31], t_af1_co[-4:]])
		RE_af1_co = np.hstack([RE_af1_co[2], RE_af1_co[31], RE_af1_co[-4:]])		
	#


   	#if No == 1 and tr == 1 and jj == 0:
   		#t_af1_co = np.hstack([t_af1_co[1:3], t_af1_co[5:7], t_af1_co[10:13], t_af1_co[-1]])
		#RE_af1_co = np.hstack([RE_af1_co[1:3], RE_af1_co[5:7], RE_af1_co[10:13], RE_af1_co[-1]])
			
	return t_af1_co, RE_af1_co
#
def r01_finder_1F(No, tr, t_du_co, RE_du_co, t_af1_co, RE_af1_co,jj):
   	##################################### c calculation
	if No == 1 and tr == 1 and jj == 0:
		r0 = RE_af1_co[1]
		r1 = RE_af1_co[2] - RE_af1_co[1]
	#
	if No == 5 and tr == 1 and jj == 0:
		r0 = RE_af1_co[0]
		r1 = RE_af1_co[1] - RE_af1_co[0]

	return r0, r1  

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h