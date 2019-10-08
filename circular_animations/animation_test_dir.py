# animation for small circle 60 degrees
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import math
import pandas as pd

####### 60 mins (10, 40, 10)
####### 2 LED 60 degrees apart, 15 s

############# find nearest
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
no = 25
af1_d = 2

mod = 5 # 1: before, 2: during, 3: after 1, 4: after 2, 5: after, 6: arbitrary
mod2 = 1 # no average: 0, average:1 

# for mod == 6 only
t_s = 0
t_e = 10
    
No = 27 # fly
tr = 1 # trial

Nam = 'Gr5a_small_circle40_2led60_v2_'+str(No)+'_'+str(tr) # Name of the directory

##########
RF = 1.1
F1_x = RF*np.cos(-1*np.pi/3)
F1_y = RF*np.sin(-1*np.pi/3)
#
F2_x = RF*np.cos(-2*np.pi/3)
F2_y = RF*np.sin(-2*np.pi/3)
##########

#xxc = [1.0000, 0.8696, 0.0652, -0.0652, -0.8696, -1.0000];
xxc = [1.0000, 0.8, 0.1, -0.1, -0.8, -1.0000];
yyc = xxc;

x1 = np.mean([xxc[5], xxc[4]])
x2 = np.mean([xxc[3], xxc[2]])
x3 = np.mean([xxc[1], xxc[0]])
x4 = np.mean([xxc[1], xxc[0]])
x5 = np.mean([xxc[1], xxc[0]])
x6 = np.mean([xxc[3], xxc[2]])
x7 = np.mean([xxc[5], xxc[4]])
x8 = np.mean([xxc[5], xxc[4]])
x9 = np.mean([xxc[3], xxc[2]])

y1 = np.mean([yyc[1], yyc[0]])
y2 = np.mean([yyc[1], yyc[0]])
y3 = np.mean([yyc[1], yyc[0]])
y4 = np.mean([yyc[3], yyc[2]])
y5 = np.mean([yyc[5], yyc[4]])
y6 = np.mean([yyc[5], yyc[4]])
y7 = np.mean([yyc[5], yyc[4]])
y8 = np.mean([yyc[3], yyc[2]])
y9 = np.mean([yyc[3], yyc[2]])
 


xx = np.loadtxt(Nam+'/pos_x.txt')
yy = np.loadtxt(Nam+'/pos_y.txt')

if No == 1 and tr == 1:
	xx_t = np.loadtxt(Nam+'/pos_x.txt')
	yy_t = np.loadtxt(Nam+'/pos_y.txt')

	xx = np.zeros(len(xx_t))
	yy = np.zeros(len(yy_t))

	th_x = 300
	th_y = 350
	for ii in range(len(xx)):
		if xx_t[ii] < th_x:
			xx[ii] = xx_t[ii]
		if xx_t[ii] > th_x:
			xx[ii] = xx_t[ii-1]
		#	
		if yy_t[ii] > th_y:
			yy[ii] = yy_t[ii]
		if yy_t[ii] < th_y:
			yy[ii] = yy_t[ii-1]			

	plt.figure(91)
	plt.subplot(211)
	plt.plot(xx_t, '.g')
	plt.plot(xx, '.r')
	plt.grid()
	plt.subplot(212)
	plt.plot(yy_t, '.g')
	plt.plot(yy, '.r')
	plt.grid()

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
#######
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
		LL[ii] = 300
		TT[ii] = t_adj[ii]
	if val_f[ii] == 100 and pin_f[ii] == 3:
		LL[ii] = -400
		TT[ii] = t_adj[ii]	

LL_nz = LL[np.nonzero(LL)]
TT_nz = TT[np.nonzero(TT)]	

#print('TT_nz:', TT_nz[-1])
#print('LL_nz:', LL_nz[-1])	


xx = xx - 180
yy = yy - 120
xx = np.divide(xx,46*0.95)
yy = np.divide(yy,46*0.95)


########## indices
#ind_be_e = find_nearest(t, 10)
ind_be_e = find_nearest(t, t_adj[0])
ind_du_e = find_nearest(t, 50)
ind_af_e = find_nearest(t, 60)
ind_af1_e = find_nearest(t, 50+af1_d)
#ind_af2_e = 

ind_be = range(0, ind_be_e, 1)
ind_du = range(ind_be_e  + 1, ind_du_e, 1)
ind_af = range(ind_du_e  + 1, ind_af_e, 1)
ind_af1 = range(ind_du_e  + 1, ind_af1_e, 1)
ind_af2 = range(ind_af1_e  + 1, ind_af_e, 1)

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

plt.figure(1000)
plt.plot(xx,yy, '.')

plt.figure(2000)
plt.subplot(211)
plt.plot(t, xx)
plt.grid()
plt.ylabel('x')
plt.subplot(212)
plt.plot(t, yy)
plt.grid()
plt.ylabel('y')
plt.xlabel('time (min)')

#plt.figure(100)
#plt.plot(t,'ob')
#print(t[0:100])
############ breaking the regions
t_be = t[ind_be[0]:ind_be[-1]]
xx_be = xx[ind_be[0]:ind_be[-1]]
yy_be = yy[ind_be[0]:ind_be[-1]]
print(t_be[-1])
#print(t[-1])
t_du = t[ind_du[0]:ind_du[-1]]
xx_du = xx[ind_du[0]:ind_du[-1]]
yy_du = yy[ind_du[0]:ind_du[-1]]
print(t_du[-1])
t_af1 = t[ind_af1[0]:ind_af1[-1]]
xx_af1 = xx[ind_af1[0]:ind_af1[-1]]
yy_af1 = yy[ind_af1[0]:ind_af1[-1]]
print(t_af1[-1])
t_af2 = t[ind_af2[0]:ind_af2[-1]]
xx_af2 = xx[ind_af2[0]:ind_af2[-1]]
yy_af2 = yy[ind_af2[0]:ind_af2[-1]]
print(t_af2[-1])
t_af = t[ind_af[0]:ind_af[-1]]
xx_af = xx[ind_af[0]:ind_af[-1]]
yy_af = yy[ind_af[0]:ind_af[-1]]



x_avg_be = np.mean(xx_be)
y_avg_be = np.mean(yy_be)

x_avg_du = np.mean(xx_du)
y_avg_du = np.mean(yy_du)

x_avg_af = np.mean(xx_af)
y_avg_af = np.mean(yy_af)

x_avg_af1 = np.mean(xx_af1)
y_avg_af1 = np.mean(yy_af1)

x_avg_af2 = np.mean(xx_af2)
y_avg_af2 = np.mean(yy_af2)

plt.figure(50)
plt.subplot(131)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.axis('equal')
plt.plot(x_avg_be, y_avg_be, 'ob')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(132)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.axis('equal')
plt.plot(x_avg_du, y_avg_du, 'or')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(133)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.axis('equal')
plt.plot(x_avg_af1, y_avg_af1, 'oc')
plt.plot(x_avg_af2, y_avg_af2, 'om')
plt.plot(x_avg_af, y_avg_af, 'ok')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')


fig = plt.figure(100)
plt.subplot(131)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.plot(xx_be, yy_be, '.b')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(132)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.plot(xx_du, yy_du, '.b')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(133)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.plot(xx_af, yy_af, '.b')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
fig.savefig(Nam+'/small_circle_res_'+str(No)+'_'+str(tr)+'.png')


########################################
plt.figure(200)
plt.subplot(131)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
plt.hexbin(xx_be, yy_be, bins='log', cmap=plt.cm.Reds, alpha = 0.3)
#plt.plot(xx_be, yy_be, '.b')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(132)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
plt.hexbin(xx_du, yy_du, bins='log', cmap=plt.cm.Reds, alpha = 0.3)
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(133)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
plt.hexbin(xx_af, yy_af, bins='log', cmap=plt.cm.Reds, alpha = 0.3)
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
###########################################
########################################
fig = plt.figure(201)
plt.subplot(141)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
plt.hexbin(xx_be, yy_be, bins='log', cmap=plt.cm.Reds, alpha = 0.3)
#plt.plot(xx_be, yy_be, '.b')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(142)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
plt.hexbin(xx_du, yy_du, bins='log', cmap=plt.cm.Reds, alpha = 0.3)
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(143)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
plt.hexbin(xx_af1, yy_af1, bins='log', cmap=plt.cm.Reds, alpha = 0.3)
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(144)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
plt.hexbin(xx_af2, yy_af2, bins='log', cmap=plt.cm.Reds, alpha = 0.3)
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
fig.savefig(Nam+'/small_circle_red_'+str(No)+'_'+str(tr)+'.png')
###########################################


################################################ zones
reg_be = np.zeros(len(xx_be))
reg_du = np.zeros(len(xx_du))
reg_af1 = np.zeros(len(xx_af1))
reg_af2 = np.zeros(len(xx_af2))
reg_af = np.zeros(len(xx_af))

xxz = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
yyz = xxz

# initializing the positions for the residency plots
x_z = np.zeros(no)
y_z = np.zeros(no)

th = np.linspace(-np.pi, np.pi, num = no)
#print('th:', th)

theta_be = np.zeros(len(xx_be))
theta_du = np.zeros(len(xx_du))
theta_af = np.zeros(len(xx_af))
theta_af1 = np.zeros(len(xx_af1))
theta_af2 = np.zeros(len(xx_af2))

##### zones before
for ii in range(len(xx_be)):
	theta_be[ii] = math.atan2(yy_be[ii], xx_be[ii])
	for jj in range(len(th) - 1):
		if theta_be[ii] > th[jj] and theta_be[ii] <= th[jj+1]:
			reg_be[ii] = jj + 1
			x_z[jj] = np.cos(0.5*(th[jj] + th[jj+1]))
			y_z[jj] = np.sin(0.5*(th[jj] + th[jj+1]))
	

reg_count_be = np.zeros(no)
for ii in range(no):
	ind = ii + 1
	reg_count_be[ii] = len(reg_be[reg_be == ind])

### zones during
for ii in range(len(xx_du)):
	theta_du[ii] = math.atan2(yy_du[ii], xx_du[ii])
	for jj in range(len(th) - 1):
		if theta_du[ii] > th[jj] and theta_du[ii] <= th[jj+1]:
			reg_du[ii] = jj + 1
			x_z[jj] = np.cos(0.5*(th[jj] + th[jj+1]))
			y_z[jj] = np.sin(0.5*(th[jj] + th[jj+1]))
	

reg_count_du = np.zeros(no)
for ii in range(no):
	ind = ii + 1
	reg_count_du[ii] = len(reg_du[reg_du == ind])

### zones after 1
for ii in range(len(xx_af1)):
	theta_af1[ii] = math.atan2(yy_af1[ii], xx_af1[ii])
	for jj in range(len(th) - 1):
		if theta_af1[ii] > th[jj] and theta_af1[ii] <= th[jj+1]:
			reg_af1[ii] = jj + 1
			x_z[jj] = np.cos(0.5*(th[jj] + th[jj+1]))
			y_z[jj] = np.sin(0.5*(th[jj] + th[jj+1]))
	

reg_count_af1 = np.zeros(no)
for ii in range(no):
	ind = ii + 1
	reg_count_af1[ii] = len(reg_af1[reg_af1 == ind])

### zones after 2
for ii in range(len(xx_af2)):
	theta_af2[ii] = math.atan2(yy_af2[ii], xx_af2[ii])
	for jj in range(len(th) - 1):
		if theta_af2[ii] > th[jj] and theta_af2[ii] <= th[jj+1]:
			reg_af2[ii] = jj + 1
			x_z[jj] = np.cos(0.5*(th[jj] + th[jj+1]))
			y_z[jj] = np.sin(0.5*(th[jj] + th[jj+1]))
	

reg_count_af2 = np.zeros(no)
for ii in range(no):
	ind = ii + 1
	reg_count_af2[ii] = len(reg_af2[reg_af2 == ind])

### zones after 
for ii in range(len(xx_af)):
	theta_af[ii] = math.atan2(yy_af[ii], xx_af[ii])
	for jj in range(len(th) - 1):
		if theta_af[ii] > th[jj] and theta_af[ii] <= th[jj+1]:
			reg_af[ii] = jj + 1
			x_z[jj] = np.cos(0.5*(th[jj] + th[jj+1]))
			y_z[jj] = np.sin(0.5*(th[jj] + th[jj+1]))
	

reg_count_af = np.zeros(no)
for ii in range(no):
	ind = ii + 1
	reg_count_af[ii] = len(reg_af[reg_af == ind])																	


###########################################
fig = plt.figure(101)
plt.subplot(211)
plt.plot(t_be, (180/np.pi)*(theta_be + np.pi/4), 'b')
plt.plot(t_du, (180/np.pi)*(theta_du + np.pi/4), 'r')
plt.plot(t_af, (180/np.pi)*(theta_af + np.pi/4), 'k')
#
plt.plot(TT_nz, LL_nz, 'og')
#
#plt.xlabel('time (min)')
plt.ylabel('angle from food 1 (deg)')
plt.grid()
plt.subplot(212)
plt.plot(t_be, (180/np.pi)*(theta_be - np.pi/4 + np.pi), 'b')
plt.plot(t_du, (180/np.pi)*(theta_du - np.pi/4 + np.pi), 'r')
plt.plot(t_af, (180/np.pi)*(theta_af - np.pi/4 + np.pi), 'k')
#
plt.plot(TT_nz, LL_nz, 'og')
#
plt.xlabel('time (min)')
plt.ylabel('angle from food 2 (deg)')
plt.grid()
fig.savefig(Nam+'/small_circle_angles_'+str(No)+'_'+str(tr)+'.png')

plt.figure(102)
plt.plot(t_be, (180/np.pi)*theta_be, 'b')
plt.plot(t_du, (180/np.pi)*theta_du, 'r')
plt.plot(t_af, (180/np.pi)*theta_af, 'k')
plt.xlabel('time (min)')
plt.ylabel('angle (deg)')
plt.grid()

###############################
reg_be_r = np.zeros(len(reg_be), dtype = int)
reg_du_r = np.zeros(len(reg_du), dtype = int)
reg_af_r = np.zeros(len(reg_af), dtype = int)
reg_af1_r = np.zeros(len(reg_af1), dtype = int)
reg_af2_r = np.zeros(len(reg_af2), dtype = int)

for qq in range(len(reg_be)-1):
	    for ww in range(49):
	    	if (reg_be[qq] == ww+1 and reg_be[qq+1] != ww+1): 
	    	    reg_be_r[qq] = ww+1
for qq in range(len(reg_du)-1):
	    for ww in range(49):
	    	if (reg_du[qq] == ww+1 and reg_du[qq+1] != ww+1): 
	    	    reg_du_r[qq] = ww+1
for qq in range(len(reg_af)-1):
	    for ww in range(49):
	    	if (reg_af[qq] == ww+1 and reg_af[qq+1] != ww+1): 
	    	    reg_af_r[qq] = ww+1
for qq in range(len(reg_af1)-1):
	    for ww in range(49):
	    	if (reg_af1[qq] == ww+1 and reg_af1[qq+1] != ww+1): 
	    	    reg_af1_r[qq] = ww+1	
for qq in range(len(reg_af2)-1):
	    for ww in range(49):
	    	if (reg_af2[qq] == ww+1 and reg_af2[qq+1] != ww+1): 
	    	    reg_af2_r[qq] = ww+1

reg_be_nz = list(reg_be_r[np.nonzero(reg_be_r)])
reg_du_nz = list(reg_du_r[np.nonzero(reg_du_r)])
reg_af_nz = list(reg_af_r[np.nonzero(reg_af_r)])
reg_af1_nz = list(reg_af1_r[np.nonzero(reg_af1_r)])
reg_af2_nz = list(reg_af2_r[np.nonzero(reg_af2_r)])

L_reg_be_nz = list(zip(reg_be_nz[:-2],reg_be_nz[1:-1],reg_be_nz[2:]))
L_reg_du_nz = list(zip(reg_du_nz[:-2],reg_du_nz[1:-1],reg_du_nz[2:]))
L_reg_af_nz = list(zip(reg_af_nz[:-2],reg_af_nz[1:-1],reg_af_nz[2:]))
L_reg_af1_nz = list(zip(reg_af1_nz[:-2],reg_af1_nz[1:-1],reg_af1_nz[2:]))
L_reg_af2_nz = list(zip(reg_af2_nz[:-2],reg_af2_nz[1:-1],reg_af2_nz[2:]))	    	    	    	        	    


######################
#print('x_z:', x_z)
#print('y_z:', y_z)


sc = 0.1 # scaling
plt.figure(60)
plt.subplot(131)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_be)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('before')
#
plt.subplot(132)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_du)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('during')
#
plt.subplot(133)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_af)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('after')

###########################################
fig = plt.figure(61)
plt.subplot(141)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_be)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('before')
#
plt.subplot(142)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_du)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('during')
#
plt.subplot(143)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_af1)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('after 1')
#
plt.subplot(144)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_af2)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('after 2')
fig.savefig(Nam+'/small_circle_zones_'+str(No)+'_'+str(tr)+'.png')

####################################################
np.savetxt(Nam + '/reg_count_be.txt', reg_count_be)
np.savetxt(Nam + '/reg_count_du.txt', reg_count_du)
np.savetxt(Nam + '/reg_count_af.txt', reg_count_af)
np.savetxt(Nam + '/reg_count_af1.txt', reg_count_af1)
np.savetxt(Nam + '/reg_count_af2.txt', reg_count_af2)

###########################################
ang = np.linspace(0, 360-360/no, no)
fig = plt.figure(161)
plt.subplot(411)
plt.plot(ang, reg_count_be/max(reg_count_be), 'o')
plt.grid()
plt.ylabel('before')
#
plt.subplot(412)
plt.plot(ang, reg_count_du/max(reg_count_du), 'o')
plt.grid()
plt.ylabel('during')
#
plt.subplot(413)
plt.plot(ang, reg_count_af1/max(reg_count_af1), 'o')
plt.grid()
plt.ylabel('after 1')
#
plt.subplot(414)
plt.plot(ang, reg_count_af2/max(reg_count_af2), 'o')
plt.grid()
plt.ylabel('after 2')


# saving the residency
#np.savetxt(Nam+'/x_z.txt', x_z)
#np.savetxt(Nam+'/y_z.txt', y_z)

		
################################################ end of zones
print('###########')	

################################################# ADJUSTING the Regions
################################################# 
################################################# 
################################################# 
########## indices
ind_be_e1 = find_nearest(t, t_adj[0])

ind_be1 = range(0, ind_be_e1, 1)
ind_du1 = range(ind_be_e1  + 1, ind_du_e, 1)


t_be1 = t[ind_be1[0]:ind_be1[-1]]
xx_be1 = xx[ind_be1[0]:ind_be1[-1]]
yy_be1 = yy[ind_be1[0]:ind_be1[-1]]
print(t_be1[-1])

t_du1 = t[ind_du1[0]:ind_du1[-1]]
xx_du1 = xx[ind_du1[0]:ind_du1[-1]]
yy_du1 = yy[ind_du1[0]:ind_du1[-1]]
print(t_du1[-1])

fig = plt.figure(110)
plt.subplot(131)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.plot(xx_be1, yy_be1, '.b')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(132)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.plot(xx_du1, yy_du1, '.b')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
#
plt.subplot(133)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
#
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.plot(xx_af, yy_af, '.b')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.axis('equal')
fig.savefig(Nam+'/small_circle_adj_res_'+str(No)+'_'+str(tr)+'.png')

#### ADJUSTING _ zones
##### zones before 
theta_be1 = np.zeros(len(xx_be1))
theta_du1 = np.zeros(len(xx_du1))

reg_be1 = np.zeros(len(xx_be1))
reg_du1 = np.zeros(len(xx_du1))


for ii in range(len(xx_be1)):
	theta_be1[ii] = math.atan2(yy_be1[ii], xx_be1[ii])
	for jj in range(len(th) - 1):
		if theta_be1[ii] > th[jj] and theta_be1[ii] <= th[jj+1]:
			reg_be1[ii] = jj + 1
			x_z[jj] = np.cos(0.5*(th[jj] + th[jj+1]))
			y_z[jj] = np.sin(0.5*(th[jj] + th[jj+1]))
	

reg_count_be1 = np.zeros(no)
for ii in range(no):
	ind = ii + 1
	reg_count_be1[ii] = len(reg_be1[reg_be1 == ind])

### zones during
for ii in range(len(xx_du1)):
	theta_du1[ii] = math.atan2(yy_du1[ii], xx_du1[ii])
	for jj in range(len(th) - 1):
		if theta_du1[ii] > th[jj] and theta_du1[ii] <= th[jj+1]:
			reg_du1[ii] = jj + 1
			x_z[jj] = np.cos(0.5*(th[jj] + th[jj+1]))
			y_z[jj] = np.sin(0.5*(th[jj] + th[jj+1]))
	

reg_count_du1 = np.zeros(no)
for ii in range(no):
	ind = ii + 1
	reg_count_du1[ii] = len(reg_du1[reg_du1 == ind])
															

###########################################
fig = plt.figure(71)
plt.subplot(141)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_be1)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('before')
#
plt.subplot(142)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_du1)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('during')
#
plt.subplot(143)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_af1)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('after 1')
#
plt.subplot(144)
alpha = np.linspace(0, 2*np.pi, num=1000)
R1 = 0.9
b_x = R1*np.cos(alpha)
b_y = R1*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
R2 = 1.1
b_x = R2*np.cos(alpha)
b_y = R2*np.sin(alpha)
plt.plot(b_x, b_y, 'r')
plt.scatter(x_z, y_z, marker='o', c='b', s=sc*reg_count_af2)
############
plt.plot(F1_x, F1_y, 'sg', markersize=12)
plt.plot(F2_x, F2_y, 'sg', markersize=12)
############
plt.axis('equal')
plt.title('after 2')
fig.savefig(Nam+'/small_circle_adj_zones_'+str(No)+'_'+str(tr)+'.png')

###########################################################
########################################################### Saving the zones count with modified before/during (first food encounter)
np.savetxt(Nam+'/reg_count_be1.txt', reg_count_be1)	
np.savetxt(Nam+'/reg_count_du1.txt', reg_count_du1)	

fig = plt.figure(111)
plt.subplot(211)
plt.plot(t_be1, (180/np.pi)*(theta_be1 + np.pi/4), 'b')
plt.plot(t_du1, (180/np.pi)*(theta_du1 + np.pi/4), 'r')
plt.plot(t_af, (180/np.pi)*(theta_af + np.pi/4), 'k')
#
plt.plot(TT_nz, LL_nz, 'og')
#
#plt.xlabel('time (min)')
plt.ylabel('angle from food 1 (deg)')
plt.grid()
plt.subplot(212)
plt.plot(t_be1, (180/np.pi)*(theta_be1 - np.pi/4 + np.pi), 'b')
plt.plot(t_du1, (180/np.pi)*(theta_du1 - np.pi/4 + np.pi), 'r')
plt.plot(t_af, (180/np.pi)*(theta_af - np.pi/4 + np.pi), 'k')
#
plt.plot(TT_nz, LL_nz, 'og')
#
plt.xlabel('time (min)')
plt.ylabel('angle from food 2 (deg)')
plt.grid()
fig.savefig(Nam+'/small_circle_adj_angles_'+str(No)+'_'+str(tr)+'.png')

np.savetxt(Nam+'/theta_be1.txt', theta_be1)
np.savetxt(Nam+'/theta_du1.txt', theta_du1)
np.savetxt(Nam+'/theta_af.txt', theta_af)	

###########################################
#################################################
#################################################
#################################################
################################################# END of ADJUSTING the Regions
f_r_be = np.fft.fft((theta_be))
f_r_du = np.fft.fft((theta_du))
f_r_af = np.fft.fft((theta_af))

N_be = len(f_r_be)
freq_be = np.arange(0, N_be -1)
ts_be = 60*np.mean(t_be[1:] - t_be[0:-1])
#print('1/ts_be:', 1/ts_be)
freq_be = 1.0*freq_be/(N_be*ts_be)

N_du = len(f_r_du)
freq_du = np.arange(0, N_du -1)
ts_du = 60*np.mean(t_du[1:] - t_du[0:-1])
#print('1/ts_du:', 1/ts_du)
freq_du = 1.0*freq_du/(N_du*ts_du)

N_af = len(f_r_af)
freq_af = np.arange(0, N_af -1)
ts_af = 60*np.mean(t_af[1:] - t_af[0:-1])
#print('1/ts_af:', 1/ts_af)
freq_af = 1.0*freq_af/(N_af*ts_af)

#plt.figure(51)
#plt.subplot(311)
#plt.loglog(freq_be, abs(f_r_be[:-1]))
#plt.xlim([0.001, 10])
#plt.grid()
#plt.subplot(312)
#plt.loglog(freq_du, abs(f_r_du[:-1]))
#plt.xlim([0.001, 10])
#plt.grid()
#plt.subplot(313)
#plt.loglog(freq_af, abs(f_r_af[:-1]))
#plt.xlim([0.001, 10])
#plt.grid()

#r = np.sin(np.linspace(0,3.14,100))
#t = np.linspace(0, 10, 100)
#sample_path = np.c_[r*(np.sin(t)+np.cos(t)), r*(np.cos(t)-np.sin(t))]/1.5


ind_ar_s = find_nearest(t, t_s)
ind_ar_e = find_nearest(t, t_e)
ind_ar = range(ind_ar_s, ind_ar_e, 1)
t_ar = t[ind_ar[0]:ind_ar[-1]]
xx_ar = xx[ind_ar[0]:ind_ar[-1]]
yy_ar = yy[ind_ar[0]:ind_ar[-1]]


downsample = 10
xx = xx[::downsample]
yy = yy[::downsample]

t_be = t_be[::downsample]
xx_be = xx_be[::downsample]
yy_be = yy_be[::downsample]

t_du = t_du[::downsample]
xx_du = xx_du[::downsample]
yy_du = yy_du[::downsample]

t_af1 = t_af1[::downsample]
xx_af1 = xx_af1[::downsample]
yy_af1 = yy_af1[::downsample]

t_af2 = t_af2[::downsample]
xx_af2 = xx_af2[::downsample]
yy_af2 = yy_af2[::downsample]

t_ar = t_ar[::downsample]
xx_ar = xx_ar[::downsample]
yy_ar = yy_ar[::downsample]

# no average
if mod2 == 0:
	if mod == 1: # before
		sample_path = np.c_[xx_be, yy_be]
		tt = t_be
	if mod == 2: # during
		sample_path = np.c_[xx_du, yy_du]
		tt = t_du
	if mod == 3: # after (40-42)
		sample_path = np.c_[xx_af1, yy_af1]
		tt = t_af1
	if mod == 4: # after (42-50)
		sample_path = np.c_[xx_af2, yy_af2]
		tt = t_af2
	if mod == 5: # after (40-50)
		sample_path = np.c_[xx_af, yy_af]	
		tt = t_af
	if mod == 6: # arbitrary
		sample_path = np.c_[xx_ar, yy_ar]	
		tt = t_ar				

	fig, ax = plt.subplots()
	####################################
	line, = ax.plot(sample_path[0,0], sample_path[0,1], "ro-")

	def connect(i):
	    start=max((i-5,0))
	    line.set_data(sample_path[start:i,0],sample_path[start:i,1])
	    plt.title('time: %f' %tt[i])
	    return line,


	alpha = np.linspace(0, 2*np.pi, num=1000)
	R1 = 0.9
	b_x = R1*np.cos(alpha)
	b_y = R1*np.sin(alpha)
	plt.plot(b_x, b_y, 'r')
	R2 = 1.1
	b_x = R2*np.cos(alpha)
	b_y = R2*np.sin(alpha)
	plt.plot(b_x, b_y, 'r')
	plt.plot(F1_x, F1_y, 'sg', markersize=12)
	plt.plot(F2_x, F2_y, 'sg', markersize=12)
	plt.axis('equal')

	ax.set_xlim(-1.1,1.1)
	ax.set_ylim(-1.1,1.1)
	ani = animation.FuncAnimation(fig, connect, np.arange(1, len(sample_path)), interval=1)

# average
if mod2 == 1:
	if mod == 1: # before
		sample_path = np.c_[xx_be, yy_be]
		tt = t_be
	if mod == 2: # during
		sample_path = np.c_[xx_du, yy_du]
		tt = t_du
	if mod == 3: # after (40-42)
		sample_path = np.c_[xx_af1, yy_af1]
		tt = t_af1
	if mod == 4: # after (42-50)
		sample_path = np.c_[xx_af2, yy_af2]
		tt = t_af2
	if mod == 5: # after (40-50)
		sample_path = np.c_[xx_af, yy_af]	
		tt = t_af
	if mod == 6: # arbitrary
		sample_path = np.c_[xx_ar, yy_ar]	
		tt = t_ar				

	fig, ax = plt.subplots()
	################################################
	#line, = ax.plot(sample_path[0,0], sample_path[0,1], "ro-")
	line2, = ax.plot(sample_path[0,0], sample_path[0,1], "c-")
	line3, = ax.plot(sample_path[0,0], sample_path[0,1], "ko")

	def connect(i):
	    #start=max((i-5,0))
	    start2=max((i-200,0))
	    end2=max((i,0))
	    #line.set_data(sample_path[start:i,0],sample_path[start:i,1])
	    line2.set_data(sample_path[start2:end2,0],sample_path[start2:end2,1])
	    line3.set_data(sample_path[i,0],sample_path[i,1])
	    plt.title('Time: %.2f' %tt[i])
	    return line2, line3


	alpha = np.linspace(0, 2*np.pi, num=1000)
	R1 = 0.9
	b_x = R1*np.cos(alpha)
	b_y = R1*np.sin(alpha)
	plt.plot(b_x, b_y, 'k')
	R2 = 1.1
	b_x = R2*np.cos(alpha)
	b_y = R2*np.sin(alpha)
	plt.plot(b_x, b_y, 'k')
	plt.plot(F1_x, F1_y, 'sg', markersize=12)
	plt.plot(F2_x, F2_y, 'sg', markersize=12)
	plt.setp(ax.xaxis.get_ticklabels(), visible=False)
	plt.setp(ax.yaxis.get_ticklabels(), visible=False)
	plt.xticks([])
	plt.yticks([])
	plt.axis('off')
	plt.axis('equal')
	ax.set_xlim(-1.1,1.1)
	ax.set_ylim(-1.1,1.1)
	ani = animation.FuncAnimation(fig, connect, np.arange(1, len(sample_path)), interval=1)	

	#ani.save(Nam + '/im.mp4')

#print('x_z:', x_z)
#print('y_z:', y_z)
plt.show()
