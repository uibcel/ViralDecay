import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
import sys



######## Functions
# function that returns dU/dt, 
def dUdt(U,t,kab,kac,kbc):	#kc=kac=kbc
	
	A,B=U
	
	# the function takes log values, but for dlgBdt calculations
	# we need non-log values, hence np.exp(A) an np.exp(B)
	dlgAdt = -kab -kac
	dlgBdt = -kbc + (kab*np.exp(A)/np.exp(B))
		
	return [dlgAdt, dlgBdt]
	
def ODEmodel(params,t):
	kab = params['kab']
	kac = params['kac']
	kbc = params['kbc']
	A0 = params['A0']
	B0 = params['B0']
	t0=params['t_0']
	
	#make sure the simulation time starts with t0,
	#otherwise we would use wrong initial conditions
	time_flag=0
	time=t
	if t[0]!=t0:
	#append t0 to t	
		time_flag=1
		time=np.insert(t,0,t0.value) 
	y = odeint(dUdt,[A0.value,B0.value],time,args=(kab,kac,kbc)) 
	if time_flag==1:
		#delete first row from y (corresponding to y(t0))
		y_return=np.delete(y,0,0)
	else:
		y_return=y
	
	return y_return

def help():
	print("-A initial number of particles in stat A, default 1E+5")
	print("-B initial number of particles in stat B, default 1E+7")	
	print("-id runid, default '' ")
	print("-s save plot")
	print("-l show legend in plot")
	print("-kab rate constant: State A->B, default 0.15")
	print("-kbc rate constant: State B->C, default 0.05")
	print("-kac rate constant: State A->C, default 0.05")


print (sys.argv)
"-h" in sys.argv

####### input
if "-h" in sys.argv:
	help()
	exit()

if "-A" in sys.argv:			# initial number in state A
	pos=sys.argv.index("-A")
	A0=float(sys.argv[pos+1])
else:
	A0=1E+5
	
if "-B" in sys.argv:			# initial number in state B
	pos=sys.argv.index("-B")
	B0=float(sys.argv[pos+1])
else: 
	B0=1E+7

if "-id" in sys.argv:			# runid
	pos=sys.argv.index("-id")
	runid=sys.argv[pos+1]
else: 
	runid=""

if "-s" in sys.argv:			# save plot
	save_plot=True
else: 
	save_plot=False


if "-kab" in sys.argv:			# initial number in state B
	pos=sys.argv.index("-kab")
	kab=float(sys.argv[pos+1])
else: 
	kab=0.15
	
if "-kac" in sys.argv:			# initial number in state B
	pos=sys.argv.index("-kac")
	kac=float(sys.argv[pos+1])
else: 
	kac=0.05

if "-kbc" in sys.argv:			# initial number in state B
	pos=sys.argv.index("-kbc")
	kbc=float(sys.argv[pos+1])
else: 
	kbc=0.05


if "-l" in sys.argv:			# legend true of false
	legend=True
else: 
	legend=False

if "-lp" in sys.argv:			# legend position
	pos=sys.argv.index("-lp")
	lp=int(sys.argv[pos+1])
else: 
	lp='best'
	

if "-dpi" in sys.argv:			# plot resolution
	pos=sys.argv.index("-dpi")
	dpi=int(sys.argv[pos+1])
else: 
	dpi=100

print ("Runsettings")
print ("Runid: "+runid)
print ("kab: "+str(kab))
print ("kac: "+str(kac))
print ("kbc: "+str(kbc))
print ("A0: "+str(A0))
print ("B0: "+str(B0))
print ("save_plot: "+str(save_plot))



######### Define parameters
params = Parameters()
params.add('kab', value=kab)
params.add('kac', value=kac)
params.add('kbc', value=kbc)
params.add('A0',value=np.log(A0))
params.add('B0',value=np.log(B0))
params.add('t_0', value=0, vary=False)


######## plot settings
plt.rcParams.update({'font.size': 15})

########### Run similation with parameters
time=np.linspace(params['t_0'],20,200)

#compute ODE solution
sol1=ODEmodel(params,time)


# modelled VLP
log_VLP_sol1=np.log (np.exp(sol1[:, 0])+np.exp(sol1[:, 1]) )


# straight line between vlp start-stop
t_vlp_line=[time[0],time[-1]]
y_vlp_line=[log_VLP_sol1[0],log_VLP_sol1[-1]]


# plot results
plt.plot(time, sol1[:, 0],'r-', label='A')
plt.plot(time, sol1[:, 1], 'g-', label='B')
#plt.plot(t_vlp_line, y_vlp_line, 'b:', label='vlp-line')
plt.plot(time, log_VLP_sol1,'k:', label='A+B')
#plt.title(runid)

if (legend):
	plt.legend(loc=lp)

plt.xlabel('Time')
plt.ylabel('log Abundance')

if (save_plot):
	plt.savefig('Out'+str(runid)+'.png', dpi=dpi)
else:
	plt.show()
	
	



# REFs
# https://apmonitor.com/pdc/index.php/Main/SolveDifferentialEquations
#https://stackoverflow.com/questions/51808922/how-to-solve-a-system-of-differential-equations-using-scipy-odeint
#https://www.youtube.com/watch?v=MM3cBamj1Ms
#https://www.youtube.com/watch?v=peBOquJ3fDo&t=16s
# https://www.youtube.com/watch?v=1H-SdMuJXTk #curve_fit
#https://lmfit.github.io/lmfit-py/lmfit.pdf # minimize long pdf

