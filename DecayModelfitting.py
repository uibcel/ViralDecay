import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
import sys


######## plot settings
plt.rcParams.update({'font.size': 14})

######## DEFs
# The model 
def dUdt(U,t,kab,kc):	#kc=kac=kbc
	
	A,B=U
	
	dlgAdt = -kab -kc
	dlgBdt = -kc + (kab*np.exp(A)/np.exp(B))
		
	return [dlgAdt, dlgBdt]

# Running the model with input	
def ODEmodel(params,t):
	kab = params['kab']
	kc = params['kc']
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
	y = odeint(dUdt,[A0.value,B0.value],time,args=(kab,kc)) 
	if time_flag==1:
		#delete first row from y (corresponding to y(t0))
		y_return=np.delete(y,0,0)
	else:
		y_return=y
	
	return y_return

# Find the residuals    
def residual(params, t, data):
	uncertainty=1
	model = ODEmodel(params,t)
	return ((data-model)/uncertainty) 

# Find B observed (vlp - mpn). Note that B ca= VLP when A/B is small so even if mpn data are lacking, this may not
# be critical.
def get_bobs(vlp,mpn): 
	out=vlp
	c=0
	for i in vlp:
		if mpn[c]>0:
			out[c]=vlp[c]-mpn[c]
		c+=1
	
	return(out)


####### input

if "-obs" in sys.argv:			# fileneame without extension (foo for foo.txt)
	pos=sys.argv.index("-obs")
	obs_file=sys.argv[pos+1]

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
	


####### read in data (e.g. Winter21_V1_time_MPN_VLP_mod)
obs=np.loadtxt(obs_file+".txt")
t_obs=obs[:,0]
mpn_obs=obs[:,1]
vlp_obs=obs[:,2]
A_obs=mpn_obs 
B_obs=get_bobs(vlp_obs, mpn_obs)	# B_obs=vlp-mpn when mpn is available, otherwise =vlp



##### convert to log
log_mpn_obs = np.log(mpn_obs)
log_vlp_obs = np.log(vlp_obs)
log_A_obs = np.log(A_obs)
log_B_obs = np.log(B_obs)



# stack
data=np.stack((log_A_obs,log_B_obs),axis=1)


######### Define parameters
params = Parameters()
params.add('kab', value=0.25, min=0, max=1)
params.add('kc', value=0.05, min=0, max=1)
params.add('A0',value=np.log(A0), min=np.log(1E+3), max=np.log(1E+9), vary=True)
params.add('B0',value=np.log(B0), min=np.log(1E+3), max=np.log(1E+9), vary=True)
params.add('t_0', value=0, vary=False)


#print(np.log(A0))
#print(np.log(B0))


if (False):
	########### Run similation with initial parameter guess
	time=np.linspace(params['t_0'],40,200)
	#compute ODE solution with initial parameter guess
	sol1=ODEmodel(params,time)



	# modelled VLP
	log_VLP_sol1=np.log ( np.exp(sol1[:, 0])+np.exp(sol1[:, 1]) )



	# plot results
	plt.plot(time, sol1[:, 0],'r:', label='A-Inf')
	plt.plot(time, sol1[:, 1], 'g:', label='B')
	#plt.plot(time, sol1[:, 2], 'b:', label='C')
	#plt.plot(time, log_VLP_sol1,'k:', label='VLP')
	plt.plot(t_obs,log_mpn_obs,'ro',label='mpn_obs')
	plt.plot(t_obs,log_A_obs,'k1',label='A_obs')
	plt.plot(t_obs,log_vlp_obs,'ko',label='vlp_obs')
	plt.plot(t_obs,log_B_obs,'g1', label='B_obs')
	plt.title(obs_file+"_"+runid)

	plt.legend()
	plt.xlabel('time')




if (True):

	# time (40 days)
	time=np.linspace(params['t_0'],40,200)

	############## optimize parameter values
	out = minimize(residual, params, args=(t_obs, data), nan_policy='omit')


	############ print output
	print ("RID:"+runid)
	out.params.pretty_print()



	#compute solution with optimized parameter values
	sol2=ODEmodel(out.params,time)
	
		

	# modelled VLP
	log_VLP_sol2=np.log ( np.exp(sol2[:, 0])+np.exp(sol2[:, 1]) )

	
	#plot	
	#plt.plot(t_obs,log_mpn_obs,'ro',label='mpn_obs')
	plt.plot(t_obs,log_A_obs,'ro',label='A_obs')
	#plt.plot(t_obs,log_vlp_obs,'ko',label='vlp_obs')
	plt.plot(t_obs,log_B_obs,'ko', label='B_obs')
	
	plt.plot(time,sol2[:,0],'r-',label='A_fitted')
	plt.plot(time,sol2[:,1],'g-',label='B_fitted')
	plt.plot(time, log_VLP_sol2,'k:', label='VLP')
	
	if (legend):
		plt.legend(loc=lp)

	plt.xlabel('Time (days)')
	plt.ylabel('log Abundance')

	if (save_plot):
		plt.savefig('DDecayPlot_'+obs_file+'.png', dpi=dpi)
	else:
		plt.show()
	
	

else:
	a=0
	plt.show()	



# REFs
# https://apmonitor.com/pdc/index.php/Main/SolveDifferentialEquations
#https://stackoverflow.com/questions/51808922/how-to-solve-a-system-of-differential-equations-using-scipy-odeint
#https://www.youtube.com/watch?v=MM3cBamj1Ms
#https://www.youtube.com/watch?v=peBOquJ3fDo&t=16s
# https://www.youtube.com/watch?v=1H-SdMuJXTk #curve_fit
#https://lmfit.github.io/lmfit-py/lmfit.pdf # minimize long pdf

