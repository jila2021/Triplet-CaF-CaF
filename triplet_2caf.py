# created by Dibyendu Sardar, on march-3, 2023
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, WhiteKernel, ConstantKernel as C
from symfuncs import *
from sklearn.model_selection import train_test_split
from numpy import asarray
from numpy import savetxt


cm1=1.0/2.19474631e5 #unit
data_array=np.loadtxt('ccsdt-energy.dat') #load data
Easymp = -273.15478 # ccsdt asymptote of CaF-CaF

#If true then extra symmetric equivalent training points are added to the training set
add_sympoints=True
#If true, the fit is symmetrized when evaluating
symmetrize=True

#For R<Rsym, symmetrically equivalent points are added to the training set. one can play with Rsym
Rsym=8

#Remove too high energy points
V = data_array[:,6]
V = V-Easymp 
inddel=np.where(V>np.amin(V)+0.2)[0] #one can change this number to take care repulsive barrier
V=np.delete(V,inddel)
data_array=np.delete(data_array,inddel,axis=0)

R13 = data_array[:,0] 
R24 = data_array[:,1]
R = data_array[:,2]
th1 = data_array[:,3]*np.pi/180
th2 = data_array[:,4]*np.pi/180
phi = data_array[:,5]*np.pi/180
Eccsdt = data_array[:,6]
V = Eccsdt-Easymp #scaled energy with respect to CaF-CaF threshold

x_data_Jac = np.asarray([R13,R24,R,th1,th2,phi]).T
y_data = V
mA = 40.078
mB = 18.998403
#Coordinate transformation from Jacobi to atomic distances
def invchangecoords2(coJ,mA,mB):
	coordsnew=np.zeros(np.shape(coJ))
	r13=coJ[:,0]
	r24=coJ[:,1]
	R=coJ[:,2]
	th1=np.pi-coJ[:,3]
	th2=coJ[:,4]
	cth=np.cos(th2)
	sth=np.sin(th2)
	cth1=np.cos(th1)
	sth1=np.sin(th1)
	phi=coJ[:,5]
	cphi=np.cos(phi)
	sphi=np.sin(phi)
	r34=mA/(mB+mA)*np.sqrt((r24*sth*cphi-r13*sth1)**2+r24**2*sth**2*sphi**2+((mB+mA)/mA*R-r13*cth1-r24*cth)**2)
	r12=mB/(mB+mA)*np.sqrt((r24*sth*cphi-r13*sth1)**2+r24**2*sth**2*sphi**2+((mB+mA)/mB*R+r13*cth1+r24*cth)**2)
	r14=np.sqrt((mB/(mB+mA)*r24*sth*cphi+mA/(mB+mA)*r13*sth1)**2+(mB/(mB+mA))**2*r24**2*sth**2*sphi**2+(R-mA/(mB+mA)*r13*cth1+mB/(mB+mA)*r24*cth)**2)
	r23=np.sqrt((mA/(mB+mA)*r24*sth*cphi+mB/(mB+mA)*r13*sth1)**2+(mA/(mB+mA))**2*r24**2*sth**2*sphi**2+(R+mB/(mB+mA)*r13*cth1-mA/(mB+mA)*r24*cth)**2)
	coordsnew[:,0]=r12	
	coordsnew[:,1]=r13
	coordsnew[:,2]=r23
	coordsnew[:,3]=r14
	coordsnew[:,4]=r24
	coordsnew[:,5]=r34
	return coordsnew

x_data=1/invchangecoords2(x_data_Jac,mA,mB)

#Add symmetrically equivalent training points
if add_sympoints==True:
    print('Number of points before adding extra symmetric points:', len(y_data))
    extrapoints=np.where(R<Rsym)[0]
    extrax=perm2(x_data[extrapoints,:])
    extray=y_data[extrapoints]	
    x_data=np.concatenate([x_data,extrax])
    y_data=np.concatenate([y_data,extray])
    print('Number of points after adding extra symmetric points:',len(y_data))


#splitting data set for training and test part    
x_training, x_test, y_training, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
Ntest = len(x_test)    
#Define kernel
kernel =  Matern(length_scale=[1.0,1.0,1.0,1.0,1.0,1.0], length_scale_bounds=(1e-5, 1e5), nu=2.5) + WhiteKernel(noise_level=0.1)
#Construct GP fit
gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',n_restarts_optimizer=4)
model = gp.fit(x_training, y_training)

#Display trained kernel and marginal Log-likelihood
print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f", gp.log_marginal_likelihood(gp.kernel_.theta))

y_pred = gp.predict(x_test)
y_test = y_test.reshape(-1,1)     # reshape them because of course you do
y_pred = y_pred.reshape(-1,1)
y_diff = y_test-y_pred
rms = np.sqrt(sum(y_diff*y_diff)/Ntest)/cm1
print(rms)

