#This folder contains the necessary files to calculate the potential energy surface of the triplet
CaF-CaF.

#The codes are written in python having version 3.6. 

#The training data sets is in atomic coordinates and the code converted the configurations 
in inverse atomic coordinates that we mentioned in our manuscript. 
The energy is expressed in atomic unit. 
 
#We set zero of the energy at CaF-CaF threshold. Inside the code, 
we put a condition to exclude the high energy data points with respect to the CaF-CaF threshold. 
But, one can set the repuslive barrier of the barrier accordingly.

#The auxiliary python function symfuncs.py is used for the symmetrization.

#We divide the total data points randomly into training and test sets, comprising  80% and 20% of the data, respectively.  
 Then we fit these training set of data by GP regression.

#The code triplet_2caf.py is used for making the fit of the surface using the training data set
& it is run by the command: 
python triplet_2caf.py 

#To check the accuracy of the GP fit or to determine the uncertainty of the calculation, we calculate the
 root mean squared error (RMSE) on remaining 20% test data.
