#Please refer to the README File for the full formulas and explanation

import time #To Calculate the execution time
import numpy as np #To deal with Math
import pandas as pd #To handle the data's part
import scipy.optimize as opt


strt = time.time() #Calculate the time


#------------------------------------------------------------

def sigmoid(z):
    return 1/(1+np.exp(-z) )


#------------------------------------------------------------

def costfunc(thetas,x,y):  
    
    thetas = np.matrix(thetas)
    x = np.matrix(x)
    y = np.matrix(y)
    
    first = np.multiply( -y ,np.log( sigmoid( np.dot(x, thetas.T) )  )   )
    second = np.multiply( (1-y)  , np.log(  1 - ( sigmoid( np.dot(x, thetas.T) )  )   )  ) 
    c = np.sum( first - second ) / len(x)
    
    return c


#------------------------------------------------------------


def Gradientdes( thetas, x, y):

    thetas = np.matrix(thetas)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(thetas.ravel().shape[1])

    grd =  np.zeros(parameters)
    pred = sigmoid( x.dot( thetas.T ) )
    
    for e in range(parameters):
        
         term = np.multiply( ( pred - y) ,x[: , e] )  
         grd[e] =  ( 1/len(x) )     *   np.sum(term)     
        
    return grd


#------------------------------------------------------------



def predc(thetas,x):
    classif = sigmoid( x * thetas.T )
    e = [ 1 if i >= 0.5 else 0 for i in classif ]
    return e        
        



#------------------------------------------------------------




#Import the Dataset file, The txt file
datas = pd.read_csv(r'C:\Users\NADY\Desktop\dsc.txt', header = None , names=['Xdata0','Xdata1','Ydata'] ) #Write your path here

datas.insert(0,'Ones',1)


#We have the data, So, I want to choose specific columns 
cols = datas.shape[1]
x = datas.iloc[:,0:cols-1]
y = datas.iloc[:,cols-1:cols]


x = np.array(x.values) 
y = np.array(y.values)
thetas =np.zeros( x.shape[1] )

#--------------------

print('\n The Cost Function before optimization',costfunc(thetas, x, y),'\n')

results = opt.fmin_tnc(func = costfunc, x0 = thetas , fprime= Gradientdes, args=(x,y)  )
print(results)


#--------------------

endt = time.time()

costafteropti = costfunc(results[0], x, y)
print('\nThe Cost Function before optimization\n',costafteropti)

newvalues = predc( np.matrix( results[0] ) , x)
print( '\nThe Expected Values \n', newvalues,'\n' )

#--------------------

correctansrs = [ 1 if (a == 1 and b == 1) or (a == 0 and b == 0) else 0 for (a,b) in zip(y,newvalues)]
print('The Correct Answers are 1 wrong are 0\n',correctansrs)

accuracy = (sum(correctansrs) * 100 )/len(correctansrs)

print( f'\nacc =  {accuracy}% \n' )



print(f'\nThe Execution Time is : {endt-strt} Seconds' )



