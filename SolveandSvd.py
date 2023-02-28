
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

trainLen = 2000
testLen = 20006
initLen = 100

Num = input("Pick files you want to test \n 1 = MackeyGlass_t17.txt \n 2 = HENON.DAT \n 3 = LORENZ.DAT \n 4 = MackeyGlass_t30.txt \n 5 = ROSSLER.DAT\n ")
data = A = np.array([],dtype='int64')
if Num == '1':
    data= np.loadtxt('MackeyGlass_t17.txt')

elif Num =='2' :
    data = np.loadtxt('HENON.DAT')
        
elif Num =='3':
    data = np.loadtxt('LORENZ.DAT')
      
elif Num =='4':
    data = np.loadtxt('MackeyGlass_t30.txt') 
    
elif Num =='5':
    data= np.loadtxt('ROSSLER.DAT')
      

        
 
inSize = outSize = 1
resSize = 1000
a = 0.3 
np.random.seed(42)
Win = (np.random.rand(resSize,1+inSize) - 0.5) * 1
W = np.random.rand(resSize,resSize) - 0.5 

print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
W *= 1.25 / rhoW
X1 = np.zeros((1+inSize+resSize,trainLen-initLen))
X2 = np.zeros((1+inSize+resSize,trainLen-initLen))
Yt = data[None,initLen+1:trainLen+1] 
x1 = np.zeros((resSize,1))
x2 = np.zeros((resSize,1))
for t in range(trainLen):
    u = data[t]
    x1 = (1-a)*x1 + a*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x1 ) )
    if t >= initLen:
        X1[:,t-initLen] = np.vstack((1,u,x1))[:,0]

for t in range(trainLen):
    u = data[t]
    x2 = (1-a)*x2 + a*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x2 ) )
    if t >= initLen:
        X2[:,t-initLen] = np.vstack((1,u,x2))[:,0]

        
reg = 1e-8  
# using solve 
Wout = linalg.solve( np.dot(X1,X1.T) + reg*np.eye(1+inSize+resSize), 
    np.dot(X1,Yt.T) ).T
# using svd
A = X2@X2.T + reg*np.eye(1+inSize+resSize)
b = X2@Yt.T
f,s,v = np.linalg.svd(A)
c = f.T@b
w = np.diag(1/s)@c
WoutSolve = (v.conj().T@w).T

Y1 = np.zeros((outSize,testLen))
Y2 = np.zeros((outSize,testLen))
u1 = data[trainLen]
u2 = data[trainLen]
for t in range(testLen):
    x1 = (1-a)*x1 + a*np.tanh( np.dot( Win, np.vstack((1,u1)) ) + np.dot( W, x1 ) )
    y = np.dot( Wout, np.vstack((1,u1,x1)) )
    Y1[:,t] = y
    u1 = y
   
for t in range(testLen):
    x2 = (1-a)*x2 + a*np.tanh( np.dot( Win, np.vstack((1,u2)) ) + np.dot( W, x2 ) )
    y = np.dot( WoutSolve, np.vstack((1,u2,x2)) )
    Y2[:,t] = y
    u2 = y

errorLen = 500
mseSolve = sum( np.square( data[trainLen+1:trainLen+errorLen+1] - 
    Y1[0,0:errorLen] ) ) / errorLen
mseSvd= sum( np.square( data[trainLen+1:trainLen+errorLen+1] - 
    Y2[0,0:errorLen] ) ) / errorLen
print("mseSolve  == ",mseSolve)
print("mseSvd  == ",mseSvd)

 

