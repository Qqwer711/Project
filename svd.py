import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 
 
A = np.array([[3,4,3],[1,2,3],[4,2,1]])
Ainv = linalg.inv(A)
u,s,v = np.linalg.svd(A)
b = np.array([[1,2,3],[1,3,5],[2,3,1]])

Q =  linalg.solve(A,b)
print( " Q == " , Q )
u,s,v = np.linalg.svd(A)
c = u.T@b
w = np.diag(1/s)@c
x = v.conj().T@w
print()
print ( "x == ",x)
print()

