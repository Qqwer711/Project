import numpy as np

q = np.array([1,2,3])
q2=np.array([4,5,6])
qv = np.vstack((q,q2))
print("ซึ่งการใช้ v stack จะทำให้ สามารถนำ arraay มาต่อแถวกันได้ qv== \n", qv)


q3 = np.array([[1], [2], [3]])
q4 = np.array([[4], [5], [6]])
qv2 =np.vstack((q3,q4))
print( "แบบนี้จะเห็นภาพได้ชัดเจนกว่า qv2 == \n ",qv2)
