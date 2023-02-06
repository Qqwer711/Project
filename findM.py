#การหา column and row  เท่าที่ผมหามาวิธีนี้น่าจะชัดเจนที่สุด
import numpy as np

MatrixR = np.random.randint(1,10, size=(5,5))

findCR = np.where(MatrixR == 7)

print(MatrixR,"\n",findCR)

##ผมได้เขียนคำอธิบายไว้เเล้วใน pdf นะครับ