#!/usr/bin/env python
# coding: utf-8

from gurobipy import *
import numpy as np
import math
import random
import csv
#define the values of torque,Geomagnetic Field,M

import csv
file1 = open('FinalFinal.xlsx', 'a')   #Creating a File
file1 = csv.writer(file1)
for i in range(0,2000):
    b = np.random.uniform(-2.5*pow(10,-5),2.5*pow(10,-5),size=(3)) #magenetic field
    #print("B:",b)
    rand_vec = np.random.uniform(-3.5*pow(10,-5),3.5*pow(10,-5),size=(3))
    rand_vec = rand_vec/np.linalg.norm(rand_vec)
    tau = np.cross(b, rand_vec)
    #print("T:",tau)
    M = random.uniform(0.8,1.2)
    #print("print M:",M)
    # Call the function and store the result in a variable
    model =Model('MagneticMoment')
    # Define variables
    mx = model.addVar(lb=-M, ub=M, name='mx')
    my = model.addVar(lb=-M, ub=M, name='my')
    mz = model.addVar(lb=-M, ub=M, name='mz')
    # Define constraints
    model.addConstr(tau[0] == b[2]*my - b[1]*mz, name='constr1')
    model.addConstr(tau[1] == b[0]*mz - b[2]*mx, name='constr2')
    model.addConstr(tau[2] == b[1]*mx - b[0]*my, name='constr3')
    # Define objective
    obj_fn=pow(mx,2) +pow(my,2) +pow(mz,2)
    model.setObjective(obj_fn,GRB.MINIMIZE)
    # Solve the problem
    model.optimize()
    # Return the solution

    print('Objective function value:%f'%model.obj_val)
    m_outputs=[]
    for v in model.getVars():
        m_outputs.append(v.x)
        #print(v.varName, v.x)

    data=[b[0],b[1],b[2],tau[0],tau[1],tau[2],M,m_outputs[0],m_outputs[1],m_outputs[2]]
    print(data)
    file1.writerow(data)

f = open('FinalDataSet1.xlsx', 'r')
file_contents = f.read()
print (file_contents)
f.close()
#print('All records inserted successfully !')

#m_star = np.array([v.x for v in model.getVars()])
#tau_obtained = np.cross(m_star, b)
#print(tau)
#print(tau_obtained)

#print(np.dot(m_star,b))
#print("Magnetic moment: ", magnetic_moment)




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




