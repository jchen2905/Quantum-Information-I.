#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
from numpy import array, dot, pi
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvals


# In[74]:


#Assignment 1.2: print Pauli operations using Python


# In[75]:


def pauli_X():
    op=sp.csr_matrix([[0,1],[1,0]])
    return op

def pauli_Z():
    op=sp.csr_matrix([[1,0],[0,-1]])
    return op

def pauli_Y():
    op=sp.csr_matrix([[0,-1j],[1j,0]])
    return op


# In[76]:


#Assignment 1.2 solution


# In[77]:


print(pauli_X(), pauli_Y(), pauli_Z())


# In[78]:


#Assignment 1.3: Compute the norm of c = [1+i, 3−2i]


# In[79]:


c = np.array([1 + 1j, 3 - 2j])

c_norm = np.sqrt(np.dot(np.conj(c).T,c))


# In[80]:


#Assignment 1.3 solution


# In[81]:


print(c, c_norm)


# In[82]:


#Assignment 1.4: Obtain the eigenvalues and eigenvectors of example matrix


# In[83]:


H = np.array([[2,1-1j],[1+1j,3]])

eigenvalues, eigenvectors = np.linalg.eig(H)


# In[ ]:


#Assignment 1.4 Solution


# In[84]:


print("H")
print(H)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)


# In[ ]:




