# -*- coding: utf-8 -*-
'''
Created on Sat Oct 15 2015

Modified on Fri Jun 16 2017

@author: David Hsu

module: LA.py

'''

import numpy as np

def upperTri(M):
    # M : m by n matrix
    m,n = M.shape
    U = np.array(M)
    rank = 0 # rank of M
    numRowExchange = 0 # number of row-exchange

    # iterations for Gaussian Elimination
    i = 0
    j = 0
    while (i<m and j<n):
        # if the row pivot is zero, find other row with non-zero pivot to switch
        if(U[i,j]==0):
            # finding starts from the next row
            k=i+1
            while(k<m):
                if(U[i,j]!=0): # find a non-zero entry as pivot, switch and then break
                    tmpRow = U[k,:]
                    U[k,:] = U[i,:]
                    U[i,:] = tmpRow
                    # row exchange counts
                    numRowExchange += 1
                    break
                else: # find the next row
                    k += 1

        # pivot for row i is decided
        pivot = U[i,j]
        if(pivot == 0): # if pivot is still zero, try U[i,j+1] as a pivot for row i 
            j +=1
        else: # non-zero pivot for row i is found, now start the elimination part
            # non-zero pivot, so rank counts
            rank +=1
            for k in range(i+1,m):
                factor = U[k,j]/pivot
                U[k,:] = U[k,:]-factor*U[i,:]
            # ready for the next row (i+1) and pivot shift right (j+1)
            i += 1
            j += 1

    # return row echelon form (upper triangular matrix), rank, and numRowExchange
    return (U,rank,numRowExchange)

def rref(M):
    # M : m by n matrix
    m,n = M.shape
    R = np.array(M)
    rank = 0 # rank of M
    numRowExchange = 0 # number of row-exchange

    # iterations for Gaussian Elimination
    i = 0
    j = 0
    while (i<m and j<n):
        # if the row pivot is zero, find other row with non-zero pivot to switch
        if(R[i,j]==0):
            # finding starts from the next row
            k=i+1
            while(k<m):
                if(R[i,j]!=0): # find a non-zero entry as pivot, switch and then break
                    tmpRow = R[k,:]
                    R[k,:] = R[i,:]
                    R[i,:] = tmpRow
                    # row exchange counts
                    numRowExchange += 1
                    break
                else: # find the next row
                    k += 1

        # pivot for row i is decided
        pivot = R[i,j]
        if(pivot == 0): # if pivot is still zero, try U[i,j+1] as a pivot for row i 
            j +=1
        else: # non-zero pivot for row i is found, now start the elimination part
            # non-zero pivot, so rank counts
            rank +=1
            # set pivot to 1
            R[i,:] = R[i,:]/pivot
            for k in range(i):
                factor = R[k,j]
                R[k,:] = R[k,:]-factor*R[i,:]
            
            for k in range(i+1,m):
                factor = R[k,j]
                R[k,:] = R[k,:]-factor*R[i,:]
                
            # ready for the next row (i+1) and pivot shift right (j+1)
            i += 1
            j += 1

    # return row echelon form (upper triangular matrix), rank, and numRowExchange
    return (R,rank,numRowExchange)

def det(M):
    # check dimension of M
    m,n = M.shape
    if (m!=n):
        print('It is not a square matrix, no determinant define !')
    else :
        # U : upper triangular (row echelon matrix)
        # nre : number of row-exchange
        U,rank,nre = upperTri(M)
        
        # the determinant of M is the product of U[i,i]
        determinant = 1;
        for i in range(m):
            determinant = determinant*U[i,i]

        # number of row-exchange will decide the sign
        if (nre%2==1): # nre is odd
            determinant = -1*determinant
        # determinant = LA.det(M)
        return determinant

def inv(M):
    # check dimension of M
    m,n = M.shape
    if (m!=n):
        print('It is not a square matrix, no inverse define !')
    else :
        augmentedM = np.zeros((n,2*n),float)
        augmentedM[:,:n] = M[:,:]
        for i in range(n):
            augmentedM[i,n+i]=1

        # reduced row echelon form               
        R, rank, nre = rref(augmentedM)
        # check if M is full rank or not
        fullRank=1;
        for i in range(n):
            if(R[i,i]==0):
                fullRank=0
                break;
        
        if (fullRank==0):
            print('The matrix is singular. Try pseudo-inverse!')
        else:
            inverse = np.array(R[:,n:])
            return inverse

def solve(A,b):
    m,n = A.shape

    # augmentedA = [A|b]
    augmentedA = np.zeros((m,n+1),float)
    augmentedA[:,:n] = A
    augmentedA[:,n] = b
    R, rankAb, nre = rref(augmentedA)

    # compute the rank of A
    rankA = 0
    i = 0
    j = 0
    while (i<m and j<n):
        pivot = R[i,j]
        if(pivot==0):
            j += 1
        else:
            rankA += 1
            i += 1
            j += 1
    print(rankA)
    print(rankAb)
    # rankA == rankAb == n : only solution
    # rankA == rankAb < n  : infinite solutions
    # rankAb > rankA : no solution
    if (rankA == rankAb):
        if (rankA==n):
            x=np.array(R[:n,n])
            return x
        else: # rankA<n
            print('infinite solutions')
    else: # rankAb>rankA
        print('no solution')
    
    
