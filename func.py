# -*- coding: utf-8 -*-
'''
Created on Fri Nov 04 2016

Modified on Wed Jun 27 2017

@author: David Hsu

module: func.py

'''

import time
import math
import numpy as np

class ComplexNumber:
    def __init__(self,re=0.0,im=0.0):
        self.re = re
        self.im = im
    def __repr__(self):
        complexNum = str(self.re)+' + j*'+str(self.im)
        return complexNum
    def __add__(self,other):
        ans = ComplexNumber()
        ans.re = self.re + other.re
        ans.im = self.im + other.im
        if(ans.im==0):
            return ans.re
        else:
            return ans
    def __mul__(self,C):
        ans = ComplexNumber()
        ans.re = self.re*C.re - self.im*C.im
        ans.im = self.re*C.im + self.im*C.re
        if(ans.im==0):
            return ans.re
        else:
            return ans
    def abs(self):
        return math.sqrt(self.re**2+self.im**2)

class ListNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.Lchild = None
        self.Rchild = None
        


def isNum(strData):
    try:
        num = float(strData)
    except ValueError:
        return False
    else:
        return True

def str2num(string): 
    try:
        num = float(string)
    except ValueError:
        return float('nan')  # checked by math.isnan()
    return num

def floor(num):
    if(isNum(num)):
        floorNum = int(num)
        if(num<0):
            diff = float(num)-float(floorNum)
            if(diff!=0):
                floorNum = floorNum-1
        return floorNum
    else:
        return float('nan')    

def ceiling(num):
    if(isNum(num)):
        ceilingNum = int(num)
        if(num>0):
            diff = float(num)-float(ceilingNum)
            if(diff!=0):
                ceilingNum = celingNum+1
        return ceilingNum
    else:
        return float('nan')

def exp(x): # recursive exp function
    if(x>1):
        return exp(x-1)*exp(1)
    elif(x<-1):
        return exp(x+1)*exp(-1)
    else: # -1<=x<=1 --> Tyler expansion
        expX = 1
        cnt = 0
        term = 1.0
        while(cnt<30):
            cnt = cnt+1
            term = term*x/cnt
            expX = expX+term
        return expX

def ln(x): # Taylor :  ln(1+y)=y-y^2/2+y^3/3-y^4/4+y^5/5-...  and -1<=y<=1
    if(x<=0):
        return float('nan') # undefined
    else: # Taylor :  ln(1+y)=y-y^2/2+y^3/3-y^4/4+y^5/5-...  and -1<=y<=1
        y = x-1    # x = 1+y
        if(y>0.2): # if(x>1.2)
            return ln(x/1.2)+ln(1.2)
        elif(y<-0.2): # if(0<x<0.8)
            return ln(x*1.2)-ln(1.2)
        else: # -0.2<=y<=0.2 --> Tyler expansion
            lnX = 0
            cnt = 0
            term = -1.0
            while(cnt<35):
                cnt = cnt+1
                term = term*(-y)
                lnX = lnX+term/cnt
            return lnX

def power(a,x): # return a**x (a^x)
    if(float(int(x))==x):
        powerAx = 1
        if(x>0):
            for i in range(0,x):
                powerAx = powerAx*a
        else:
            for j in range(x,0):
                powerAx = powerAx/a
        return powerAx
    elif(a>0): # a^x = exp(x*ln(a)), ln(a) -> a>0
        lna = ln(a)
        return exp(x*lna)
    else: # if x is not an integer and a<0 --> power(a,x) is a complex number
        return float('nan')

def log(b,x): # log of x to base b
    if(b>0):
        return ln(x)/ln(b)
    else: # if(b<=0) --> undefined
        return float('nan')

def Bsort(L): # bubble sort
    if(isinstance(L,list) or isinstance(L,np.ndarray)):
        N = len(L)
        for i in range(N-1,0,-1):  # i = N-1, N-2,..., 2, 1
            for j in range(i):   # max(j) = N-2, N-3,...., 2, 1
                if(L[j]>L[j+1]): # then SWAP
                    tmp = L[j]
                    L[j] = L[j+1]
                    L[j+1] = tmp
    else:
        print('Invalid input.')       
    return

def Qsort(L,indexL,indexH): # quick sort (list L, 0, len(L)-1)
    if(indexL<indexH):
        pivot = L[indexL]
        i = indexL+1
        j = indexH
        while True:
            while(L[i]<=pivot and i<indexH): # find i that L[i]>pivot
                i = i+1
            while(L[j]>=pivot and j>indexL): # find j that L[j]<pivot
                j = j-1
            if(i<=j):
                tmp = L[i]
                L[i] = L[j]
                L[j] = tmp
            else:
                L[indexL] = L[j]
                L[j] = pivot
                break
        Qsort(L, indexL, j-1)
        Qsort(L, j+1, indexH)
    return

def gcd(a,b): #greatest common divider
    if(b!=0):
        return gcd(b,a%b)
    else:
        return a

def bin2dec(binary):
    if(isinstance(binary,str)):
        decimal = 0
        N = len(binary)
        # checking : integer part and fractional part
        digitInteger = N
        for i in range(N): # i=0~N-1
            if(binary[i]=='.'):
                digitInteger = i
        for di in range(0,digitInteger): # d = 0~digitInteger-1
            decimal = decimal + power(2,di)*float(binary[digitInteger-1-di])
        digitFraction = N-1-digitInteger
        if(digitFraction<0): # integer number
            return int(decimal)
        else: # deal with the fractinal part
            for df in range(1,digitFraction+1): # d = 1~digitFraction
                decimal = decimal + power(2,-df)*float(binary[digitInteger+df])
            return decimal
    else:
        print('It takes \'str\' input.')
        return

def dec2bin(decimal):
    if(isNum(decimal)):
        binary = ''
        integerPart = int(decimal)
        fractionPart = decimal-integerPart
        # integer part
        while(integerPart!=0):
            binary = str(integerPart%2)+binary
            integerPart = integerPart//2
        # fractional part
        if(fractionPart!=0):
            binary = binary+'.'
            cnt = 0
            while(fractionPart!=0):
                cnt = cnt+1
                fractionPart = fractionPart*2
                if(fractionPart>=1):
                    fractionPart = fractionPart-1
                    binary = binary + '1'
                else:
                    binary = binary + '0'
            #print(cnt) # 52
        return binary
    else:
        print('It takes a number.')
        return

def int2bin2c(num,digit): # 2's complement
    if(num>power(2,digit-1)-1 or num<-power(2,digit-1)):
        print('Overflow occurs !')
        return
    else:
        posBin = dec2bin(abs(num))
        binary = ''
        for i in range(len(posBin)):
            binary = binary + posBin[i]
        for j in range(digit-len(posBin)):
            binary = '0'+binary
        if(num>0):
            return binary
        else:
            # invertor
            binaryInv = ''
            for k in range(digit):
                if(binary[k]=='1'):
                    binaryInv = binaryInv + '0'
                if(binary[k]=='0'):
                    binaryInv = binaryInv + '1'
            # binary2c = binaryInv + 1
            binary2c = ''
            for m in range(digit-1,-1,-1):
                if(binaryInv[m]=='1'):
                    binary2c = '0'+binary2c
                else: # if(binaryInv[l]=='0'):
                    binary2c = '1'+binary2c
                    for n in range(m-1,-1,-1):
                        binary2c = binaryInv[n]+binary2c
                    break
            return binary2c

def bin2c2int(binary2c): # 2's complement
    if(binary2c[0]=='0'): # positive number
        return bin2dec(binary2c)
    else: #i f(binary2c[0]=='1'): # negative number
        # invertor
        binaryInv = ''
        for i in range(len(binary2c)):
            if(binary2c[i]=='1'):
                binaryInv = binaryInv + '0'
            if(binary2c[i]=='0'):
                binaryInv = binaryInv + '1'
        # binary = binaryInv + '1'
        binary = ''
        for m in range(len(binaryInv)-1,-1,-1):
            if(binaryInv[m]=='1'):
                binary = '0'+binary
            else: # if(binaryInv[m]=='0')
                binary = '1'+binary
                for n in range(m-1,-1,-1):
                    binary = binaryInv[n]+binary
                break
        return -bin2dec(binary)

def csvRead(fileName):
    file = open(fileName)
    data = file.readlines()
    row = len(data)
    col = len(data[0].strip().split(','))
    arr = np.zeros((row,col))
    for i in range(row):
        line = data[i].strip().split(',')
        for j in range(col):
            arr[i][j] = str2num(line[j])
    file.close()
    return arr

def csvReadNum(fileName):
    file = open(fileName)
    data = file.readlines()
    row = len(data)
    col = len(data[0].strip().split(','))
    arr = np.zeros((row,col))
    for i in range(row):
        line = data[i].strip().split(',')
        for j in range(col):
            arr[i][j] = str2num(line[j])
    file.close()
    # detect the rows with all 'NaN' entries
    deleteRows = []
    for i in range(row):
        cnt = 0
        for j in range(col):
            if(math.isnan(arr[i][j])):
                cnt = cnt + 1
        if(cnt==col):
            deleteRows.append(i)
            
    # detect the cols with all 'NaN' entries
    deleteCols = []
    for j in range(col):
        cnt = 0
        for i in range(row):
            if(math.isnan(arr[i][j])):
                cnt = cnt + 1
        if(cnt==row):
            deleteCols.append(j)

    # delete rows/cols
    arr = np.delete(arr, deleteRows, 0)
    arr = np.delete(arr, deleteCols, 1)
    return arr

def csvWrite(arr,fileName):
    m,n = arr.shape
    file = open(fileName,'w+')
    for i in range(m):
        for j in range(n):
            file.write(str(arr[i,j]))
            if(j==n-1):
                file.write('\n')
            else:
                file.write(',')

    file.close()
    return

'''
def isPrime(num):
    if(num<=1):
        return False
    else:
        prime = True
        sqroot = math.sqrt(num)
        bound = int(sqroot)+1
        for i in range(2,bound):
            if(num%i==0):
                prime = False
                break
        return prime

def primeFactorize(num):
    factors = []
    while(True):
        sqroot = math.sqrt(num)
        bound = int(sqroot)+1
        prime = True
        for i in range(2,bound):
            if(num%i==0):
                prime = False
                factors.append(i)
                num = num/i
                break
        if(prime):
            factors.append(int(num))
            num = num/num
        if(num==1):
            break
    return factors
'''

def isPrime(num):
    if(num==1):
        return False
    elif(num==2):
        return True
    elif(num%2==0):
        return False
    else:
        sqrt_num = math.sqrt(num)
        bound = int(sqrt_num)+1
        for i in range(3,bound,2):
            if(num%i==0):
                return False
        return True

def primeFactorize(num):
    factors = []
    while(True):
        prime = True
        if(num%2==0):
            prime = False
            factors.append(2)
            num = num/2
        else:
            sqroot = math.sqrt(num)
            bound = int(sqroot)+1
            for i in range(3,bound,2):
                if(num%i==0):
                    prime = False
                    factors.append(i)
                    num = num/i
                    break
        if(prime):
            factors.append(int(num))
            num = num/num
        if(num==1):
            break
    return factors

def nchoosek(n,k):
    if(n<k):
        return 0
    elif(k==0):
        return 1
    else: # n>=k
        #lookup = np.zeros((n+1,k+1),np.int64)
        lookup = []
        for i in range(n+1):
            lookup.append([0]*(k+1))
        # initial conditions
        for i in range(n+1):
            lookup[i][0] = 1
            lookup[i][1] = i
        for j in range(k+1):
            lookup[j][j] = 1
        # complete the talbe
        for i in range(1,n+1):
            for j in range(1,k+1):
                if(i>j):
                    lookup[i][j] = lookup[i-1][j] + lookup[i-1][j-1]
                else:
                    break
        #print(lookup)
        return lookup[n][k]

def fibonacci(n):
    f0 = 0
    f1 = 1
    if(n==0): # F(0)
        fib = f0
    elif(n==1): # F(1)
        fib = f1
    else: # F(2)...
        fib = f0+f1
        cnt = 2
        while(cnt<n):
            f0 = f1
            f1 = fib
            fib = f0+f1
            cnt += 1
    return fib

def allPermutation(arr,n):
    if(n==len(arr)):
        print(arr)
    else:
        for i in range(n,len(arr)):
            # swap index n(head), i
            temp = arr[i]
            arr[i] = arr[n]
            arr[n] = temp
            allPermutation(arr,n+1)
            # swap back to resume arr
            temp = arr[i]
            arr[i] = arr[n]
            arr[n] = temp

    
    
