import numpy as np
import pandas as pd

data=pd.read_csv('Enjoy_sport1.csv')
print("Dataset : ",data)

concepts=np.array(data.iloc[:,:-1])
target=np.array(data.iloc[:,-1])

print("Concepts : ",concepts)
print("Target : ",target)

def initialize(concepts):
    print("Initialization of specific_h and general_h")
    specific_h=['0']*len(concepts[0])
    print("Initial specific_h : ",specific_h)

    general_h=[['?' for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("Initial general_h : ",general_h)

    return specific_h,general_h

def learn(concepts,target):
    specific_h,general_h=initialize(concepts)
    for i,h in enumerate(concepts):
        print("Instance ",i+1," is ",h)
        if target[i]=="Yes":
            print("Instance is positive..")
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x] and i==0:
                    specific_h=concepts[0].copy()
                elif h[x]!=specific_h[x]:
                    specific_h[x]="?"
                    general_h[x][x]="?"
        if target[i]=="No":
            print("Instance is negative..")
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else:
                    general_h[x][x]='?'
    print("SB : ",i+1," is ",specific_h)
    print("GB : ",i+1," is ",general_h)

    indices=[i for i,val in enumerate(general_h) if val==['?','?','?','?','?','?']]

    for i in indices:
        general_h.remove(['?','?','?','?','?','?'])

    indices=[i for i,val in enumerate(general_h) if val==['?']*len(concepts[0])]

    return specific_h,general_h
    
s_final,g_final=learn(concepts,target)
print("FP : ",s_final,sep='\n')
print("FG : ",g_final,sep='\n')
 

