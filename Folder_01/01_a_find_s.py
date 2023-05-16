import numpy as np
import pandas as pd


data=pd.read_csv('Enjoy_sport1.csv')

print("Dataset : ",data)

concepts=data.iloc[:,:-1].values
target=data.iloc[:,-1].values

print("Concepts : ",concepts)
print("Target : ",target)

def train(conc,tar):
    for i,val in enumerate(tar):
        if val=="Yes":
            specific_h=conc[i].copy()
            break
    for i,val in enumerate(conc):
        if tar[i]=="Yes":
            for x in range(len(specific_h)):
                if val[x]!=specific_h[x]:
                    specific_h[x]="?"
                else:
                    pass
                
    return specific_h

print("Final specific_h : ",train(concepts,target))
