import numpy as np
import pandas as pd

data=pd.read_csv('enjoy_sport.csv')

print("Dataset :",data)

features=data.iloc[:,:-1].values

print("Features :",features)

target=data.iloc[:,-1].values


print("Target : ",target)

def train(feat,tar):
    for i,val in enumerate(tar):
        
        #NOTE : PLEASE MAKE SURE WHETHER THE DATASET TARGET VALUES MATCHES WITH THE CORRESPONDING CONDITIONAL VALUES** Eg:'Yes' and 'yes' are dissimilar phrases....**(Values are Case-Sensitive**)
        if val=="Yes":
            specific_h=feat[i].copy()
            break
        
    for i,val in enumerate(feat):
        if tar[i]=="Yes":
            for x in range(len(specific_h)):
                if val[x]!=specific_h[x]:
                    specific_h[x]="?"
                else:
                    pass
    return specific_h

print("---Output---")
print(train(features,target))

