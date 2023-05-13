import numpy as np
import pandas as pd


dataset=pd.read_csv("enjoy_sport.csv")
print("Dataset : ",dataset)


# Creating dataset of concepts and target values

concepts=dataset.iloc[:,:-1].values
target=dataset.iloc[:,1].values

print("\nConcepts : ",concepts)
print("\nTarget : ",target)


# Initialization of specific_h and general_h hypothesis


def initialize(concepts):
    print("\nInitialization of specific_h and general_h\n")
    specific_h=concepts[0]
    
    print("Initial Specific Boundary :\n ",specific_h)


    general_h=[["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    
    print("Initial General Boundary :\n ",general_h)

    return specific_h,general_h



def learn(concepts,target):
    specific_h,general_h=initialize(concepts)
    
    
    for i,h in enumerate(concepts):
        print("Instance ",i+1," : ",h)
        
        if(target[i]=="Yes"):    #Note: MAkE SURE THAT THE COMPARING VALUE MATCHES WITH CORRESPONDING DATASET VALUES
            
            print("Instance is Positive...")
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x] and i==0:
                    specific_h[x]=concepts[0].copy()
                elif h[x]!=specific_h[x]:
                    specific_h[x]='?'
                    general_h[x][x]='?'
                    
                    
                    
        if(target[i]=="No"):    #Note: MAkE SURE THAT THE COMPARING VALUE MATCHES WITH CORRESPONDING DATASET VALUES

            print("Instacne is Negative ...")
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else:
                    general_h[x][x]='?'
                    
        print("Specific Boundary after ",i+1," Instance is : ",specific_h)
        print("General Boundary after ",i+1," Instance is : ",general_h)


        indices=[i for i,val in enumerate(general_h) if(val==['?','?','?','?','?','?'])]

        for i in indices:
            general_h.remove(['?','?','?','?','?','?'])
            
        indices=[i for i,val in enumerate(general_h) if(val==['?']*len(concepts[0]))]

    return specific_h,general_h

s_final,g_final=learn(concepts,target)


print("Final specific_h : ",s_final,sep='\n')
print("Final general_h : ",g_final,sep='\n')


