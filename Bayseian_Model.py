
@author: Andy
"""
"""
Created on Thu Apr  4 14:33:37 2019

@author: Andy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from collections import Counter
from tqdm import tqdm
from pgmpy.inference import VariableElimination as VE

#TIme variables
days = 24
months = 720
train_start=0
train_end = 349*days
train_t=list(range(train_start, train_end+1))
time_start = 335*days
time_end = 345*days


#Global functions
def findmaxprob(cpd):
    zero = cpd.values[0]
    
    if zero > 0.5:
        return 0
    else:
        return 1
    
    
def findmaxprob2(cpd):
    zero = cpd.values[0]
    
    if zero > 0.8:
        return 0
    else:
        return 1
    
def getweeklydata(data):
    Week = 7*24
    Weeks = []

    
    CSV = data

    for i in range(0,len(CSV), Week):
        Weeks += [CSV[i:i+Week]]

    for i in range(len(Weeks)):
        if sum(Weeks[i]) >= (len(Weeks[i]) - sum(Weeks[i])): #more 1s than zeros
            Weeks[i] = 1
        else:
            Weeks[i] = 0
    
    
    print(Weeks, len(Weeks))
    
def getTypeError(estimate,data):
    if estimate == data:
        return 0
    if estimate != data:
        if data == 0:
            return 1
        else:
            return 2
            


#Bristol-Wide DATA
###TEMP###
T = pd.read_csv('Binary_temp_2014.csv', delimiter=',')
LT = len(T)
print(LT)
T1_b=T.loc[:,'0']
T1 = [ T1_b[i] for i in train_t]


###Wind Speed###
WS = pd.read_csv('Binary_ws_2014_threshMean.csv', delimiter=',')
LW = len(WS)
WS1_b=WS.loc[:,'1']
WS1 = [ WS1_b[i] for i in train_t]

###Traffic###
Traf = pd.read_csv('Traffic_own_model.csv', delimiter=',')
Traffic_b=Traf.loc[:,'0']
Traffic = [ Traffic_b[i] for i in train_t]

####LOCAL DATA####
###St Pauls###
no_Stp = pd.read_csv('StPauls_Binary_No2_Thresh60.csv', delimiter=',')
NO_Stp_b=no_Stp.loc[:,'0']
NO_Stp = [ NO_Stp_b[i] for i in train_t]

no_raw = pd.read_csv('StPauls_No2_2014.csv', delimiter=',')
NO_raw_b=no_raw.loc[:,'16.5']
NO_raw = [ NO_raw_b[i] for i in train_t]

Traf_S = pd.read_csv('StPauls_Binary_traffic.csv', delimiter=',')
Traffic_b_S=Traf_S.loc[:,'0']
Traffic_S = [ Traffic_b_S[i] for i in train_t]

###A37###
no_A37 = pd.read_csv('A37_Binary_No2_Thresh60.csv', delimiter=',')
NO_A37_b=no_A37.loc[:,'0']
NO_A37 = [ NO_A37_b[i] for i in train_t]

Traf_A = pd.read_csv('A37_Binary_traffic.csv', delimiter=',')
Traffic_b_A=Traf_A.loc[:,'1']
Traffic_A = [ Traffic_b_A[i] for i in train_t]

###Brislington###
no_Bris = pd.read_csv('Bris_Binary_No2_Thresh60.csv', delimiter=',')
NO_Bris_b=no_Bris.loc[:,'0']
NO_Bris = [ NO_Bris_b[i] for i in train_t]

Traf_B = pd.read_csv('Bris_Binary_traffic.csv', delimiter=',')
Traffic_b_B=Traf_B.loc[:,'1']
Traffic_B = [ Traffic_b_B[i] for i in train_t]

# With all the data formatted for binary input the bayseian network can now be trained

####### GENERATING MODEL ########
d = {'NO_Stp': NO_Stp,'NO_A37': NO_A37,'NO_Bris': NO_Bris, 'Temp': T1,'Wind_Speed':WS1,'Traffic_S':Traffic_S ,'Traffic_A':Traffic_A,'Traffic_B':Traffic_B}
#print(d)
data = pd.DataFrame(data=d)


#data = pd.DataFrame(np.random.randint(low=0, high=2, size=(5000, 4)), columns=['A', 'B', 'C', 'D'])
model = BayesianModel([('Temp', 'NO_Stp'), ('Temp', 'NO_A37'),('Temp', 'NO_Bris'),('Wind_Speed', 'NO_Stp'),('Wind_Speed', 'NO_A37'),('Wind_Speed', 'NO_Bris'),('Traffic_S', 'NO_Stp'),('Traffic_A', 'NO_A37'),('Traffic_B', 'NO_Bris'),('Traffic_A', 'NO_Bris'),('Traffic_A', 'NO_Stp'),('Traffic_B', 'NO_A37'),('Traffic_B', 'NO_Stp'),('Traffic_S', 'NO_A37'),('Traffic_S', 'NO_Bris')])

model.fit(data, estimator=BayesianEstimator, prior_type="BDeu") # default equivalent_sample_size=5
#for cpd in model.get_cpds():
    #print(cpd)
#print(('\n'))

    
infer = VE(model)




NO_OUT_Stp=[]
NO_OUT_A37=[]
NO_OUT_Bris=[]
NO_OUT_Stp_p=[]
NO_OUT_A37_p=[]
NO_OUT_Bris_p=[]
Error_Stp=[]
Error_Bris=[]
Error_A37=[]

t=list(range(time_start, time_end))
#### GLOBAL TEST DATA ####
test_T1 = [ T1_b[i] for i in t]
test_WS1 = [ WS1_b[i] for i in t]
test_Traffic = [ Traffic_b[i] for i in t]

#### LOCAL TEST DATA####
test_NO2_Stp  = [ NO_Stp_b[i] for i in t]
test_NO2_A37 = [ NO_A37_b[i] for i in t]
test_NO2_Bris = [ NO_Bris_b[i] for i in t]
Traffic_S_test = [Traffic_b_S[i] for i in t]
Traffic_A_test = [Traffic_b_A[i] for i in t]
Traffic_B_test = [Traffic_b_B[i] for i in t]
NO_raw_test = [NO_raw_b[i] for i in t]
for i in tqdm(range(time_end-time_start)):
   
    Stp = infer.query(['NO_Stp'], evidence={'Temp':test_T1[i], 'Wind_Speed':test_WS1[i],'Traffic_S':Traffic_S_test[i],'Traffic_A':Traffic_A_test[i],'Traffic_B':Traffic_B_test[i]}) ['NO_Stp']
    NO_OUT_Stp.append(findmaxprob(Stp))
    NO_OUT_Stp_p.append(Stp.values[1])
    Error_Stp.append(getTypeError(NO_OUT_Stp[i],test_NO2_Stp[i]))
    A37= infer.query(['NO_A37'], evidence={'Temp':test_T1[i], 'Wind_Speed':test_WS1[i],'Traffic_S':Traffic_S_test[i],'Traffic_A':Traffic_A_test[i],'Traffic_B':Traffic_B_test[i]}) ['NO_A37']
    NO_OUT_A37.append(findmaxprob(A37))
    NO_OUT_A37_p.append(A37.values[1])
    Error_A37.append(getTypeError(NO_OUT_A37[i],test_NO2_A37[i]))
    Bris = infer.query(['NO_Bris'], evidence={'Temp':test_T1[i], 'Wind_Speed':test_WS1[i],'Traffic_S':Traffic_S_test[i],'Traffic_A':Traffic_A_test[i],'Traffic_B':Traffic_B_test[i]}) ['NO_Bris']
    NO_OUT_Bris.append(findmaxprob2(Bris))
    NO_OUT_Bris_p.append(Bris.values[1])
    Error_Bris.append(getTypeError(NO_OUT_Bris[i],test_NO2_Bris[i]))
    
Error_Stp=Counter(Error_Stp)
Error_Bris=Counter(Error_Bris)
Error_A37=Counter(Error_A37)
print(Error_Stp)
print(Error_Bris)
print(Error_A37)


#print(NO_OUT_Stp)
#print(NO_OUT_A37)
#print(NO_OUT_Bris)
Result_Stp =[]
Result_A37=[]
Result_Bris = []
for i in tqdm(range(len(NO_OUT_Stp))):
    if NO_OUT_Stp[i]==test_NO2_Stp[i]:
        Result_Stp.append(1)
    else:
        Result_Stp.append(0)
      
for i in tqdm(range(len(NO_OUT_A37))):
    if NO_OUT_A37[i]==test_NO2_A37[i]:
        Result_A37.append(1)
    else:
        Result_A37.append(0)
        
for i in tqdm(range(len(NO_OUT_Bris))):
    if NO_OUT_Bris[i]==test_NO2_Bris[i]:
        Result_Bris.append(1)
    else:
        Result_Bris.append(0) 

print('\n St Pauls',sum(Result_Stp)/len(Result_Stp))
print('\n A37',sum(Result_A37)/len(Result_A37))
print('\n Brislington',sum(Result_Bris)/len(Result_Bris))


tx = list(range(time_start, time_end))
plt.plot(tx, NO_OUT_Stp_p, label='Predicted probability of spike')
plt.plot(tx, test_NO2_Stp, label='Recorded Spikes')


plt.xlabel('Time (Hours)')
plt.ylabel('Probability of NO2 Spike')

plt.title("St Pauls Polution Levels Complex Model")

plt.legend()
plt.savefig('S1_complex.png', bbox_inches='tight',transparent=True)
plt.show()

plt.plot(tx, NO_OUT_Bris_p, label='Predicted probability of spike')
plt.plot(tx, test_NO2_Bris, label='Recorded Spikes')


plt.xlabel('Time (Hours)')
plt.ylabel('Probability of NO2 Spike')

plt.title("Brislington Polution Levels Complex Model")

plt.legend(loc='upper center')
plt.savefig('B1_complex.png', bbox_inches='tight',transparent=True)

plt.show()

plt.plot(tx, NO_OUT_A37_p, label='Predicted probability of spike')
plt.plot(tx, test_NO2_A37, label='Recorded Spikes')


plt.xlabel('Time (Hours)')
plt.ylabel('Probability of NO2 Spike')

plt.title("A37 Polution Levels Complex Model")

plt.legend(loc='upper center')
plt.savefig('A1_complex.png', bbox_inches='tight',transparent=True)
plt.show()
    








labels = 'Correct', 'Type 1 Error', 'Type 2 Error'
sizes_S = [Error_Stp[0], Error_Stp[1], Error_Stp[2]]

explode = (0.1, 0.1, 0.1)

fig1, ax1 = plt.subplots()
ax1.pie(sizes_S,explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("St Pauls Error Complex Model")
plt.savefig('S2_complex.png', bbox_inches='tight',transparent=True)
plt.show()

sizes_A = [Error_A37[0], Error_A37[1], Error_A37[2]]


fig1, ax1 = plt.subplots()
ax1.pie(sizes_A,explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("A37 Error Complex Model")
plt.savefig('A2_complex.png', bbox_inches='tight',transparent=True)
plt.show()



sizes_B = [Error_Bris[0], Error_Bris[1], Error_Bris[2]]


fig1, ax1 = plt.subplots()
ax1.pie(sizes_B,explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Brislington Error Complex Model")
plt.savefig('B2_complex.png', bbox_inches='tight',transparent=True)
plt.show()

#plt.figure(1)
#plt.plot(t, NO_OUT_Stp, 'r', t, NO_OUT_A37, 'g',t,NO_OUT_Bris,'b')
#plt.show()
"""
fig, ax1 = plt.subplots()

color = 'r'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('NO probability', color=color)
ax1.plot(tx,NO_OUT_Stp_p , color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'b'
ax2.set_ylabel('Recorded NO2', color=color)  # we already handled the x-label with ax1
ax2.plot(tx, NO_raw_test, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("Recorded NO2 against Predicted NO2 St Pauls")
plt.savefig('simple_real.png', bbox_inches='tight',transparent=True)
plt.show()
""" 