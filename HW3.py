
# Q1

import helper as hp
import numpy as np
import pandas as pd
import copy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# sigma
s = 0.015
# spot rate of ZCB
r = np.array([0.05, 0.055, 0.057, 0.059, 0.06, 0.061])
rate_tree = np.zeros((6,6))
[rate_tree,m] = hp.Ho_Lee(r,s)
print (rate_tree)

# check answer
for i in range(1,7):
    payoff = np.repeat(100.0, i)
    pr = hp.discount_payoff(payoff,rate_tree[0:i,0:i])[0,0]
    print (100.0/pr)**(1.0/i)-1

df = pd.DataFrame(rate_tree,
                 columns=["1 year","2 years","3 years","4 years","5 years","6 years"])
df.to_csv("output/Q1.csv")


# Q2

import helper as hp
import numpy as np
import pandas as pd
import copy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv("output/Q1.csv",header=0)
rate_tree = np.array(df.ix[:,1:])
T = 6
r = 0.055
ann_pay = hp.annuity_payment(r,T)

# a
price_a = 0
CF = np.repeat(ann_pay,T)
non_prepay_value_tree = hp.discount_cash_flow(CF,rate_tree)
non_prepayable_price = non_prepay_value_tree[0,0]

# b
principal_array = hp.annuity_principal(r,T)
principal_tree = np.zeros((T,T))
for i in range(T):
    principal_tree[0:i+1,i] = np.repeat(principal_array[i],i+1)

prepay_value_tree = np.zeros(non_prepay_value_tree.shape)
prepay_value_tree[:,T-1] = np.minimum(principal_tree[:,T-1],non_prepay_value_tree[:,T-1])
for i in range(T-1,-1,-1):
    prepay_value_tree[0:i,i-1] = np.minimum(                 hp.discount_one_period(prepay_value_tree[0:i+1,i]+ann_pay,rate_tree[0:i,i-1]),                                          principal_tree[0:i,i-1])
    
prepayable_price = prepay_value_tree[0,0]
  
barrier = np.zeros(prepay_value_tree.shape)
barrier[prepay_value_tree == principal_tree] = 1
barrier[prepay_value_tree == 0] = 0


# c
po_array = principal_array
po_array[0:T-1] = po_array[0:T-1]-po_array[1:T]
po_tree = hp.cf_tree(po_array)
po_value_tree = hp.discount_state_cash_flow(po_tree,rate_tree)
prepay_po_tree = hp.po_value(principal_tree,rate_tree,barrier)
po_value = prepay_po_tree[0,0]

# d
prepay_io_tree = prepay_value_tree - prepay_po_tree
io_value = prepay_io_tree[0,0]


# output
Q2a = pd.DataFrame(non_prepay_value_tree)
Q2b = pd.DataFrame(prepay_value_tree)
Q2c = pd.DataFrame(prepay_po_tree)
Q2d = pd.DataFrame(prepay_io_tree)
Q2a.to_csv("output/Q2a.csv")
Q2b.to_csv("output/Q2b.csv")
Q2c.to_csv("output/Q2c.csv")
Q2d.to_csv("output/Q2d.csv")



# Q3

import helper as hp
import numpy as np
import pandas as pd
import copy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.style.use('ggplot')

r = pd.read_csv("Q8_Part1.csv",header=0).fitted_spot[0:20]
dt = 0.5
s = 0.15

[bdt_tree, state_price] = hp.bdt_calibrate(s,r[0:20],dt)

Q3_bdt_rate = pd.DataFrame(bdt_tree)
Q3_bdt_rate.to_csv("output/Q3_bdt_rate.csv")




# Q4

import helper as hp
import numpy as np
import pandas as pd
import copy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.style.use('ggplot')

r = pd.read_csv("Q8_Part1.csv",header=0).fitted_spot[0:20]
dt = 0.5
zcb = 100.0/(1+r*dt)**range(1,21)
kappa = 0.075
s = 0.01
M = np.exp(-kappa*dt)-1.0
V = s**2/2.0/kappa*(1.0-np.exp(-2.0*kappa*dt))
dr = np.sqrt(V*3.0)
j_max = np.ceil(-0.184/M)

rate_tree = hp.hw_calibrate(zcb,dt,j_max,M,dr)
Q4 = pd.DataFrame(rate_tree)
Q4.to_csv("output/Q4.csv")




# Q5

import helper as hp
import numpy as np
import pandas as pd
import copy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.style.use('ggplot')

dt = 0.5
kappa = 0.075
s = 0.01
M = np.exp(-kappa*dt)-1.0
V = s**2/2.0/kappa*(1.0-np.exp(-2.0*kappa*dt))
dr = np.sqrt(V*3.0)
j_max = np.ceil(-0.184/M)
c = 0.0575

# a
rate_tree = np.array(pd.read_csv("output/Q4.csv",header=0).ix[:,1:11])
# prepay ratio
ratio = hp.prepay_ratio(rate_tree)
prepayable_MBS_price = hp.discount_prepay_mbs(c,rate_tree,ratio,dt,M)
print ("Theoretical price: ", prepayable_MBS_price[5,0])
print (np.round(prepayable_MBS_price,4))
df = pd.DataFrame(prepayable_MBS_price)
df.to_csv("output/Q5a.csv")

# b
N = 1000
price = np.zeros(N)
for i in range(N):
    [rates,path] = hp.hw_tree_random_path(rate_tree,j_max,M)
    price[i] = hp.discount_path_prepay_mbs(c,path,rates,ratio,dt,M)[0]
print ("Random paths price: ", np.mean(price))
print ("Standard error of 1000 random paths: ", np.std(price)/np.sqrt(N))

# c
K = 500
price_a = np.zeros(2*K)
for i in range(K):
    [[rates1,path1],[rates2,path2]] = hp.hw_tree_antithetic_path(rate_tree,j_max,M)
    price_a[2*i] = hp.discount_path_prepay_mbs(c,path1,rates1,ratio,dt,M)[0]
    price_a[2*i+1] = hp.discount_path_prepay_mbs(c,path2,rates2,ratio,dt,M)[0]
print ("Antithetic paths price: ", np.mean(price_a))
print ("Standard error of 500 antithetic paths: ", np.std(price_a)/np.sqrt(2*K))



