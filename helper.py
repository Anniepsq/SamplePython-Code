
import numpy as np
import pandas as pd
import copy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# discount back one period
def discount_one_period(payoff, period_effective_rate, p = 0.5):
    l = payoff.size
    discounted = np.zeros(l-1)
    for i in range(l-1):
        discounted[i] = (payoff[i]*p + payoff[i+1]*(1-p))/(1+period_effective_rate[i])
    return discounted


# discount terminal cash-flow to present
def discount_payoff(payoff, rate_tree, p = 0.5):
    # dim of payoff is the same as depth of tree
    l = payoff.size
    value_tree = np.zeros(rate_tree.shape)
    value_tree[:,l-1] = payoff / (1+rate_tree[:,l-1])
    for i in range(1,l):
        rates = rate_tree[0:l-i,l-i-1]
        value_tree[0:l-i,l-i-1] = discount_one_period(value_tree[0:l-i+1,l-i],rates,p)
    return value_tree


# discount cash-flow to present
def discount_cash_flow(CF, rate_tree, p = 0.5):
    T = CF.size
    value_tree = np.zeros(rate_tree.shape)
    value_tree[:,T-1] = CF[T-1] / (1+rate_tree[:,T-1])
    for i in range(1,T):
        rates = rate_tree[0:T-i,T-i-1]
        value_tree[0:T-i,T-i-1] = discount_one_period \
        (value_tree[0:T-i+1,T-i] + CF[T-i-1],rates,p)
    return value_tree


# discount state specific cash-flow
# CF at diff state of same time might diff
def discount_state_cash_flow(CF, rate_tree, p = 0.5):
    T = CF.shape[0]
    value_tree = np.zeros(rate_tree.shape)
    value_tree[:,T-1] = CF[:,T-1] / (1.0+rate_tree[:,T-1])
    for i in range(T-2,-1,-1):
        rates = rate_tree[0:i+1,i]
        value_tree[0:i+1,i] = discount_one_period \
        (value_tree[0:i+2,i+1] + CF[0:i+2,i],rates,p)
    return value_tree


# given yields return ZCB value (FV = 100)
def zcb_val(yields):
    l = yields.size
    t = np.arange(l)+1
    zcb = 100.0 / (1+yields)**t
    return zcb


# returns the Ho-Lee interest rate tree
def Ho_Lee(yields,s):
    l = yields.size
    zcb = zcb_val(yields)
    rate_tree = np.zeros((l,l))
    rate_tree[0,0] = yields[0]
    
    def tree_plus_m(rate_tree,m):
        l = rate_tree.shape[1]
        add_rate = copy.copy(rate_tree)
        add_rate[:,l-1] += m
        return add_rate
    m = np.zeros(l)
    for i in range(1,l):
        rate_tree[0:i,i] = rate_tree[0:i,i-1] + s
        rate_tree[i,i] = rate_tree[i-1,i-1] - s
        cur_tree = rate_tree[0:i+1,0:i+1]
        payoff = np.repeat(100.0,i+1)
        discount_func_m = lambda m : discount_payoff(payoff, tree_plus_m(cur_tree,m))[0,0]-zcb[i]
        m[i] = fsolve(discount_func_m,0.01)[0]
        rate_tree[0:i+1,i] += m[i]
        
    return [rate_tree,m]


# calculate annuity payment amount
def annuity_payment(r,T):
    if (r == 0.0): return 100.0/T 
    return 100.0 * r / (1 - 1/(1+r)**T)


# return annuity principal array
def annuity_principal(r,T):
    principal = np.repeat(100.0,T)
    ann_pay = annuity_payment(r,T)
    for i in range(1,T):
        principal[i] = principal[i-1]*(1+r)-ann_pay
    return principal


# create state independent CF tree
def cf_tree(CF):
    T = CF.size
    CF_new = np.zeros((T,T))
    for i in range(0,T-1):
        CF_new[0:i+2,i] = np.repeat(CF[i],i+2)
    CF_new[:,T-1] = np.repeat(CF[T-1],T)
    return CF_new


# po value tree
def po_value(principal_tree,rate_tree,barrier,p=0.5):
    T = rate_tree.shape[0]
    po_value_tree = copy.copy(principal_tree)
    CF = copy.copy(principal_tree[0,:])
    CF[0:T-1] = CF[0:T-1] - principal_tree[0,1:T]
    CF = cf_tree(CF)
    po_value_tree[:,T-1] = np.minimum(po_value_tree[:,T-1],CF[:,T-1]/(1+rate_tree[:,T-1]))
    po_value_tree[:,T-1] *= 1-barrier[:,T-1]
    po_value_tree[:,T-1] += barrier[:,T-1] * principal_tree[:,T-1]
    for i in range(T-2,-1,-1):
        po_value_tree[0:i+1,i] = \
            np.minimum(po_value_tree[0:i+1,i], \
            discount_one_period(po_value_tree[0:i+2,i+1]+CF[0:i+2,i],rate_tree[0:i+1,i]))
        po_value_tree[:,i] *= 1-barrier[:,i]
        po_value_tree[:,i] += barrier[:,i] * principal_tree[:,i]
    
    return po_value_tree


# BDT tree calibration, assuming const sigma
def bdt_calibrate(s,r,dt):
    period = r.size
    periods = range(1,period+1) # 1,2, ..., 20
    maturities = np.arange(dt,period+dt,dt)*dt # 0.5, 1, 1.5, ..., 10
    zcb = 100.0/(1+r*dt)**periods # zcb values
    
    def tree_multiply_m(rate_tree,m):
        l = rate_tree.shape[1]
        multiply_rate = copy.copy(rate_tree)
        multiply_rate[:,l-1] *= m
        return multiply_rate
    
    bdt_tree = np.zeros((period,period))
    bdt_tree[0,0] = r[0]
    state_price = np.zeros((period,period))
    state_price[0,0] = 1.0
    vol_multiplier = np.exp(-2.0*s)
    for i in range(1,period):
        subtree = bdt_tree[0:i+1,0:i+1]
        subtree[0,i] = subtree[0,i-1] * np.exp(s)
        for k in range(1,i+1):
            subtree[k,i] = subtree[k-1,i] * vol_multiplier
        zcb_val = zcb[i]
        
        diff = lambda m : np.inner(discount_one_period(np.repeat(100.0,i+1)/
                            (1+tree_multiply_m(subtree,m)[0:i+1,i]*dt),
                                  tree_multiply_m(subtree,m)[0:i,i-1]*dt),state_price[0:i,i-1])-zcb_val
        m = fsolve(diff,1.01)[0]
        bdt_tree[:,i] *= m
        state_price[0,i] = state_price[0,i-1]/(1+dt*bdt_tree[0,i-1])/2.0
        state_price[1:i,i] = (state_price[0:i-1,i-1]/(1+dt*bdt_tree[0:i-1,i-1]) +
                             state_price[1:i,i-1]/(1+dt*bdt_tree[1:i,i-1]))/2.0
        state_price[i,i] = state_price[i-1,i-1]/(1+dt*bdt_tree[i-1,i-1])/2.0
    return [bdt_tree, state_price]


# create probability q matrix
def q_vector(j,j_max=5.0,M=-0.0368055822792):
    vector = np.zeros(3)
    if -j_max < j < j_max:
        vector[0] = 1.0/6.0 + (j**2 * M**2 + j*M) / 2.0
        vector[1] = 2.0/3.0 - j**2 * M**2
        vector[2] = 1.0/6.0 + (j**2 * M**2 - j*M) / 2.0
        return vector
    u = 7.0/6.0 + (j_max**2 * M**2 + 3.0*j_max*M) / 2.0
    m = -1.0/3.0 - j_max**2 * M**2 - 2.0 * j_max * M
    d = 1.0/6.0 + (j_max**2 * M**2 + j_max*M) / 2.0
    if (j == j_max):
        vector[0:3] = [u,m,d]
    if (j == -j_max):
        vector[0:3] = [d,m,u]
    return vector
    

# Hull White discount back one period, i is the period of rates, one before payoff
def hw_discount_one_period(payoff,rates,dt,i,j_max,M):
    value = np.zeros(rates.size)
    if i < j_max:
        for j in range(2*i+1):
            q = q_vector(i-j,j_max,M)
            value[j] = np.inner(q,payoff[j:j+3]) * np.exp(-dt*rates[j])
    else:
        q = q_vector(j_max,j_max,M)
        value[0] = np.inner(q,payoff[0:3]) * np.exp(-dt*rates[0])
        for j in range(1,2*int(j_max)):
            q = q_vector(j_max-j,j_max,M)
            value[j] = np.inner(q,payoff[j-1:j+2]) * np.exp(-dt*rates[j])
        q = q_vector(-j_max,j_max,M)
        value[2*j_max] = np.inner(q,payoff[2*j_max-2:2*j_max+1]) * np.exp(-dt*rates[2*j_max])
    return value
    

# payoff that happens at period time i + 1, discount to present
def hw_discount_payoff(payoff,rate_tree,dt,i,j_max,M):
    k = min(i,j_max)
    value_tree = np.zeros((2*k+1,i+1))
    rates = rate_tree[j_max-k:j_max+k+1,i]
    value_tree[:,i] = payoff * np.exp(-dt*rates)
    for l in range(i-1,int(j_max)-1,-1):
        rates = rate_tree[:,l]
        value_tree[:,l] = hw_discount_one_period(value_tree[:,l+1],rates,dt,l,j_max,M)
    
    for l in range(int(j_max)-1,-1,-1):
        rates = rate_tree[j_max-l:j_max+l+1,l]
        value_tree[j_max-l:j_max+l+1,l] = hw_discount_one_period(value_tree[j_max-l-1:j_max+l+2,l+1],rates,dt,l,j_max,M)
        
    return value_tree
    

# Hull White calibration
def hw_calibrate(zcb,dt,j_max,M,dr):
    T = zcb.size
    rate_tree = np.zeros((j_max*2+1,T))
    rate_tree[j_max,0] = -np.log(zcb[0]/100.0)/dt
    state_price = np.zeros(rate_tree.shape)
    state_price[j_max,0] = 1.0
    
    def rates_add_m(rates,m):
        tree = copy.copy(rates)
        T = rates.shape[1]
        tree[:,T-1] += m
        return tree
    
    for i in range(1,T):
        k = min(i,j_max)
        pre_k = min(i-1,j_max)
        rates = rate_tree[j_max-k:j_max+k+1,0:i+1]
        rates[:,i] += np.repeat(rate_tree[pre_k,i-1],2*k+1) + np.arange(k,-k-1,-1)*dr
        cost_func = lambda m : hw_discount_payoff(np.repeat(100.0,2*k+1),rates_add_m(rates,m),dt,i,k,M)[k,0] - zcb[i]
        m = fsolve(cost_func,0.01)
        rates[:,i] += m

    return rate_tree
        
        
# prepayment ratio according to future interest rate
def prepay_ratio(rate_tree):
    [m,T] = rate_tree.shape
    ratio = np.zeros(rate_tree.shape)
    j_max = m/2
    today = rate_tree[j_max,0]
    for i in range(0,T):
        k = min(i,j_max)
        col_first = m/2-k
        col_last = m/2+k
        for j in range(col_first,col_last+1):
            if rate_tree[j,i] > today:
                ratio[j,i] = 0.03
            elif today - 0.005 < rate_tree[j,i] <= today:
                ratio[j,i] = 0.03 + 4.0 * (today - rate_tree[j,i])
            elif today - 0.01 < rate_tree[j,i] <= today - 0.005:
                ratio[j,i] = 0.05 + 6.0 * ((today - rate_tree[j,i]) - 0.005)
            elif today - 0.02 < rate_tree[j,i] <= today - 0.01:
                ratio[j,i] = 0.08 + 9.0 * ((today - rate_tree[j,i]) - 0.01)
            else:
                ratio[j,i] = 0.17
    return ratio
                

# prepayment mortgage securities value, using Hull White rate tree
def discount_prepay_mbs(c,rate_tree,prepay_ratio,dt,M):
    [m,T] = rate_tree.shape
    value_tree = np.zeros(rate_tree.shape)
    j_max = m/2
    ann_pay = annuity_payment(c*dt,T)
    principal_array = annuity_principal(c*dt,T)
    k = min(T-1,j_max)
    rates = rate_tree[j_max-k:j_max+k+1,T-1]
    value_tree[j_max-k:j_max+k+1,T-1] = hw_discount_one_period(np.repeat(ann_pay,2*k+1),rates,dt,T-1,j_max,M)
    value_tree[j_max-k:j_max+k+1,T-1] = value_tree[j_max-k:j_max+k+1,T-1] * (1.0-prepay_ratio[j_max-k:j_max+k+1,T-1])+\
                                        principal_array[T-1] * (prepay_ratio[j_max-k:j_max+k+1,T-1])
    
    for i in range(T-2,-1,-1):
        k = min(i,j_max)
        former_k = min(i+1,j_max)
        rates = rate_tree[j_max-k:j_max+k+1,i]
        value_tree[j_max-k:j_max+k+1,i] = hw_discount_one_period(value_tree
                                            [j_max-former_k:j_max+former_k+1,i+1]+ann_pay,rates,dt,i,j_max,M)
        value_tree[j_max-k:j_max+k+1,i] = value_tree[j_max-k:j_max+k+1,i] * (1.0-prepay_ratio[j_max-k:j_max+k+1,i])+\
                                            principal_array[i] * (prepay_ratio[j_max-k:j_max+k+1,i])
        
    return value_tree


# no prepayment mortgage securities value, using Hull White rate tree
def discount_nonprepay_mbs(c,rate_tree,dt,M):
    [m,T] = rate_tree.shape
    value_tree = np.zeros(rate_tree.shape)
    j_max = m/2
    ann_pay = annuity_payment(c*dt,T)
    principal_array = annuity_principal(c*dt,T)
    k = min(T-1,j_max)
    rates = rate_tree[j_max-k:j_max+k+1,T-1]
    value_tree[j_max-k:j_max+k+1,T-1] = \
        hw_discount_one_period(np.repeat(ann_pay,2*k+1),rates,dt,T-1,j_max,M)
    
    for i in range(T-2,-1,-1):
        k = min(i,j_max)
        former_k = min(i+1,j_max)
        rates = rate_tree[j_max-k:j_max+k+1,i]
        value_tree[j_max-k:j_max+k+1,i] = hw_discount_one_period(value_tree
                        [j_max-former_k:j_max+former_k+1,i+1]+ann_pay,rates,dt,i,j_max,M)
    
    return value_tree


# generate path
def hw_tree_random_path(rate_tree,j_max,M):
    [m,T] = rate_tree.shape
    path = np.zeros(T)
    rates = np.zeros(T)
    cur = j_max
    path[0] = cur
    rates[0] = rate_tree[cur,0]
    for i in range(1,T):
        k = min(i,j_max)
        up = max(cur-1,0)
        if i > j_max and cur == m-1:
            up = cur-2
        q = q_vector(cur-j_max,j_max,M)
        z = np.random.rand(1)
        if z < q[0]:
            path[i] = up
        elif q[0] <= z < q[0]+q[1]:
            path[i] = up+1
        else:
            path[i] = up+2
        cur = path[i]
        rates[i] = rate_tree[cur,i]
    return [rates,path]


# generate antithetic path
def hw_tree_antithetic_path(rate_tree,j_max,M):
    [m,T] = rate_tree.shape
    path1 = np.zeros(T)
    path2 = np.zeros(T)
    rates1 = np.zeros(T)
    rates2 = np.zeros(T)
    cur1 = j_max
    cur2 = j_max
    path1[0] = cur1
    path2[0] = cur2
    rates1[0] = rate_tree[cur1,0]
    rates2[0] = rate_tree[cur2,0]
    for i in range(1,T):
        k = min(i,j_max)
        up1 = max(cur1-1,0)
        up2 = max(cur2-1,0)
        if i > j_max and cur1 == m-1:
            up1 = cur1-2
        if i > j_max and cur2 == m-1:
            up2 = cur2-2
        q1 = q_vector(cur1-j_max,j_max,M)
        q2 = q_vector(cur2-j_max,j_max,M)
        z1 = np.random.rand(1)
        z2 = 1.0-z1
        if z1 < q1[0]:
            path1[i] = up1
        elif q1[0] <= z1 < q1[0]+q1[1]:
            path1[i] = up1+1
        else:
            path1[i] = up1+2
        cur1 = path1[i]
        rates1[i] = rate_tree[cur1,i]
        
        if z2 < q2[0]:
            path2[i] = up2
        elif q2[0] <= z2 < q2[0]+q2[1]:
            path2[i] = up2+1
        else:
            path2[i] = up2+2
        cur2 = path2[i]
        rates2[i] = rate_tree[cur2,i]
    return [[rates1,path1],[rates2,path2]]


# discount along path of rates
def discount_path_prepay_mbs(c,path,rates,prepay_ratio,dt,M):
    T = path.size
    ratio = np.zeros(T)
    for i in range(T):
        ratio[i] = prepay_ratio[path[i],i]
    value_array = np.zeros(T)
    ann_pay = annuity_payment(c*dt,T)
    principal_array = annuity_principal(c*dt,T)
    value_array[T-1] = ann_pay*np.exp(-dt*rates[T-1])*(1.0-ratio[T-1])+principal_array[T-1]*(ratio[T-1])
    for i in range(T-2,-1,-1):
        value_array[i] = principal_array[i]*(ratio[i]) + \
                    (value_array[i+1]+ann_pay)*np.exp(-dt*rates[i])*(1-ratio[i])
    
    return value_array
    

