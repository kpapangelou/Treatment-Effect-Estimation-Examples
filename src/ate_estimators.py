import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

def conf_int(mu,s,n,a=0.05):
    z = stats.t.ppf(1-a/2,n-1)
    lower = mu-z*s/np.sqrt(n)
    upper = mu+z*s/np.sqrt(n)
    return lower,upper

def ate_unadjusted_cont(trt,y):

    mu1 = np.mean(y[trt.ravel()==1]) 
    mu0 = np.mean(y[trt.ravel()==0])
    ate = mu1 - mu0
    
    mu1 = mu1.reshape(-1,1)
    mu0 = mu0.reshape(-1,1)
    trt = trt.reshape(-1,1)
    y = y.reshape(-1,1)

    ate_std = np.sqrt(np.mean(((2*trt*(y-mu1)) - (2*(1-trt)*(y-mu0)))**2))

    return ate, ate_std

def ate_tmle_cont(trt,x,y):

    trt = trt.reshape(-1,1)
    if x.ndim == 1:
        x = x.reshape(-1,1)
    if y.ndim == 1:
        y = y.reshape(-1,1)
    n = trt.shape[0]

    covariates = np.column_stack((trt,x,x*trt))
    model = LinearRegression().fit(covariates,y)
    covariates0 = np.column_stack((np.zeros([n,1]),x,np.zeros_like(x)))
    m0_pred = model.predict(covariates0)
    covariates1 = np.column_stack((np.ones([n,1]),x,x))
    m1_pred = model.predict(covariates1)

    ate = np.mean(m1_pred) - np.mean(m0_pred)

    ate_std = np.sqrt(np.mean(((2*trt*(y-m1_pred)+m1_pred-np.mean(m1_pred))- (2*(1-trt)*(y-m0_pred)+m0_pred-np.mean(m0_pred)))**2))

    return ate, ate_std

