import numpy as np

def ex1_sample(n,d,d_t,d_y):
    X = np.random.multivariate_normal(np.zeros(d),np.eye(d),size=n)
    logit_pt = (1/d_t)*np.sum(X[:,:d_t],axis=1)
    p_t = 1/(1+np.exp(-logit_pt))
    T = np.random.binomial(1,p=p_t)
    eps = np.random.normal(0,0.5,size=n)
    Y0 = Y1 = (1/d_y)*np.sum(X[:,:d_y],axis=1)
    Y = Y0+eps
    ITE = np.zeros(n)

    results = {'x':X, 'y': Y, 't':T, 'pt':p_t,'y1':Y1, 'y0':Y0, 'ite':ITE}

    return results

def ex2_sample(n,d,alpha):
    X = np.random.multivariate_normal(np.zeros(d),np.eye(d),size=n)
    logit_pt = alpha*np.sum(X,axis=1)
    p_t = 1/(1+np.exp(-logit_pt))
    T = np.random.binomial(1,p=p_t)
    eps = np.random.normal(0,0.5,size=n)
    Y0 = Y1 = (1/d)*np.sum(X,axis=1)
    Y = Y0+eps
    ITE = np.zeros(n)

    results = {'x':X, 'y': Y, 't':T, 'pt':p_t,'y1':Y1, 'y0':Y0, 'ite':ITE}

    return results

def ex3_sample1(n,d):
    X = np.random.multivariate_normal(np.zeros(d),np.eye(d),size=n)
    T = np.random.choice([0,1],size=n)
    eps = np.random.normal(0,0.5,size=n)
    Y0 = np.sum(X,axis=1)
    Y1 = 1+Y0
    Y = Y1*T + (1-T)*Y0 + eps
    ATE = 1

    results = {'x':X, 'y': Y, 't':T, 'y1':Y1, 'y0':Y0, 'ate':ATE}

    return results

def ex3_sample2(n,d):
    X = np.random.multivariate_normal(np.zeros(d),np.eye(d),size=n)
    T = np.random.choice([0,1],size=n)
    eps = np.random.normal(0,0.5,size=n)
    Y0 = np.sum(X,axis=1) + (X[:,0]>0)
    Y1 = 1+Y0
    Y = Y1*T + (1-T)*Y0 + eps
    ATE = 1

    results = {'x':X, 'y': Y, 't':T, 'y1':Y1, 'y0':Y0, 'ate':ATE}

    return results

def ex4_sample(n,d):
    X = np.random.multivariate_normal(np.zeros(d),np.eye(d),size=n)
    logit_pt = (1/d)*np.sum(X,axis=1)
    p_t = 1/(1+np.exp(-logit_pt))
    T = np.random.binomial(1,p=p_t)
    logit_py = (1/d)*np.sum(X,axis=1)
    p_y = 1/(1+np.exp(-logit_py))
    Y0 = Y1 = np.random.binomial(1,p=p_y)
    Y = Y1*T + (1-T)*Y0
    ITE = np.zeros(n)

    results = {'x':X, 'y': Y, 't':T, 'pt':p_t,'y1':Y1, 'y0':Y0, 'ite':ITE}

    return results


