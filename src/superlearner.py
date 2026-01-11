import sklearn 
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
from scipy.optimize import minimize

class SuperLearner():

    def __init__(self,k,seed,learners):
        self.k = k
        self.seed = seed
        self.learners = learners

    def train_cont(self,x,y,y_stratified):
        cv = StratifiedKFold(n_splits=self.k,shuffle=True,random_state=self.seed)
        predictions = np.nan*np.ones([x.shape[0],len(self.learners)])

        # Step 1: For each fold get predictions using different learners
        for i, (idx_tr, idx_te) in enumerate(cv.split(x, y_stratified)):
            x_train = x[idx_tr]
            y_train = y[idx_tr]
            x_test = x[idx_te]

            for j in range(len(self.learners)):
                model_j = self.learners[j].fit(x_train,y_train)
                predictions[idx_te,j] = model_j.predict(x_test)

        # Step 2: Non-negative Least Squares
        model = sklearn.linear_model.LinearRegression(positive=True,fit_intercept=False)
        model = model.fit(predictions,y)
        self.weights = model.coeff_
        self.weights[self.weights<0] = 0
        self.weights = self.weights/np.sum(self.weights)
        self.model = model

        # Step 3: Fit each learner in the full data
        self.fitted_learners = [self.learners[i].fit(x,y) for i in range(len(self.learners))]
    
    def predict_cont(self,x):
        predictions = np.nan*np.ones([x.shape[0],len(self.fitted_learners)])
        for i in range(self.fitted_learners):
            predictions[:,i] = self.fitted_learners[i].predict(x)
        y_pred = np.sum(self.weights.reshape(1,-1)*predictions,axis=1)
        return y_pred
    
    def train_binary(self,x,y):
        cv = KFold(n_splits=self.k,shuffle=True,random_state=self.seed)
        predictions = np.nan*np.ones([x.shape[0],len(self.learners)])

        # Step 1: For each fold get predictions using different learners
        for i, (idx_tr, idx_te) in enumerate(cv.split(x, y)):
            x_train = x[idx_tr]
            y_train = y[idx_tr]
            x_test = x[idx_te]

            for j in range(len(self.learners)):
                model_j = self.learners[j].fit(x_train,y_train)
                predictions[idx_te,j] = model_j.predict_proba(x_test)[:,model_j.classes_==1].ravel()

        # Step 2: Minimize negative log-likelihood with positive coefficients
        model = optim_ll(np.zeros(len(self.learners)),predictions,y.reshape(-1,1))
        self.weights = model.x
        self.weights[self.weights<0] = 0
        self.weights = self.weights/np.sum(self.weights)

        self.model = model

        # Step 3: Fit each learner in the full data
        self.fitted_learners = [self.learners[i].fit(x,y) for i in range(len(self.learners))]
    
    def predict_binary(self,x):
        predictions = np.nan*np.ones([x.shape[0],len(self.fitted_learners)])
        for i in range(len(self.fitted_learners)):
            predictions[:,i] = self.fitted_learners[i].predict_proba(x)[:,self.fitted_learners[i].classes_==1].ravel()
        y_pred = np.sum(self.weights.reshape(1,-1)*predictions,axis=1)
        return y_pred
    
def ll(beta,x,y):
    '''
    x: shape (n,d)
    y: shape (n,1)
    beta: shape (d,)
    '''
    p = 1/(1+np.exp(-x@beta.reshape(-1,1)))
    return -np.sum(y*np.log(p)+(1-y)*np.log(1-p))

def ll_grad(beta,x,y):
    '''
    x: shape (n,d)
    y: shape (n,1)
    beta: shape (d,)
    '''
    p = 1/(1+np.exp(-x@beta.reshape(-1,1)))
    return -np.dot(x.T,(y-p)).ravel()

def optim_ll(beta0,x,y):
    res = minimize(ll, x0=beta0, args=(x,y), jac=ll_grad,method='l-bfgs-b',bounds=[(0.0,None)]*len(beta0))
    return res

        

