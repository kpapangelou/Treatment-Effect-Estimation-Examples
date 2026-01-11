'''
CATE estimators using pseudo-outcomes

Relevant references:
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. Electronic Journal of Statistics, 17(2), 3008-3049.
- Curth, A., & Van der Schaar, M. (2021, March). Nonparametric estimation of heterogeneous treatment effects: From theory to learning algorithms. 
In International Conference on Artificial Intelligence and Statistics (pp. 1810-1818). PMLR.
'''

import sklearn 
from sklearn.model_selection import StratifiedKFold
import numpy as np
import inspect

class PseudoLearner:
    def __init__():
        pass

    def train():
        pass

    def predict(self,X):
        return self.model.predict(X)

class PseudoDRLearner(PseudoLearner):

    def __init__(self,k,seed,m_model=sklearn.linear_model.LinearRegression,pt_model=sklearn.linear_model.LogisticRegression,cate_model=sklearn.linear_model.LinearRegression):
        '''
        k: number of folds for cross-fitting
        seed: random seed used to initialize K fold cross-validation and sklearn models
        m_model: sklearn model for outcome
        pt_model: sklearn model for treatment
        cate_model: sklearn model for CATE
        '''
        self.k = k
        self.m_model = m_model
        self.pt_model = pt_model
        self.cate_model = cate_model
        self.seed = seed

    def train(self,x,y,trt,**kwargs):
        '''
        x,y,trt: training data as numpy arrays
        kwargs: additional arguments for models - dictionary of the form: {'y':{arg1:value1, ...},'t':{arg1:value1, ...}, 'cate': {arg1:value1,...}}
        '''
        cv = StratifiedKFold(n_splits=self.k,shuffle=True,random_state=self.seed)
        self.cv = cv
        n = x.shape[0]
        x_outcome = np.column_stack((x,trt))

        self.pseudo_outcome = np.nan*np.ones([n])
        self.y0_pred = np.nan*np.ones([n])
        self.y1_pred = np.nan*np.ones([n])
        self.p_pred = np.nan*np.ones([n])
        self.y_trained_models = []
        self.p_trained_models = []

        m_model_kwargs = kwargs.get('y',{})
        pt_model_kwargs = kwargs.get('t',{})
        cate_model_kwargs = kwargs.get('cate',{})
        # Overwrite seed if required by the model
        if 'seed' in inspect.getfullargspec(self.m_model).args:
            m_model_kwargs['seed'] = self.seed
        if 'seed' in inspect.getfullargspec(self.pt_model).args:
            pt_model_kwargs['seed'] = self.seed
        if 'seed' in inspect.getfullargspec(self.cate_model).args:
            cate_model_kwargs['seed'] = self.seed

        for i, (idx_tr, idx_te) in enumerate(cv.split(x, trt)):

            x_train = x[idx_tr]
            x_outcome_train = x_outcome[idx_tr]
            y_train = y[idx_tr]
            trt_train = trt[idx_tr]
            trt_test = trt[idx_te]
            x_test = x[idx_te]
            y_test = y[idx_te]

            # Training of nuisance parameters in k-1 folds
            m_curr_model = self.m_model(**m_model_kwargs).fit(x_outcome_train,y_train)
            pt_curr_model = self.pt_model(**pt_model_kwargs).fit(x_train,trt_train)

            self.y_trained_models.append(m_curr_model)
            self.p_trained_models.append(pt_curr_model)

            # Esimation of pseudo-outcome in k-th fold 
            m0_pred = m_curr_model.predict(np.column_stack((x_test,np.zeros([len(idx_te),1]))))
            m1_pred = m_curr_model.predict(np.column_stack((x_test,np.ones([len(idx_te),1]))))
            pt_pred = pt_curr_model.predict_proba(x_test)[:,pt_curr_model.classes_==1].ravel()
            m_pred = np.array([m0_pred[j] if trt_test[j]==0 else m1_pred[j] for j in range(len(idx_te))])
            self.pseudo_outcome[idx_te]  = ((trt_test-pt_pred)/(pt_pred*(1-pt_pred)))*(y_test-m_pred) + m1_pred - m0_pred
            self.y0_pred[idx_te] = m0_pred
            self.y1_pred[idx_te] = m1_pred
            self.p_pred[idx_te] = pt_pred

        self.ate_pred = np.mean(self.pseudo_outcome)

        # Regression on covariates
        self.model = self.cate_model(**cate_model_kwargs).fit(x,self.pseudo_outcome)

class PseudoIPWLearner(PseudoLearner):

    def __init__(self,k,seed,pt_model=sklearn.linear_model.LogisticRegression,cate_model=sklearn.linear_model.LinearRegression):
        self.k = k
        self.pt_model = pt_model
        self.cate_model = cate_model
        self.seed = seed

    def train(self,x,y,trt,**kwargs):
        '''
        x,y,trt: training data as numpy arrays
        kwargs: additional arguments for models - dictionary of the form: {'t':{arg1:value1, ...}, 'cate': {arg1:value1,...}}
        '''
        cv = StratifiedKFold(n_splits=self.k,shuffle=True,random_state=self.seed)
        self.cv = cv
        n = x.shape[0]
        self.pseudo_outcome = np.nan*np.ones([n])
        self.y0_pred = np.nan*np.ones([n])
        self.y1_pred = np.nan*np.ones([n])
        self.p_pred = np.nan*np.ones([n])
        self.p_trained_models = []

        pt_model_kwargs = kwargs.get('t',{})
        cate_model_kwargs = kwargs.get('cate',{})
        # Overwrite seed if required by the model
        if 'seed' in inspect.getfullargspec(self.pt_model).args:
            pt_model_kwargs['seed'] = self.seed
        if 'seed' in inspect.getfullargspec(self.cate_model).args:
            cate_model_kwargs['seed'] = self.seed

        for i, (idx_tr, idx_te) in enumerate(cv.split(x, trt)):

            x_train = x[idx_tr]
            x_test = x[idx_te]
            trt_train = trt[idx_tr]
            trt_test = trt[idx_te]
            y_test = y[idx_te]

            # Training of propensity model in k-1 folds
            pt_curr_model = self.pt_model(**pt_model_kwargs).fit(x_train,trt_train)
            self.p_trained_models.append(pt_curr_model)

            # Estimation of pseudo-outcome in k-th fold
            pt_pred = pt_curr_model.predict_proba(x_test)[:,pt_curr_model.classes_==1].ravel()
            psi = ((trt_test-pt_pred)/(pt_pred*(1-pt_pred)))*y_test
            self.pseudo_outcome[idx_te] = psi
            self.p_pred[idx_te] = pt_pred
            self.y0_pred[idx_te] = ((1-trt_test)/(1-pt_pred))*y_test
            self.y1_pred[idx_te] = (trt_test/pt_pred)*y_test

        self.ate_pred = np.mean(self.pseudo_outcome)

        # Regression on covariates
        self.model = self.cate_model(**cate_model_kwargs).fit(x,self.pseudo_outcome)

class PseudoTLearner(PseudoLearner):

    def __init__(self,k,seed,m1_model=sklearn.linear_model.LinearRegression,m0_model=sklearn.linear_model.LinearRegression,cate_model=sklearn.linear_model.LinearRegression):
        self.k = k
        self.m1_model = m1_model
        self.m0_model = m0_model
        self.cate_model = cate_model
        self.seed = seed

    def train(self,x,y,trt,**kwargs):
        '''
        x,y,trt: training data as numpy arrays
        kwargs: additional arguments for models - dictionary of the form: {'y1':{arg1:value1, ...},'y0':{arg1:value1, ...}, 'cate': {arg1:value1,...}}
        '''
        cv = StratifiedKFold(n_splits=self.k,shuffle=True,random_state=self.seed)
        self.cv = cv
        n = x.shape[0]
        self.pseudo_outcome = np.nan*np.ones([n])
        self.y0_pred = np.nan*np.ones([n])
        self.y1_pred = np.nan*np.ones([n])

        m1_model_kwargs = kwargs.get('y1',{})
        m0_model_kwargs = kwargs.get('y0',{})
        cate_model_kwargs = kwargs.get('cate',{})
        # Overwrite seed if required by the model
        if 'seed' in inspect.getfullargspec(self.m1_model).args:
            m1_model_kwargs['seed'] = self.seed
        if 'seed' in inspect.getfullargspec(self.m0_model).args:
            m0_model_kwargs['seed'] = self.seed
        if 'seed' in inspect.getfullargspec(self.cate_model).args:
            cate_model_kwargs['seed'] = self.seed

        for i, (idx_tr, idx_te) in enumerate(cv.split(x, trt)):

            x_train = x[idx_tr]
            x_test = x[idx_te]
            trt_train = trt[idx_tr]
            y_train = y[idx_tr]

            # Training of outcome models in k-1 folds
            m1_model = self.m1_model(**m1_model_kwargs).fit(x_train[trt_train==1],y_train[trt_train==1])
            m0_model = self.m0_model(**m0_model_kwargs).fit(x_train[trt_train==0],y_train[trt_train==0])

            # Predict pseudo-outcome in k-th fold
            m1_pred = m1_model.predict(x_test)
            m0_pred = m0_model.predict(x_test)
            self.pseudo_outcome[idx_te] = m1_pred - m0_pred
            self.y1_pred[idx_te] = m1_pred
            self.y0_pred[idx_te] = m0_pred

        self.ate_pred = np.mean(self.pseudo_outcome)
        # Regression on covariates
        self.model = self.cate_model(**cate_model_kwargs).fit(x,self.pseudo_outcome)
    
class PseudoSLearner(PseudoLearner):

    def __init__(self,k,seed,m_model=sklearn.linear_model.LinearRegression,cate_model=sklearn.linear_model.LinearRegression):
        self.k = k
        self.m_model = m_model
        self.cate_model = cate_model
        self.seed = seed

    def train(self,x,y,trt,**kwargs):
        '''
        x,y,trt: training data as numpy arrays
        kwargs: additional arguments for models - dictionary of the form: {'y':{arg1:value1, ...}, 'cate': {arg1:value1,...}}
        '''
        cv = StratifiedKFold(n_splits=self.k,shuffle=True,random_state=self.seed)
        self.cv = cv
        n = x.shape[0]
        x_outcome = np.column_stack((x,trt))
        self.pseudo_outcome = np.nan*np.ones([n])
        self.y0_pred = np.nan*np.ones([n])
        self.y1_pred = np.nan*np.ones([n])
        self.y_trained_models = []

        m_model_kwargs = kwargs.get('y',{})
        cate_model_kwargs = kwargs.get('cate',{})
        # Overwrite seed if required by the model
        if 'seed' in inspect.getfullargspec(self.m_model).args:
            m_model_kwargs['seed'] = self.seed
        if 'seed' in inspect.getfullargspec(self.cate_model).args:
            cate_model_kwargs['seed'] = self.seed

        for i, (idx_tr, idx_te) in enumerate(cv.split(x, trt)):

            x_train = x_outcome[idx_tr]
            x_test = x_outcome[idx_te]
            y_train = y[idx_tr]

            # Training of outcome model in k-1 folds
            m_curr_model = self.m_model(**m_model_kwargs).fit(x_train,y_train)

            # Predict pseudo-outcome in k-th fold
            m0_pred = m_curr_model.predict(np.column_stack((x_test,np.zeros([len(idx_te),1]))))
            m1_pred = m_curr_model.predict(np.column_stack((x_test,np.ones([len(idx_te),1]))))
            self.pseudo_outcome[idx_te] = m1_pred - m0_pred
            self.y1_pred[idx_te] = m1_pred
            self.y0_pred[idx_te] = m0_pred
            self.y_trained_models.append(m_curr_model)

        self.ate_pred = np.mean(self.pseudo_outcome)
        # Regression on covariates
        self.model = self.cate_model(**cate_model_kwargs).fit(x,self.pseudo_outcome)