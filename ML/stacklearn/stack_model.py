import numpy as np
class stack_model():
    #
    X=[]
    y=[]
    # list of models
    lv1_model_list = []
    lv2_model_list = []
    lv3_model_list = []

    def __init__(self,X,y,model_list_lv1,model_list_lv2,model_list_lv3=None):
        self.X=X
        self.y=y
        self.lv1_model_list=model_list_lv1
        self.lv2_model_list=model_list_lv2
        self.lv3_model_list=model_list_lv3

    # by level implementation for debugging and alternative structure
    def fit1(self):
        for i, _ in enumerate(self.lv1_model_list):
            _.fit(self.X, self.y)
    def fit2(self):
        # first layer
        self.fit1(self.X,self.y) # train
        t_res =self.predict1(self.X) # predict
        # second layer
        for i, _ in enumerate(self.lv2_model_list):
            _.fit(t_res,self.y)
    def fit3(self):
        # first two layers
        self.fit2(self.X,self.y) # train
        t_res=self.predict(self.X) # predict
        # second layer
        for i, _ in enumerate(self.lv3_model_list):
            _.fit(t_res, self.y)
    def predict1(self,X):
        t_res = []
        for i, _ in enumerate(self.lv1_model_list):
            if i == 0:
                t_res = _.predict_proba(X)
            else:
                t_res = np.vstack([t_res, _.predict_proba(X)])
        return t_res
    def predict2(self,X):
        # first layer
        t_res = self.predict1(X)
        # second layer
        t_res_2 = []
        for i, _ in enumerate(self.lv2_model_list):
            if i == 0:
                t_res_2 = _.predict_proba(t_res)
            else:
                t_res_2 = np.vstack([t_res_2, _.predict_proba(t_res)])
        return t_res_2
    def predict3(self,X):
        # first two layer
        t_res_2 = self.predict2(X)
        t_res_3 = []
        for i, _ in enumerate(self.lv2_model_list):
            if i == 0:
                t_res_3 = _.predict_proba(t_res_2)
            else:
                t_res_3 = np.vstack([t_res_3, _.predict_proba(t_res_2)])
        return t_res_3

    # full implementation below (fixed structure)
    def fit_2layers(self):
        self.fit1(self.X,self.y)
        t_res =self.predict1(self.X)
        # second layer
        self.lv2_model_list[0].fit(t_res,self.y)
    def fit_3layers(self):
        # first two layers
        self.fit_2layers(self.X,self.y)
        t_res=self.predict2(self.X)
        # third layer
        self.lv3_model_list[0].fit(t_res,self.y)
    def predict_2layers(self, X):
        # first layer
        t_res = self.predict1(X)
        # second layer
        return self.lv2_model_list[0].predict(t_res)
    def predict_2layers_w_raw_features(self, X):
        # first layer
        t_res = np.hstack(X, self.predict1(X))
        # second layer
        return self.lv2_model_list[0].predict(t_res)
    def predict_3layers(self,X):
        # first two layers
        t_res=self.predict_2layers(X)
        # third layer
        return self.lv3_model_list[0].predict(t_res)
    def predict_3layers_w_raw_features(self,X):
        # first two layers
        t_res=np.hstack(X,self.predict_2layers(X))
        # third layer
        return self.lv3_model_list[0].predict(t_res)
