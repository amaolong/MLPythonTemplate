import numpy as np
class stack_model():

    def __init__(self,X,y,model_list_lv1,model_list_lv2,model_list_lv3=None):
        self.X=X
        self.y=y
        self.lv1_model_list=model_list_lv1
        self.lv2_model_list=model_list_lv2
        self.lv3_model_list=model_list_lv3
        print('number of level 1 models: ',len(self.lv1_model_list))
        print('number of level 2 models: ', len(self.lv2_model_list))
        if self.lv3_model_list!=None:
            print('number of level 3 models: ', len(self.lv3_model_list))

    # by level implementation for debugging and alternative structure
    def fit1(self):
        for i, _ in enumerate(self.lv1_model_list):
            print('fitting level 1 model - ',i+1,' out of ',len(self.lv1_model_list))
            _.fit(self.X, self.y)
    def fit2(self):
        # first layer
        self.fit1() # train
        t_res =self.predict1(self.X) # predict
        # second layer
        for i, _ in enumerate(self.lv2_model_list):
            print('fitting level 2 model - ', i+1, ' out of ', len(self.lv2_model_list))
            _.fit(t_res,self.y)
    def fit3(self):
        # first two layers
        self.fit2() # train
        t_res=self.predict2(self.X) # predict
        # third layer
        for i, _ in enumerate(self.lv3_model_list):
            print('fitting level 3 model - ', i+1, ' out of ', len(self.lv3_model_list))
            _.fit(t_res, self.y)
    #
    def predict1(self,X):
        t_res = []
        if len(self.lv1_model_list) > 1:
            for i, _ in enumerate(self.lv1_model_list):
                if i == 0:
                    t_res = _.predict_proba(X)
                else:
                    t_res = np.hstack([t_res, _.predict_proba(X)])
        else:
            t_res = self.lv1_model_list[0].predict(t_res)
        return t_res
    def predict2(self,X):
        # first layer
        t_res = self.predict1(X)
        # second layer
        t_res_2 = []
        if len(self.lv2_model_list)>1:
            for i, _ in enumerate(self.lv2_model_list):
                if i == 0:
                    t_res_2 = _.predict_proba(t_res)
                else:
                    t_res_2 = np.hstack([t_res_2, _.predict_proba(t_res)])
        else:
            t_res_2 = self.lv2_model_list[0].predict(t_res)
        return t_res_2
    def predict3(self,X):
        # first two layer
        t_res_2 = self.predict2(X)
        t_res_3 = []
        if len(self.lv3_model_list)>1:
            for i, _ in enumerate(self.lv3_model_list):
                if i == 0:
                    t_res_3 = _.predict_proba(t_res_2)
                else:
                    t_res_3 = np.hstack([t_res_3, _.predict_proba(t_res_2)])
        else:
            t_res_3=self.lv3_model_list[0].predict(t_res_2)
        return t_res_3
    #
    # fixed structure with raw features
    def fit2_w_raw_features(self):
        # first layer
        self.fit1() # train
        t_res =np.hstack([self.X,self.predict1(self.X)]) # predict
        # second layer
        for i, _ in enumerate(self.lv2_model_list):
            print('fitting level 2 model - ', i+1, ' out of ', len(self.lv2_model_list))
            _.fit(t_res,self.y)
    def fit3_w_raw_features(self):
        # first two layers
        self.fit2_w_raw_features()  # train
        t_res = np.hstack([self.X,self.predict_2(self.X)])  # predict
        # third layer
        for i, _ in enumerate(self.lv3_model_list):
            print('fitting level 3 model - ', i+1, ' out of ', len(self.lv3_model_list))
            _.fit(t_res, self.y)
    #
    def predict_2_w_raw_features(self, X):
        # first layer
        t_res = np.hstack([X,self.predict1(X)])
        # second layer
        t_res_2 = []
        if len(self.lv2_model_list)>1:
            for i, _ in enumerate(self.lv2_model_list):
                if i == 0:
                    t_res_2 = _.predict_proba(t_res)
                else:
                    t_res_2 = np.hstack([t_res_2, _.predict_proba(t_res)])
        else:
            t_res_2 = self.lv2_model_list[0].predict(t_res)
        return t_res_2
    def predict_3_w_raw_features(self,X):
        # first two layer
        t_res_2 = np.hstack([X,self.predict2(X)])
        t_res_3 = []
        if len(self.lv3_model_list)>1:
            for i, _ in enumerate(self.lv3_model_list):
                if i == 0:
                    t_res_3 = _.predict_proba(t_res_2)
                else:
                    t_res_3 = np.hstack([t_res_3, _.predict_proba(t_res_2)])
        else:
            t_res_3=self.lv3_model_list[0].predict(t_res_2)
        return t_res_3
    #
    def get_best_level_1_estimator(self):
        pass
    def get_best_level_2_estimator(self):
        pass
