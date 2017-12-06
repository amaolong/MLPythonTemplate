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
        # intialize sample weight and lv1_rank_list to be empty
        self.lv1_rank_list=None
        self.sample_weight=None

    # load validation, sample weight, and other stuffs
    def load_validation_set(self,X1,y1):
        self.X1=X1
        self.y1=y1
    def load_sample_weight(self,weight):
        self.sample_weight=weight
    def additional_keywords(self,dict):
        '''
        this is mainly for swtich xgb/lgbm's objective function
        :param dict:
        :return:
        '''
        self.additional_keywords=dict
        # plug in those keywords into existing model after key comparison
        for _ in self.lv1_model_list:
            tmp_dict={}
            for _2 in _.get_params().keys():
                if _2 in dict.keys():
                    tmp_dict[_2]=dict[_2]
            _.set_params(**tmp_dict)

    # fit and predict
    def fit1(self):
        for i, _ in enumerate(self.lv1_model_list):
            print('fitting level 1 model - ',i+1,' out of ',len(self.lv1_model_list))
            if self.sample_weight==None:
                _.fit(self.X, self.y)
            else:
                _.fit(self.X,self.y,self.sample_weight)
    def fit2(self):
        # first layer
        self.fit1() # train
        t_res =self.predict1(self.X) # predict
        # second layer
        for i, _ in enumerate(self.lv2_model_list):
            print('fitting level 2 model - ', i+1, ' out of ', len(self.lv2_model_list))
            if self.sample_weight==None:
                _.fit(t_res,self.y)
            else:
                _.fit(t_res, self.y,self.sample_weight)
    def fit3(self):
        # first two layers
        self.fit2() # train
        t_res=self.predict2(self.X) # predict
        # third layer
        for i, _ in enumerate(self.lv3_model_list):
            print('fitting level 3 model - ', i+1, ' out of ', len(self.lv3_model_list))
            if self.sample_weight==None:
                _.fit(t_res, self.y)
            else:
                _.fit(t_res, self.y,self.sample_weight)
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
    # fit and predict with raw features and fixed structure
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
        self.fit2()  # train
        t_res = np.hstack([self.X,self.predict2(self.X)])  # predict
        # third layer
        for i, _ in enumerate(self.lv3_model_list):
            print('fitting level 3 model - ', i+1, ' out of ', len(self.lv3_model_list))
            _.fit(t_res, self.y)
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

    # evaluation and trimming
    def evaluate_level_1_base_learner(self):
        scorelist=[]
        for _ in self.lv2_model_list:
            if self.sample_weight==None:
                scorelist.append(_.score(self.X1,self.y1))
            else:
                scorelist.append(_.score(self.X1,self.y1,self.sample_weight))
        #
        ranklist=np.ones(len(scorelist))*len(scorelist)-argsort(scorelist)
        ranklist=[int(_) for _ in ranklist]
        self.lv1_rank_list=ranklist
        self.lv1_score_list=scorelist
        self.best_lv1_estimator=self.lv1_model_list[ranklist[0]]
        ''' report following numbers '''
        # best score with model parameters
        print('best lv1 learner: ', ' score: ', scorelist[ranklist[0]], 'model: ', self.lv1_model_list[ranklist[0]])
        # worst score with model parameters
        print('worst lv1 learner: ', ' score: ', scorelist[ranklist[-1]], 'model: ', self.lv1_model_list[ranklist[-1]])
        # mean
        print('score mean: ',np.mean(scorelist))
        # std
        print('score std: ',np.std(scorelist))
        print('score percentile at 10%, 20%, and 30% (the larger the better) ', np.percentile(scorelist,[10,20,30]))
    def trim_level_1_base_learner(self, number_threshold=None, score_threshold=None):
        '''
        :param N:
        :param score:
        :return:
        '''
        if number_threshold!=None:
            if number_threshold>len(self.lv1_rank_list):
                print('number threshold too large, including all level 1 learners')
            else:
                del self.lv1_model_list[self.lv1_rank_list[number_threshold:]]
                print('level 1 learner size: ', len(self.lv1_model_list))
        if score_threshold!=None:
            remove_list=[]
            for _ in self.lv1_rank_list[::-1]:
                if self.lv1_score_list[_]<score_threshold:
                    remove_list.append(_)
                else:
                    break
            del self.lv1_model_list[remove_list]
            print('level 1 learner size: ', len(self.lv1_model_list))

    # override lv 2/3 output layer estimators/parameters
    def override_lv2_learner(self):
        '''
        this will override lv 2 estimator with best structure in lv 1
        :return:
        '''
        if self.best_lv1_estimator.__class__==self.lv2_model_list[0].__class__:
            self.lv2_model_list[0]=self.best_lv1_estimator.__class__()
            self.lv2_model_list[0].set_params(**self.best_lv1_estimator.get_params())
        else:
            print('change model from ',self.lv2_model_list[0].__class__, ' to ', self.best_lv1_estimator.__class__)
            self.lv2_model_list[0] = self.best_lv1_estimator.__class__()
            self.lv2_model_list[0].set_params(**self.best_lv1_estimator.get_params())
    def override_lv3_learner(self):
        '''
        this will override lv 3 estimator with best structure in lv 1
        :return:
        '''
        if self.best_lv1_estimator.__class__==self.lv3_model_list[0].__class__:
            self.lv3_model_list[0]=self.best_lv1_estimator.__class__()
            self.lv3_model_list[0].set_params(**self.best_lv1_estimator.get_params())
        else:
            print('change model from ',self.lv3_model_list[0].__class__, ' to ', self.best_lv1_estimator.__class__)
            self.lv3_model_list[0] = self.best_lv1_estimator.__class__()
            self.lv3_model_list[0].set_params(**self.best_lv1_estimator.get_params())

