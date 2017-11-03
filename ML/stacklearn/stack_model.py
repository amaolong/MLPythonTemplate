class stack_model():
    #
    X=[]
    y=[]
    # list of models
    lv1_model_list = []
    lv2_model_list = []
    lv3_model_list = []
    # list of predictions
    lv1_predictions=[]
    lv2_predictions = []
    lv3_predictions = []
    # model dict
    model_level_dict={}
    model_level_dict[1]=lv1_model_list
    model_level_dict[2] = lv2_model_list
    model_level_dict[3] = lv3_model_list
    # prediction dict
    model_pred_dict = {}
    model_pred_dict[1] = lv1_model_pred
    model_pred_dict[2] = lv2_model_pred
    model_pred_dict[3] = lv3_model_pred


    def __init__(self,X,y,model_list_lv1,model_list_lv2,model_list_lv3=None):
        self.X=X
        self.y=y
        self.lv1_model_list=model_list_lv1
        self.lv2_model_list=model_list_lv2
        self.lv3_model_list=model_list_lv3


    # by level implementation for debugging and alternative structure
    def fit1(self,X,y):
        pass

    def fit2(self,X=None):
        pass

    def fit3(self,X=None):
        pass

    def predict1(self,X):
        pass

    def predict2(self,X=None):
        pass

    def predict3(self,X=None):
        pass



    # full implementation below (fixed structure)
    def fit(self, X, y):
        pass

    def predict(self,X):
        pass


