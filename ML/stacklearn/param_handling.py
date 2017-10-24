import numpy as np

class model_param(object):
    model_name=''
    number_of_combination=0
    param_combination = []
    unique_combinations=0
    loadtype=0         # 0:sklearn, 1:xgb and lgbm,
                       # 0 initialize model with model(**param_dict),
                       # 1 initialize model with model(param_dict) directly

    sampled_model_params=[]

    def __init__(self,name,val,loadtype):
        self.model_name=name
        self.unique_combinations=val
        self.loadtype=loadtype

    def insert(self,_dict):
        self.param_combination.append(_dict)
        self.number_of_combination+=1

def populate_params(param_collection,param_collection_names,load_type):
    '''
    :param param_collection:
    :return:
    '''
    #
    param_combination=[]
    # populate each combination for different models
    collection=param_collection
    collection_names=param_collection_names
    for idx, _ in enumerate(collection):
        # get length, keys, unique combinations
        keys=list(_.keys())      # keys of individual model
        combination_numbers=[]  # available parameters for each key
        combination_total=1     # total parameter combination
        for _key in keys:
            combination_numbers.append(len(_[_key]))
        for _iter in combination_numbers:
            combination_total=np.multiply(combination_total,_iter)
        #
        # model_param object
        t = model_param(collection_names[idx], combination_total,load_type)
        #
        # generate parameter index
        r_seed=[]
        rvec=[]
        prev_size = 0
        for _2 in range(len(combination_numbers)):
            r_seed.append(0)
        #
        for idx1 in range(len(combination_numbers)):
            _size = len(rvec)
            for i in range(combination_numbers[idx1]):
                if idx1 == 0:
                    tmp = []
                    for idx2 in range(len(combination_numbers)):
                        if idx2 != idx1:
                            tmp.append(r_seed[idx2])
                        else:
                            tmp.append(i)
                    rvec.append(tmp)
                else:
                    for _2 in range(prev_size,_size):
                        tmp = []
                        for idx2 in range(len(combination_numbers)):
                            if idx2 != idx1:
                                tmp.append(rvec[_2][idx2])
                            else:
                                tmp.append(i)
                        rvec.append(tmp)
            prev_size=_size
        rvec=rvec[prev_size:]   # last couple enumeration are the unique ones
        #
        # print(param_collection_names[idx],', current: ',len(rvec), ', unique: ',combination_total)
        rvec=np.unique(rvec,axis=0)
        assert len(rvec)==combination_total, 'unique combinations not equal'
        #
        # converting parameter index to parameter dictionaries
        for param_idx in rvec:
            tdict={}
            for t_idx,val in enumerate(param_idx):
                tdict[keys[t_idx]]=_[keys[t_idx]][val]
            t.insert(tdict)
        #
        param_combination.append(t)
    #
    return param_combination

def sample_model(model_params_object):
    sampled_model_params=[]   # object containing list of dictionaries

    '''
    sample 2-4 different model from each category
    :param model_params_by_type:
    :return:
    '''

    for _ in model_params_object:
        size=len(_.param_combination)
        sample_size=0
        if (size>10):
            sample_size=3
        else:
            sample_size=2
        idx=[]
        #
        while(len(np.unique(idx))<sample_size):  # getting non duplicated index
            idx=np.random.randint(0,size-1,sample_size)
        for _2 in idx:
            _.sampled_model_params.append(_.param_combination[_2])
        #
    return sampled_model_params

