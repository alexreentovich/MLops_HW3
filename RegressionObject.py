import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
from database import api



class RegressionObject():
    def __init__(self, db_session, ML_model):
        self.db_session = db_session
        self.ML_model = ML_model
    
    def get_pred(self, id, request):
        '''Make prediction for given dataset contained in \
            request and regression id'''
        my_query = self.db_session.query(self.ML_model).get(id)
        if my_query is not None:
            parsed = request
            if 'Data' not in set(parsed.keys()):
                api.abort(400, "Wrong input format")
            X_test = pd.DataFrame(parsed['Data'])
            regr = pickle.loads(my_query.model)
            if X_test.shape[1] != len(regr.coef_):
                api.abort(400, "Data contains wrong number of features")
            if X_test.isnull().values.any():
                api.abort(400, "Data contains NaN")
            if not X_test.applymap(np.isreal).values.all():
                api.abort(400, "Data contains non numeric values")
            try:
                pred = regr.predict(X_test)
            except Exception as e:
                api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                          Try another input")
            return pd.DataFrame(pred).to_json(), 200
        else:
            api.abort(404, "Regression {} doesn't exist".format(id))

    def create(self, request):
        '''Train regression on given dataset with given hyperparameters \
            (contained in request) and save it'''
        parsed = request
        required_keys = set(['Data', 'Model_class', 'Hyperparam_dict'])
        if set(parsed.keys()) != required_keys:
            api.abort(400, "Wrong input format")
        input_dataframe = pd.DataFrame(parsed['Data'])
        if input_dataframe.isnull().values.any():
            api.abort(400, "Data contains NaN")
        if not input_dataframe.applymap(np.isreal).values.all():
            api.abort(400, "Data contains non numeric values")
        X_train = input_dataframe.iloc[:, 1:]
        y_train = input_dataframe.iloc[:, 0]
        try:
            regr = getattr(linear_model, parsed['Model_class'])
            regr = regr(**parsed['Hyperparam_dict'])
            regr.fit(X_train, y_train)
        except Exception as e:
            api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                      Try another input")
        new_model = self.ML_model(model=pickle.dumps(regr))
        self.db_session.add(new_model)
        self.db_session.commit()
        if new_model.id is None:
            new_model.id = 1
        return f'Regression successfully trained and saved under id {new_model.id}', 200

    def update(self, id, request):
        '''Retrain regression with given id on a \
            new dataset contained in request'''
        parsed = request
        required_keys = set(['Data'])
        if set(parsed.keys()) != required_keys:
            api.abort(400, "Wrong input format")
        input_dataframe = pd.DataFrame(parsed['Data'])
        if input_dataframe.isnull().values.any():
            api.abort(400, "Data contains NaN")
        if not input_dataframe.applymap(np.isreal).values.all():
            api.abort(400, "Data contains non numeric values")
        X_train = input_dataframe.iloc[:, 1:]
        y_train = input_dataframe.iloc[:, 0]
        my_query = self.db_session.query(self.ML_model).get(id)
        if my_query is not None:
            regr = pickle.loads(my_query.model)
            try:
                regr = regr.fit(X_train, y_train)
            except Exception as e:
                api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                      Try another input")
            my_query.model = pickle.dumps(regr)
            self.db_session.commit()
            return f'Regression {id} successfully retrained', 200
        else:
            api.abort(404, "Regression {} doesn't exist".format(id))

    def remove(self, id):
        '''Delete a trained regression given its id'''
        my_query = self.db_session.query(self.ML_model).get(id)
        if my_query is not None:
            self.db_session.query(self.ML_model).filter_by(id=id).delete()
            #self.ML_model.query.filter_by(id=id).delete()
            self.db_session.commit()
            return f'Regression {id} successfully deleted', 204
        else:
            api.abort(404, "Regression {} doesn't exist".format(id))
