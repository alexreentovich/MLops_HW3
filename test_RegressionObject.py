import pytest
#from requests import put, get, delete, post, patch
#from flask import Flask
#from flask_restx import Api
from flask_sqlalchemy import SQLAlchemy
from RegressionObject import RegressionObject
from sklearn import datasets
import pandas as pd
import json
import os
#from conftest import mock_ML_model, mock_RegressionObject, mock_RegressionList
from mock_alchemy.mocking import UnifiedAlchemyMagicMock
db_session = UnifiedAlchemyMagicMock()

db = SQLAlchemy()
class ML_model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.LargeBinary, nullable=False)

# Prepare data for tests
diabetes_X, _ = datasets.load_diabetes(return_X_y=True)
norm_data = pd.DataFrame(diabetes_X,
                         columns=['zero', 'one', 'two', 'three',
                                  'four', 'five', 'six', 'seven', 'eight',
                                  'nine']).iloc[:, :10].to_dict()
norm_data_pred = pd.DataFrame(diabetes_X,
                         columns=['zero', 'one', 'two', 'three',
                                  'four', 'five', 'six', 'seven', 'eight',
                                  'nine']).iloc[:, 1:10].to_dict()
norm_parsed = {'Data': norm_data, 'Hyperparam_dict': {'fit_intercept': True},
               'Model_class': 'LinearRegression'}
norm_renew_parsed = {'Data': norm_data}
norm_pred_parsed = {'Data': norm_data_pred}

norm_new = json.dumps(norm_parsed)
norm_renew = json.dumps(norm_renew_parsed)
norm_pred = json.dumps(norm_pred_parsed)

# Define url for the api
baseUrl = 'http://localhost:4000/Regressions'

# TESTS FOR REGRESSIONOBJECT CLASS GIVEN NORMAL INPUT DATA
new_id = 0


def test_RegressionObject_create():
    global new_id
    reg_obj = RegressionObject(db_session, ML_model)
    res = reg_obj.create(norm_parsed)
    new_id = int(res[0].split(' ')[-1])
    assert res[1] == 200, res[1]
    assert new_id == 1, res


def test_RegressionObject_update():
    global new_id
    reg_obj = RegressionObject(db_session, ML_model)
    res = reg_obj.update(new_id, norm_renew_parsed)
    assert res[1] == 200, res[1]

def test_RegressionObject_pred():
    global new_id
    reg_obj = RegressionObject(db_session, ML_model)
    res = reg_obj.get_pred(new_id, norm_pred_parsed)
    assert res[1] == 200, res[1]

def test_RegressionObject_remove():
    global new_id
    reg_obj = RegressionObject(db_session, ML_model)
    res = reg_obj.remove(new_id)
    assert res[1] == 204, res[1]