import pytest
from flask import Flask
from flask_restx import Api
from flask_sqlalchemy import SQLAlchemy
from RegressionObject import RegressionObject
from app import RegressionList
import os
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope='session')
def database(request):
    '''
    Create a Postgres database for the tests, and drop it when the tests are done.
    '''
    pg_host = DB_OPTS.get("host")
    pg_port = DB_OPTS.get("port")
    pg_user = DB_OPTS.get("username")
    pg_db = DB_OPTS["database"]

    init_postgresql_database(pg_user, pg_host, pg_port, pg_db)

    @request.addfinalizer
    def drop_database():
        drop_postgresql_database(pg_user, pg_host, pg_port, pg_db, 9.6)


@pytest.fixture(scope='session')
def app(database):
    '''
    Create a Flask app context for the tests.
    '''
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "mybestsecret"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_DB')}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    

    return app


@pytest.fixture(scope='session')
def _db(app):
    '''
    Provide the transactional fixtures with access to the database via a Flask-SQLAlchemy
    database connection.
    '''
    db = SQLAlchemy(app=app)

    return db
