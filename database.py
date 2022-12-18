from flask import Flask
from flask_restx import Api
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv

d = os.getcwd()
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "mybestsecret"
app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_DB')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
api = Api(app)

class ML_model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.LargeBinary, nullable=False)

db.create_all()