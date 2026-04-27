
from db import db

class Landlord(db.Model):
    __tablename__="landlord"
    id=db.Column(db.Integer,primary_key=True)
    name=db.Column(db.String(100),nullable=False)
    email=db.Column(db.String(100),unique=True)
    password = db.Column(db.String(100), nullable=False)

    farm_rls=db.relationship("Farm",back_populates="landlord_rls")



