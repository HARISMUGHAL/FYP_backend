from db import db

class Fruit(db.Model):
    __tablename__="fruits"
    id=db.Column(db.Integer,primary_key=True)
    name=db.Column(db.String(100),nullable=False)

    farm_fruit_batch_rls=db.relationship("Farm_Fruit_Batch",back_populates="fruit")




