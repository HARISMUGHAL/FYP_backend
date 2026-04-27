from db import db

class Farm_Fruit_Batch(db.Model):
    __tablename__="Farm_Fruit_Batch"
    id=db.Column(db.Integer,primary_key=True)
    farm_id=db.Column(db.Integer,db.ForeignKey('farms.id'),nullable=False)
    fruit_id=db.Column(db.Integer,db.ForeignKey('fruits.id'),nullable=False)
    batch_id=db.Column(db.Integer,db.ForeignKey('batch.id'),nullable=False)

    farm=db.relationship("Farm",back_populates="farm_fruit_batch_rls")
    fruit=db.relationship("Fruit",back_populates="farm_fruit_batch_rls")
    batch=db.relationship("Batch",back_populates="farm_fruit_batch_rls")












