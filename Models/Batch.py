from db import db

class Batch(db.Model):
    __tablename__="batch"
    id=db.Column(db.Integer,primary_key=True)
    total_weight=db.Column(db.String(100),nullable=False)
    timestamp=db.Column(db.DateTime)
    class_A = db.Column(db.Integer, nullable=True)
    class_B = db.Column(db.Integer, nullable=True)
    class_C = db.Column(db.Integer, nullable=True)

    farm_fruit_batch_rls = db.relationship("Farm_Fruit_Batch", back_populates="batch")











