from db import db

class Farm(db.Model):
    __tablename__="farms"
    id=db.Column(db.Integer,primary_key=True)
    Landlord_id=db.Column(db.Integer,db.ForeignKey('landlord.id'))
    name=db.Column(db.String(100),nullable=False)
    province=db.Column(db.String(100),nullable=False)
    city=db.Column(db.String(100),nullable=False)
    type=db.Column(db.String(100))

    landlord_rls = db.relationship("Landlord", back_populates="farm_rls")
    farm_fruit_batch_rls=db.relationship("Farm_Fruit_Batch",back_populates="farm")

