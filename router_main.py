from flask import Flask,jsonify

from db import db, init_db

from Models import Landlord
from Models import Farm
from Models import Batch
from Models import Farm_Fruit_Batch
from Models import Fruit
from Controllers.Lanlord_controller import LandlordController
from Controllers.Farm_controller import FarmController
from Controllers.Batch_controller import BatchController
from Controllers.FruitAnalyzer_controller import FruitAnalyzerController
app = Flask(__name__)
init_db(app)
with app.app_context():
    db.create_all()

@app.route("/")
def home():
    return "Fruitalyzer API Running"

# Landlord routes
@app.route('/sign_up', methods=['POST'])
def sign_up():
    return LandlordController.sign_up()
@app.route('/login',methods=['POST'])
def login():
    return LandlordController.login()
@app.route('/profile/<int:id>',methods=['PUT'])
def update_profile(id):
    return LandlordController.update_profile(id)
@app.route('/get_landlord_by_id/<int:l_id>',methods=['GET'])
def get_landlord_by_id(l_id):
    return LandlordController.get_landlord_by_id(l_id)

#farm routes
@app.route('/Add_farm',methods=['POST'])
def create_farm():
    return FarmController.add_farm()
@app.post("/get_all_farms_of_landlord")
def get_all_farms():
    return FarmController.get_all_farms()
@app.get("/farm/<int:farm_id>")
def get_farmby_id(farm_id):
    return FarmController.get_farmby_id(farm_id)
@app.put("/update_farm/<int:farm_id>")
def update_farm(farm_id):
    return FarmController.update_farm(farm_id)
@app.route('/farm_location',methods=['POST'])
def get_farm_by_location():
    return FarmController.get_farm_by_location()
# @app.delete('/delete_farm')
# def delete_farm():
#     return FarmController.delete_fruit()

#Batch Controller
@app.post('/add_batch')
def add_batch():
    return BatchController.add_batch()
@app.get('/compare_batch')
def compare_batches():
    return BatchController.compare_batches()
@app.post('/best_batch')
def best_batch_of_year():
    return BatchController.best_batch_of_year()
@app.get('/all_batch')
def get_all_batch():
    return BatchController.get_all_batch()
@app.get('/batch_report')
def get_batches_report():
    return BatchController.get_batches_report()

# FruitAnalyzer realtime APIs
@app.post('/predict')
def predict():
    return FruitAnalyzerController.predict()

@app.get('/batch')
def get_batch():
    # Reuse existing batch controller logic as-is.
    return BatchController.get_all_batch()

@app.get('/health')
def health():
    return FruitAnalyzerController.health()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)