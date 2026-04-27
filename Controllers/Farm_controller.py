from flask import request,jsonify
from Models.Farm import Farm
from Models.Fruit import Fruit
from db import db

class FarmController:
    @staticmethod
    def add_farm():

        data = request.get_json()


        try:
            new_farm = Farm(
                Landlord_id=data['Landlord_id'],
                name=data['name'],
                type=data['type'],
                province=data['province'],
                city=data['city']
            )
            db.session.add(new_farm)
            db.session.commit()
            return jsonify({
                "message": "Farm added successfully",
                "landlord_id": new_farm.Landlord_id,
                "farm_id":new_farm.id
            }),200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @staticmethod
    def get_all_farms():
        data=request.get_json()
        l_id=data["landlord_id"]
        farms = Farm.query.filter(Farm.Landlord_id==l_id).all()
        farm_list = []
        for farm in farms:
            farm_list.append({
                "id": farm.id,
                "Landlord_id": farm.Landlord_id,
                "name": farm.name,
                "province": farm.province,
                "city":farm.city,
                "type": farm.type
            })

        return jsonify(farm_list)

    @staticmethod
    def get_farmby_id(farm_id):
        farm=Farm.query.get(farm_id)
        if not farm:
            return jsonify({'farm not found'}),500
        return jsonify({
            "id": farm.id,
            "Landlord_id": farm.Landlord_id,
            "name": farm.name,
            "province": farm.province,
            "city": farm.city,
            "type": farm.type
        })

    @staticmethod
    def get_farm_by_location():
        data=request.get_json()
        city=data.get('city')
        farm=Farm.query.filter_by(city=city).all()
        if not farm:
            return jsonify({"message": "farm not found in this city"})
        result = []
        for farm in farm:
            result.append({
                "id": farm.id,
                "name": farm.name,
                "type": farm.type,
                "province": farm.province,
                "city": farm.city,            })
        return jsonify(result)

    @staticmethod
    def update_farm(farm_id):
        farm = Farm.query.get(farm_id)
        if not farm:
            return jsonify({"error": "Farm not found"})
        data = request.get_json()

        if "name" in data:
            farm.name = data["name"]
        if "city" in data:
            farm.location = data["city"]
        if "type" in data:
            farm.type = data["type"]

        db.session.commit()
        return jsonify({"message": "Farm updated successfully"})

    # @staticmethod
    # def delete_fruit():
    #     data = request.get_json()
    #     farm_id = data.get('farm_id')
    #     if not farm_id:
    #         return jsonify({"message": "farm_id is required"}), 400
    #
    #     farm = Farm.query.filter_by(id=farm_id).first()
    #     if not farm:
    #         return jsonify({"message": "farm not found"}), 404
    #
    #     db.session.delete(farm)
    #     db.session.commit()
    #     return jsonify({"message": "farm deleted successfully"}), 200
    #
