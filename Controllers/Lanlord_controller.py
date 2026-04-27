from flask import request,jsonify
from Models.Landlord import Landlord
from db import db

class LandlordController:
    @staticmethod
    def sign_up():
        data=request.get_json()

        existing=Landlord.query.filter_by(email=data['email']).first()
        if  existing:
            return jsonify({'error':'email already exist'}) ,400
        new_user=Landlord(
            name=data['name'],
            email=data['email'],
            password=data['password'],
        )

        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message':'Signup successful','user_id':new_user.id}),200

    @staticmethod
    def login():
        data=request.get_json()
        user = Landlord.query.filter_by(email=data['email']).first()
        if not user:
            return jsonify({'error':'Email not found'}), 404
        if user.password != data['password']:
            return jsonify({'error':'Incorrect password'}), 401

        return jsonify({'message': 'Login successful','user_id':user.id}),200
    @staticmethod
    def update_profile(id):
        data = request.get_json()
        landlord=Landlord.query.get(id)
        if not landlord:    
            return jsonify({'error': 'User not found'}), 404
        if "name" in data:
            landlord.name = data["name"]
        if "city" in data:
            landlord.location = data["city"]
        if "phone_number" in data:
            landlord.phone_number = data["phone_number"]
        if "email" in data:
            landlord.email = data["email"]
        if "password" in data:
            landlord.password = data["password"]


        db.session.commit()
        return jsonify({'message': 'Profile updated successfully'})

    @staticmethod
    def get_landlord_by_id(l_id):
        landlord = Landlord.query.get(l_id)

        if not landlord:
            return jsonify({"error": "user not found"}), 404

        return jsonify({
            "id": landlord.id,
            "name": landlord.name,
            "email": landlord.email
        }), 200











