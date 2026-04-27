from datetime import datetime

from flask import jsonify, request
from Models.Batch import Batch
from Models.Farm_Fruit_Batch import Farm_Fruit_Batch
from Models.Farm import Farm
from Models.Fruit import Fruit
from db import db
from sqlalchemy import  desc, extract


class BatchController:

    @staticmethod
    def add_batch():

        data = request.get_json()
        new_batch = Batch(
            total_weight=data.get("total_weight"),
            timestamp=datetime.utcnow(),
            class_A=data.get("class_A"),
            class_B=data.get("class_B"),
            class_C=data.get("class_C")
        )
        db.session.add(new_batch)
        db.session.commit()
        farm_fruit_batch_entry = Farm_Fruit_Batch(
            farm_id=data["farm_id"],
            fruit_id=data["fruit_id"],
            batch_id=new_batch.id
        )
        db.session.add(farm_fruit_batch_entry)
        db.session.commit()

        return jsonify({
            "message": "Batch added successfully",
            "batch_id": new_batch.id,

        }), 201
    @staticmethod
    def compare_batches():
        total_production = (Batch.class_A + Batch.class_B + Batch.class_C).label("total_production")
        # query with ordering
        results = db.session.query(
            Batch.id,
            Batch.class_A,
            Batch.class_B,
            Batch.class_C,
            total_production
        ).order_by(desc(total_production)).all()
        result = []
        for b in results:
            result.append({
                "batch_id": b.id,
                "class_A": b.class_A,
                "class_B": b.class_B,
                "class_C": b.class_C,
                "total_production": b.total_production
            })

        return jsonify(result)

    @staticmethod
    def best_batch_of_year():
        data = request.get_json()
        year = data.get("year")

        if not year:
            return jsonify({"message": "Year is required"}), 400

        total_production = (Batch.class_A + Batch.class_B + Batch.class_C)

        b = db.session.query(
            Batch.id,
            Batch.class_A,
            Batch.class_B,
            Batch.class_C,
            Batch.timestamp,
            total_production.label("total_production")
        ).filter(
            extract('year', Batch.timestamp) == year
        ).order_by(
            desc(total_production)
        ).first()

        if not b:
            return jsonify({"message": "No batch found for this year"})

        return jsonify({
            "batch_id": b.id,
            "class_A": b.class_A,
            "class_B": b.class_B,
            "class_C": b.class_C,
            "total_production": b.total_production,
            "date": b.timestamp.strftime("%Y-%m-%d")
        })

    @staticmethod
    def get_all_batch():
        batch = Batch.query.all()
        batch_list = []
        for batch in batch:
            batch_list.append({
                "id": batch.id,
                "class_A":batch.class_A,
                "class_B":batch.class_B,
                "class_c":batch.class_C,
                "timestamp":batch.timestamp
            })

        return jsonify(batch_list)

    @staticmethod
    def get_batches_report():

        data = request.get_json()
        selected_farm_id = data.get("farm_id")

        query = db.session.query(
            Batch.id.label('batch_id'),
            Farm.name.label('farm_name'),
            Fruit.name.label('fruit_name'),
            Batch.class_A,
            Batch.class_B,
            Batch.class_C,
            Batch.timestamp
        ).join(
            Farm_Fruit_Batch, Farm_Fruit_Batch.batch_id == Batch.id
        ).join(
            Farm, Farm.id == Farm_Fruit_Batch.farm_id
        ).join(
            Fruit, Fruit.id == Farm_Fruit_Batch.fruit_id
        )
        if selected_farm_id:
            query = query.filter(Farm.id == selected_farm_id)

        batch = query.order_by(Batch.timestamp.desc()).all()

        result = []
        for r in batch:
            result.append({
                "batch_id": r.batch_id,
                "farm_name": r.farm_name,
                "fruit_name": r.fruit_name,
                "class_A": r.class_A,
                "class_B": r.class_B,
                "class_C": r.class_C,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            })

        return jsonify(result)



