from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_db(app):
    app.config['SQLALCHEMY_DATABASE_URI'] = (
        "mssql+pyodbc://DESKTOP-CJ2TM4F\\SQLEXPRESS/Fruitalyzer"
        "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
