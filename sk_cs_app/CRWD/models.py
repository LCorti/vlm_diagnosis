from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key = True)
    prolific_pid = db.Column(db.String(100), unique = True)
    optout = db.Column(db.Boolean, default = False, nullable = False)
    annotations = db.relationship('Annotation', backref = 'user')

class Image(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    imgpath = db.Column(db.String(100), nullable = False)
    question = db.Column(db.String(100), nullable = False)
    dataset = db.Column(db.String(100), nullable = False)
    category = db.Column(db.String(100), nullable = False)
    locked = db.Column(db.Integer)
    triples = db.relationship('Triple', backref = 'image')

class Triple(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    id_json = db.Column(db.String(100), nullable = False)
    entity1 = db.Column(db.String(100), nullable = False)
    coords_entity1 = db.Column(db.String(100), nullable = False)
    relationship = db.Column(db.String(100), nullable = False)
    entity2 = db.Column(db.String(100), nullable = False)
    coords_entity2 = db.Column(db.String(100), nullable = False)
    new_user_annotation = db.Column(db.Boolean)
    id_new_user_annotation = db.Column(db.String(100))
    id_image = db.Column(db.Integer, db.ForeignKey('image.id'))
    annotations = db.relationship('Annotation', backref = 'triple')

class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    correctly_identified = db.Column(db.Boolean)
    new_label_entity1 = db.Column(db.String(100))
    new_label_entity2 = db.Column(db.String(100))
    new_coords_entity1 = db.Column(db.String(100))
    new_coords_entity2 = db.Column(db.String(100))
    entities_in_relationship = db.Column(db.Boolean)
    correct_relationship = db.Column(db.Boolean)
    new_relationship = db.Column(db.String(100))
    answer_question = db.Column(db.Boolean)
    id_triple = db.Column(db.Integer, db.ForeignKey('triple.id'))
    id_user = db.Column(db.Integer, db.ForeignKey('user.id'))
