from flask_login import login_user, logout_user, login_required
from flask import render_template, Blueprint, request, url_for, redirect, request, jsonify
from models import db, User, Image

auth_blueprint = Blueprint('auth', __name__, url_prefix='/crwd', static_folder='static')

@auth_blueprint.route('/consent')
def consent():

    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == ""):
        return jsonify(url_for('main.error'))
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first() != None and User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return redirect(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    return render_template('consent.html', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID)

@auth_blueprint.route('/consent_post', methods = ('POST', ))
def consent_post():

    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == ""):
        return jsonify(url_for('main.error'))
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first() != None and User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return redirect(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    if request.form.get('outcome', "Reject") == "Reject":
        return redirect(url_for('auth.opt_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID, REJECT = True)) # ---> Inserire Link <---

    user = User.query.filter_by(prolific_pid = PROLIFIC_PID).first()
    if(user == None):
        user = User(
            prolific_pid = PROLIFIC_PID
        )
        db.session.add(user)
        db.session.commit()

    if(user.optout):
        return redirect(url_for('main.error', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    login_user(user, remember = True)

    # return redirect(url_for('main.crowdsourcing', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))
    return redirect(url_for('main.tutorial', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

@login_required
@auth_blueprint.route('/opt_out', methods = ('POST', 'GET'))
def opt_out():
    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    if (request.args.get('REJECT')):
        return redirect(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID, REJECT = True))
    else:
        if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == "" or User.query.filter_by(prolific_pid = PROLIFIC_PID).first() == None):
            return jsonify(url_for('main.error'))
        if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
            return redirect(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    user = User.query.filter_by(prolific_pid = PROLIFIC_PID).first()
    user.optout = True
    db.session.commit()

    to_unlock_images = Image.query.filter_by(locked = user.id).all()
    for img in to_unlock_images:
        img.locked = None
        db.session.commit()

    logout_user()

    return jsonify(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))
