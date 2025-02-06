import logging.config
import yaml

with open('logging_config.yaml', 'rt') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

from flask import Flask, Blueprint, redirect, url_for, request
from flask_login import LoginManager
from models import db, User

app = Flask(__name__)

app_blueprint = Blueprint('app', __name__, url_prefix='/crwd', static_folder='static')

logger = logging.getLogger(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://crwd_user:crwd_pwd@"
    f"crwd_mysql:3307/CRWD_DB"
)

app.config['SECRET_KEY'] = 'secret-key-goes-here'

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'auth.consent'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app_blueprint.route('/')
def index():
    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == ""):
        return redirect(url_for('main.error', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first() != None and User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return redirect(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    return redirect(url_for('auth.consent', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

app.register_blueprint(app_blueprint)

from main import main_blueprint
app.register_blueprint(main_blueprint)

from auth import auth_blueprint
app.register_blueprint(auth_blueprint)
