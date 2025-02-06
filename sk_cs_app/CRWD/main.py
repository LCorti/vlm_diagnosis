from flask import render_template, request, url_for, redirect, Blueprint, jsonify
from flask_login import login_required
from models import db, User, Annotation, Triple, Image
from app import logger
import json
import numpy
import random

main_blueprint = Blueprint('main', __name__, url_prefix='/crwd', static_folder='static')
# number of [question, image, answer] validatated by each worker
max_assigned_tasks = 5
max_assigned_triples_per_task = 3
# number of triples required in the annotation step
max_assigned_ann_triples_per_task = 3

@login_required
@main_blueprint.route('/tutorial')
def tutorial():
    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == "" or User.query.filter_by(prolific_pid = PROLIFIC_PID).first() == None):
        return redirect(url_for('main.error'))
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return redirect(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    url = url_for('main.tutorial_post', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID)
    opt_out_url = url_for('auth.opt_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID)

    return render_template('tutorial.html', url = url, opt_out_url = opt_out_url, PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID)

@login_required
@main_blueprint.route('/tutorial_post', methods = ('POST', ))
def tutorial_post():
    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == "" or User.query.filter_by(prolific_pid = PROLIFIC_PID).first() == None):
        return redirect(url_for('main.error'))
    
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return redirect(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    return redirect(url_for('main.val_task', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

@login_required
@main_blueprint.route('/val_task')
def val_task():

    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    # Check if the url contains the required parameters
    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == "" or User.query.filter_by(prolific_pid = PROLIFIC_PID).first() == None):
        return redirect(url_for('main.error'))
    
    # Check if the user has opted out
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return redirect(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    num_performed_val_tasks = (
        Annotation.query
        .filter(Annotation.id_user == User.query.filter_by(prolific_pid=PROLIFIC_PID).first().id)
        .join(Triple, Annotation.id_triple == Triple.id)
        .with_entities(Triple.id_image)
        .distinct()
        .count()
    )
    num_performed_ann_tasks = (
        Triple.query
        .with_entities(Triple.id_image)
        .filter_by(id_new_user_annotation = PROLIFIC_PID)
        .distinct()
        .count()
    )
    
    # Check if the user has performed the annotation of the validated image
    if num_performed_val_tasks > num_performed_ann_tasks:
        return redirect(url_for('main.ann_task', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    progress = float(num_performed_val_tasks / max_assigned_tasks) * 100

    # Check if the user has performed the required number of tasks
    if(num_performed_val_tasks >= max_assigned_tasks):
        return redirect(url_for('main.final', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    img = get_low_annotation_image(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().id)

    if img == None:
        return redirect(url_for('main.final', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    question = img.question
    answer_option = None
    
    # Load the answers from the sampled_questions.json file
    dataset_string = 'prepareDB/datasets/' + img.dataset + '/sampled_questions.json'
    sampled_questions = []
    try:
        with open(dataset_string) as f:
            sampled_questions = json.load(f)
    except:
        # logger.error('Could not load the sampled questions file for dataset: ' + img.dataset)
        pass
    
    question_object = None
    for q in sampled_questions:
        if int(q['img'].split('.')[0]) == int(img.imgpath.split('.')[0]):
            question_object = q
            break
    
    if question_object is not None:
        if question_object['answer'] is not None:
            answer = question_object['answer']
            if question_object['options'] is not None:
                # This covers MMBench and SEED (MC questions)
                options = question_object['options']
                answer_option = options[answer] # if options is not None else None
            else:
                # This covers VQA v2 and LLaVa-Bench ('options' is None)
                answer_option = answer
        else:
            # Default (this should not happen based on the data)
            None

    chosen_triples = get_low_annotation_triples(img.id, User.query.filter_by(prolific_pid = PROLIFIC_PID).first().id)
    
    if len(chosen_triples) == 0:
        return redirect(url_for('main.final', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    triples_array = []
    for triple in chosen_triples:
        if triple.new_user_annotation == False:
            triples_array.append([
                [triple.entity1, triple.relationship, triple.entity2, triple.id],
                parse_coords(triple.coords_entity1),
                parse_coords(triple.coords_entity2)
            ])

    imgname = img.dataset + "/" + img.category + "/" + img.imgpath
    img_id = img.id

    url = url_for('main.val_task_post', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID)
    opt_out_url = url_for('auth.opt_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID)
    generate_new_annotation_url = url_for('main.generate_new_annotation_html_body', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID)

    f = open('sgg_dicts.json')
    dict = json.load(f)

    entities_list_suggestion = []
    predicates_list_suggestion = []

    i = 0
    while i < 26:
        entities_list_suggestion.append([])
        predicates_list_suggestion.append([])

        if len(predicates_list_suggestion) > 25 or len(entities_list_suggestion) > 25:
            break

    for e in list(dict['object_count'].keys()):
        if len(e) > 0:
            first_letter = e[0].lower()

            if first_letter.isalpha():
                entities_list_suggestion[ord(first_letter) - 97].append(e)

    for p in list(dict['predicate_count'].keys()):
        if len(p) > 0:
            first_letter = p[0].lower()

            if first_letter.isalpha():
                predicates_list_suggestion[ord(first_letter) - 97].append(p)

    return render_template('val_task.html', question = question, answer_option = answer_option, triples = triples_array, imgname = imgname, progress = progress, url = url, opt_out_url = opt_out_url, generate_new_annotation_url = generate_new_annotation_url, img_id = img_id, predicates_list_suggestion = predicates_list_suggestion, entities_list_suggestion = entities_list_suggestion)

@login_required
@main_blueprint.route('/val_task_post', methods = ('POST', ))
def val_task_post():

    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    # Check if the url contains the required parameters
    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == "" or User.query.filter_by(prolific_pid = PROLIFIC_PID).first() == None):
        return jsonify(url_for('main.error'))
    
    # Check if the user has opted out
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return jsonify(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    triples = request.json['triples']
    img_id = request.json['img_id']

    for t in triples:
        if t[0][0] == "" or t[0][1] == "" or t[0][2] == "" or (len(t) > 3 and t[0][3] == "") or t[0][0] == None or t[0][1] == None or t[0][2] == None or (len(t) > 3 and t[0][3] == None) or t[1][2] == 0 or t[1][3] == 0 or t[2][2] == 0 or t[2][3] == 0 or (len(t) > 3 and t[3][0] == True and (t[3][1] == None or t[3][2] == None)) or (len(t) > 3 and t[3][0] == False and (t[3][1] != None or t[3][2] != None)) or (len(t) > 3 and t[3][0] == None):
            continue

        new_annotation = None
        new_triple = None
        if len(t) > 3:
            new_annotation = Annotation(
                correctly_identified = t[3][1] if t[3][1] else False,
                new_label_entity1 = t[0][0],
                new_label_entity2 = t[0][2],
                new_coords_entity1 = array_to_string(t[1]),
                new_coords_entity2 = array_to_string(t[2]),
                entities_in_relationship = True if t[3][2] == "true" or t[3][2] == "incorrect_relationship" else False,
                correct_relationship = True if t[3][2] == "true" else False,
                new_relationship = t[0][1],
                answer_question = t[3][0],
                id_triple = t[0][3],
                id_user = User.query.filter_by(prolific_pid = PROLIFIC_PID).first().id
            )
        elif len(t) == 3:
            # open sgg_dicts file
            with open('sgg_dicts.json', 'r') as f:
                sgg_dicts = json.load(f)
            entity1_idx = sgg_dicts['label_to_idx'][t[0][0]] if t[0][0] in sgg_dicts['label_to_idx'] else 'new'
            rel_idx = sgg_dicts['predicate_to_idx'][t[0][1]] if t[0][1] in sgg_dicts['predicate_to_idx'] else 'new'
            entity2_idx = sgg_dicts['label_to_idx'][t[0][2]] if t[0][2] in sgg_dicts['label_to_idx'] else 'new'
            id_json = '{}-{}-{}'.format(entity1_idx, rel_idx, entity2_idx)
            new_triple = Triple(
                id_json = id_json,
                entity1 = t[0][0],
                coords_entity1 = array_to_string(t[1]),
                relationship = t[0][1],
                entity2 = t[0][2],
                coords_entity2 = array_to_string(t[2]),
                new_user_annotation = True,
                id_new_user_annotation = PROLIFIC_PID if User.query.filter_by(prolific_pid = PROLIFIC_PID).first().prolific_pid else 'default',
                id_image = img_id
            )

        if new_annotation != None:
            db.session.add(new_annotation)
            db.session.commit()

        if new_triple != None:
            db.session.add(new_triple)
            db.session.commit()

    return jsonify(url_for('main.ann_task', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

@login_required
@main_blueprint.route('/ann_task')
def ann_task():
    
    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    # Check if the url contains the required parameters
    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == "" or User.query.filter_by(prolific_pid = PROLIFIC_PID).first() == None):
        return redirect(url_for('main.error'))
    
    # Check if the user has opted out
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return redirect(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))
    
    num_performed_val_tasks = (
        Annotation.query
        .filter(Annotation.id_user == User.query.filter_by(prolific_pid=PROLIFIC_PID).first().id)
        .join(Triple, Annotation.id_triple == Triple.id)
        .with_entities(Triple.id_image)
        .distinct()
        .count()
    )
    num_performed_ann_tasks = (
        Triple.query
        .with_entities(Triple.id_image)
        .filter_by(id_new_user_annotation = PROLIFIC_PID)
        .distinct()
        .count()
    )
    
    # Check if the user has performed the annotation of the validated image
    if num_performed_val_tasks <= num_performed_ann_tasks:
        return redirect(url_for('main.val_task', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))        
    
    progress = float(num_performed_ann_tasks / max_assigned_tasks) * 100
    
    img = get_low_annotation_image(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().id)

    if img == None:
        return redirect(url_for('main.final', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    question = img.question
    answer_option = None
    
    # Load the MC answer from the sampled_questions.json file
    dataset_string = 'prepareDB/datasets/' + img.dataset + '/sampled_questions.json'
    sampled_questions = []
    try:
        with open(dataset_string) as f:
            sampled_questions = json.load(f)
    except:
        # logger.error('Could not load the sampled questions file for dataset: ' + img.dataset)
        pass
    
    question_object = None
    for q in sampled_questions:
        if int(q['img'].split('.')[0]) == int(img.imgpath.split('.')[0]):
            question_object = q
            break
    
    if question_object is not None:
        if question_object['answer'] is not None:
            answer = question_object['answer']
            if question_object['options'] is not None:
                # This covers MMBench and SEED (MC questions)
                options = question_object['options']
                answer_option = options[answer] # if options is not None else None
            else:
                # This covers VQA v2 and LLaVa-Bench ('options' is None)
                answer_option = answer
        else:
            # Default (this should not happen based on the data)
            None

    imgname = img.dataset + "/" + img.category + "/" + img.imgpath
    img_id = img.id

    url = url_for('main.ann_task_post', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID)
    opt_out_url = url_for('auth.opt_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID)
    generate_new_annotation_url = url_for('main.generate_new_annotation_html_body', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID)

    f = open('sgg_dicts.json')
    dict = json.load(f)

    entities_list_suggestion = []
    predicates_list_suggestion = []

    i = 0
    while i < 26:
        entities_list_suggestion.append([])
        predicates_list_suggestion.append([])

        if len(predicates_list_suggestion) > 25 or len(entities_list_suggestion) > 25:
            break

    for e in list(dict['object_count'].keys()):
        if len(e) > 0:
            first_letter = e[0].lower()

            if first_letter.isalpha():
                entities_list_suggestion[ord(first_letter) - 97].append(e)

    for p in list(dict['predicate_count'].keys()):
        if len(p) > 0:
            first_letter = p[0].lower()

            if first_letter.isalpha():
                predicates_list_suggestion[ord(first_letter) - 97].append(p)

    return render_template('ann_task.html', question = question, answer_option = answer_option, imgname = imgname, progress = progress, url = url, opt_out_url = opt_out_url, generate_new_annotation_url = generate_new_annotation_url, img_id = img_id, predicates_list_suggestion = predicates_list_suggestion, entities_list_suggestion = entities_list_suggestion, max_assigned_ann_triples_per_task = max_assigned_ann_triples_per_task)
    
@login_required
@main_blueprint.route('/ann_task_post', methods = ('POST', ))
def ann_task_post():
    
    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == "" or User.query.filter_by(prolific_pid = PROLIFIC_PID).first() == None):
        return jsonify(url_for('main.error'))
    
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return jsonify(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    triples = request.json['triples']
    img_id = request.json['img_id']

    to_unlock_image = Image.query.filter_by(id = img_id).first()
    to_unlock_image.locked = None
    db.session.commit()

    for t in triples:
        if t[0][0] == "" or t[0][1] == "" or t[0][2] == "" or (len(t) > 3 and t[0][3] == "") or t[0][0] == None or t[0][1] == None or t[0][2] == None or (len(t) > 3 and t[0][3] == None) or t[1][2] == 0 or t[1][3] == 0 or t[2][2] == 0 or t[2][3] == 0 or (len(t) > 3 and t[3][0] == True and (t[3][1] == None or t[3][2] == None)) or (len(t) > 3 and t[3][0] == False and (t[3][1] != None or t[3][2] != None)) or (len(t) > 3 and t[3][0] == None):
            continue
        
        # open sgg_dicts file
        with open('sgg_dicts.json', 'r') as f:
            sgg_dicts = json.load(f)
        entity1_idx = sgg_dicts['label_to_idx'][t[0][0]] if t[0][0] in sgg_dicts['label_to_idx'] else 'new'
        rel_idx = sgg_dicts['predicate_to_idx'][t[0][1]] if t[0][1] in sgg_dicts['predicate_to_idx'] else 'new'
        entity2_idx = sgg_dicts['label_to_idx'][t[0][2]] if t[0][2] in sgg_dicts['label_to_idx'] else 'new'
        id_json = '{}-{}-{}'.format(entity1_idx, rel_idx, entity2_idx)
        new_triple = Triple(
            id_json = id_json,
            entity1 = t[0][0],
            coords_entity1 = array_to_string(t[1]),
            relationship = t[0][1],
            entity2 = t[0][2],
            coords_entity2 = array_to_string(t[2]),
            new_user_annotation = True,
            id_new_user_annotation = PROLIFIC_PID if User.query.filter_by(prolific_pid = PROLIFIC_PID).first().prolific_pid else 'default',
            id_image = img_id
        )

        db.session.add(new_triple)
        db.session.commit()

    return jsonify(url_for('main.val_task', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

@login_required
@main_blueprint.route('/final')
def final():
    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == "" or User.query.filter_by(prolific_pid = PROLIFIC_PID).first() == None):
        return jsonify(url_for('main.error'))
    
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return jsonify(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    # performed_task = len(numpy.unique(numpy.array([Triple.query.filter_by(id = ann.id_triple).first().id_image for ann in User.query.filter_by(prolific_pid = PROLIFIC_PID).first().annotations])))
    num_performed_val_tasks = (
        Annotation.query
        .filter(Annotation.id_user == User.query.filter_by(prolific_pid=PROLIFIC_PID).first().id)
        .join(Triple, Annotation.id_triple == Triple.id)
        .with_entities(Triple.id_image)
        .distinct()
        .count()
    )
    
    if(num_performed_val_tasks < max_assigned_tasks):
        # return render_template('error.html')
        return redirect(url_for('main.error', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    return render_template('final.html')

@main_blueprint.route('/opted_out')
def opted_out():

    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    if request.args.get('REJECT'):
        return render_template('optout.html')

    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == "" or User.query.filter_by(prolific_pid = PROLIFIC_PID).first() == None):
        return jsonify(url_for('main.error'))

    return render_template('optout.html')

@login_required
@main_blueprint.route('/generate_new_annotation_html_body', methods = ('POST', ))
def generate_new_annotation_html_body():

    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')

    if(PROLIFIC_PID == None or PROLIFIC_PID == "" or STUDY_ID == None or STUDY_ID == "" or User.query.filter_by(prolific_pid = PROLIFIC_PID).first() == None):
        return jsonify(url_for('main.error'))
    if(User.query.filter_by(prolific_pid = PROLIFIC_PID).first().optout):
        return jsonify(url_for('main.opted_out', PROLIFIC_PID = PROLIFIC_PID, STUDY_ID = STUDY_ID))

    count = request.json['count']

    tag = "<div class='accordion-item'>"
    tag += "<h2 class='accordion-header' id='new_annotation_heading_" + str(count) + "'>"
    tag += "<button class='accordion-button collapsed' type='button' data-bs-toggle='collapse' data-bs-target='#new_annotation_" + str(count) + "' aria-expanded='false' aria-controls='new_annotation_" + str(count) + "' id='collapse_button_" + str(count) + "'>"
    tag += "<b style='color: red; padding-right: 5px;' id='collapse_status_" + str(count) + "'>[TO DO]</b>" + "New annotation #" + str(count) + ": <code style='padding-left: 5px;' id='collapse_code_" + str(count) + "'><></code>"
    tag += "</button>"
    tag += "</h2>"
    tag += "<div id='new_annotation_" + str(count) + "' class='accordion-collapse collapse' aria-labelledby='new_annotation_heading_" + str(count) + "' data-bs-parent='#triples_accordion'>"
    tag += "<div class='accordion-body'>"
    tag += "<div>"
    tag += "<span><b><u>New Entity #1</u></b></span>"
    tag += "<div class='d-flex flex-column'>"
    tag += "<div class='d-flex my-1'>"
    tag += "<span class='col-3'><b>Label: </b></span><input type='search' list='entities' placeholader='Label Entity #1' class='col-9' id='label_entity_1_" + str(count) + "' autocomplete='off'>"
    tag += "</div>"
    tag += "<div class='d-flex my-1 justify-content-between'>"
    tag += "<p><a href='' id='change_bounding_box_entity_1_" + str(count) + "'>Click here to create the Bounding Box</a><br>"
    tag += "Then, click, and drag over the image to draw. </p>"
    tag += "<a href='' id='reset_bounding_box_entity_1_" + str(count) + "' class='d-none'>Reset Bounding Box</a>"
    tag += "</div>"
    tag += "</div>"
    tag += "<hr>"
    # tag += "<span><b><u>Predicate</u></b></span>"
    tag += "<div class='d-flex'>"
    tag += "<span class='col-3'><b><u>New Predicate:</u></b></span><input id='predicate_" + str(count) + "' type='search' list='predicates' placeholader='Predicate' class='col-9' id='predicate_" + str(count) + "' autocomplete='off'>"
    tag += "</div>"
    tag += "<hr>"
    tag += "<span><b><u>New Entity #2</u></b></span>"
    tag += "<div class='d-flex flex-column'>"
    tag += "<div class='d-flex my-1'>"
    tag += "<span class='col-3'><b>Label: </b></span><input type='search' list='entities' placeholader='Label Entity #2' class='col-9' id='label_entity_2_" + str(count) + "' autocomplete='off'>"
    tag += "</div>"
    tag += "<div class='d-flex my-1 justify-content-between'>"
    tag += "<p><a href='' id='change_bounding_box_entity_2_" + str(count) + "'>Click here to crete the Bounding Box.</a><br>"
    tag += "Then, click and drag over the image to draw.</p>"
    tag += "<a href='' id='reset_bounding_box_entity_2_" + str(count) + "' class='d-none'>Reset Bounding Box</a>"
    tag += "</div>"
    tag += "</div>"
    tag += "<hr>"
    tag += "<a href='' id='remove_annotation_" + str(count) + "'>Remove Annotation</a>"
    tag += "</div>"
    tag += "</div>"
    tag += "</div>"
    tag += "</div>"
    tag += "</div>"

    return jsonify(tag)

@main_blueprint.route('/error')
def error():

    return render_template('error.html')

def get_low_annotation_image(user_id):

    leaderboard_img_id = []
    leaderboard_unique_users = []
    leaderboard_already_annotated = []
    leaderboard_img_locked = []

    # If an image is already assigned to a user, return it
    personal_locked_image = Image.query.filter_by(locked = user_id).first()
    if personal_locked_image != None:
        # logger.info('Retrieve image already assigned {}'.format(personal_locked_image.id))
        return personal_locked_image

    # Otherwise
    images = Image.query.filter_by().all()
    random.shuffle(images)

    for i in images:
        already_annotated = False
        all_users_annotations = []

        for t in i.triples:
            all_users_annotations[len(all_users_annotations):] = [ann.id_user for ann in t.annotations]

        unique_users = numpy.unique(numpy.array(all_users_annotations))
        if user_id in unique_users:
            already_annotated = True

        leaderboard_img_id.append(i.id)
        leaderboard_unique_users.append(len(unique_users))
        leaderboard_already_annotated.append(already_annotated)
        leaderboard_img_locked.append(i.locked)

    img_id = None
    found = False
    while len(leaderboard_unique_users) > 0:
        min_val = min(leaderboard_unique_users)
        min_pos = leaderboard_unique_users.index(min_val)

        if(not leaderboard_already_annotated[min_pos] and leaderboard_img_locked[min_pos] == None):
            img_id = leaderboard_img_id[min_pos]
            found = True

            to_lock_image = Image.query.filter_by(id = img_id).first()
            to_lock_image.locked = user_id
            db.session.commit()
        else:
            leaderboard_img_id.pop(min_pos)
            leaderboard_unique_users.pop(min_pos)
            leaderboard_already_annotated.pop(min_pos)
            leaderboard_img_locked.pop(min_pos)

        if found:
            break

    if img_id != None:
        # logger.info('Assigned image {}'.format(img_id))
        return Image.query.filter_by(id = img_id).first()

    return None

def get_low_annotation_triples(img_id, user_id):

    triples_array_ann = []

    triples_db = Triple.query.filter_by(id_image = img_id).all()
    random.shuffle(triples_db)
    annotation_per_triple = [len(t.annotations) for t in triples_db if user_id not in [ann.id_user for ann in t.annotations]]

    # while len(triples_array_ann) < max_assigned_triples_per_task and len(annotation_per_triple) > 0:
    while len(annotation_per_triple) > 0:
        min_val = min(annotation_per_triple)
        min_pos = annotation_per_triple.index(min_val)

        triples_array_ann.append(triples_db[min_pos])

        triples_db.pop(min_pos)
        annotation_per_triple.pop(min_pos)

    return triples_array_ann

def parse_coords(coord_string):
    coords = coord_string.replace("[", "").replace("]", "").split(", ")
    return [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]

def array_to_string(coord_array):
    return "[" + str(round(coord_array[0])) + ", " + str(round(coord_array[1])) + ", " + str(round(coord_array[2])) + ", " + str(round(coord_array[3])) + "]"
