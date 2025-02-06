import json

def get_content_struct():
    content = []

    # Define Basic Structure
    content.append("CREATE TABLE user (\n")
    content.append("    id              SERIAL PRIMARY KEY,\n")
    content.append("    prolific_pid    VARCHAR(100) UNIQUE,\n")
    content.append("    optout          BOOLEAN NOT NULL DEFAULT 0\n")
    content.append(");\n")
    content.append("\n")

    content.append("CREATE TABLE image (\n")
    content.append("    id        SERIAL PRIMARY KEY,\n")
    content.append("    imgpath   VARCHAR(100) NOT NULL,\n")
    content.append("    question  VARCHAR(500) NOT NULL,\n")
    content.append("    dataset   VARCHAR(100) NOT NULL,\n")
    content.append("    category  VARCHAR(100) NOT NULL,\n")
    content.append("    locked    BIGINT UNSIGNED\n")
    content.append(");\n")
    content.append("\n")

    content.append("CREATE TABLE triple (\n")
    content.append("    id                     SERIAL PRIMARY KEY,\n")
    content.append("    id_json                VARCHAR(100) NOT NULL,\n")
    content.append("    entity1                VARCHAR(100) NOT NULL,\n")
    content.append("    coords_entity1         VARCHAR(100) NOT NULL,\n")
    content.append("    relationship           VARCHAR(100) NOT NULL,\n")
    content.append("    entity2                VARCHAR(100) NOT NULL,\n")
    content.append("    coords_entity2         VARCHAR(100) NOT NULL,\n")
    content.append("    new_user_annotation    BOOLEAN NOT NULL DEFAULT 0,\n")
    content.append("    id_new_user_annotation VARCHAR(100),\n")
    content.append("    id_image               BIGINT UNSIGNED NOT NULL,\n")
    content.append("    FOREIGN KEY (id_image) REFERENCES image(id)\n")
    content.append(");\n")
    content.append("\n")

    content.append("CREATE TABLE annotation (\n")
    content.append("    id                        SERIAL PRIMARY KEY,\n")
    content.append("    correctly_identified      BOOLEAN,\n")
    content.append("    new_label_entity1         VARCHAR(100),\n")
    content.append("    new_label_entity2         VARCHAR(100),\n")
    content.append("    new_coords_entity1        VARCHAR(100),\n")
    content.append("    new_coords_entity2        VARCHAR(100),\n")
    content.append("    entities_in_relationship  BOOLEAN,\n")
    content.append("    correct_relationship      BOOLEAN,\n")
    content.append("    new_relationship          VARCHAR(100),\n")
    content.append("    answer_question           BOOLEAN,\n")
    content.append("    id_triple                 BIGINT UNSIGNED NOT NULL,\n")
    content.append("    FOREIGN KEY (id_triple) REFERENCES triple(id),\n")
    content.append("    id_user                   BIGINT UNSIGNED NOT NULL,\n")
    content.append("    FOREIGN KEY (id_user) REFERENCES user(id)\n")
    content.append(");\n")
    content.append("\n")

    return content

def main():

    datasets = {
        'llava-bench': ['main'],
        'mmbench': ['image_scene', 'image_topic'],
        'seed': ['scene_understanding', 'visual_reasoning'],
        'vqav2': ['how_many_people_are', 'what_is_the_person']
    }

    for ds in datasets.keys():
        #Get basic structure
        content = get_content_struct()

        folder_name = 'datasets/{}/'.format(ds)
        print("Working on {}...".format(ds))
        questions = []

        # Prepare Images + Questions
        ds_questions_path = '{}/sampled_questions.json'.format(folder_name)
        with open(ds_questions_path, 'r') as f:
            ds_questions = json.load(f)

        for ds_q in ds_questions:
            questions.append([str(int(ds_q["question_id"]) + 1), str(ds_q["img"].split('.')[0])])
            content.append('INSERT INTO image VALUES (' + str(int(ds_q["question_id"]) + 1) + ', "' + ds_q["img"] + '", "' + ds_q["question"] + '", "' + ds + '", "' + ds_q["class"] + '", NULL);\n')
        content.append('\n')

        # Prepare Triples
        triples_path = '{}/sg_{}_crowd.json'.format(folder_name, ds)
        with open(triples_path, 'r') as f:
            triples = json.load(f)

        for c in triples.keys():
            for img in triples[c]:
                img_idx = img['img_id']

                # List of coordinates
                coords = []
                for r in img['rel_clusters_unique']:
                    rel_id = r['rel_id']
                    # from_concept
                    coord_from = [r['from_concept']['top_left_x'],
                               r['from_concept']['top_left_y'],
                               r['from_concept']['width'],
                               r['from_concept']['height']]

                    # to_concept
                    coord_to = [r['to_concept']['top_left_x'],
                               r['to_concept']['top_left_y'],
                               r['to_concept']['width'],
                               r['to_concept']['height']]

                    coord_from_str = "[" + str(coord_from[0]) + ", " + str(coord_from[1]) + ", " + str(coord_from[2]) + ", " + str(coord_from[3]) + "]"
                    coord_to_str = "[" + str(coord_to[0]) + ", " + str(coord_to[1]) + ", " + str(coord_to[2]) + ", " + str(coord_to[3]) + "]"

                    # List of relations
                    relations = []
                    entity_from = str(r['from_concept']['bb_label']['bb_label_text'])
                    rel = str(r["rel_label"]["rel_label_text"])
                    entity_to = str(r['to_concept']['bb_label']['bb_label_text'])

                    images = [q[0] for q in questions if q[1] == str(img_idx)]

                    for i in images:
                        content.append('INSERT INTO triple VALUES (DEFAULT, "' + rel_id + '", "' + entity_from + '", "' + coord_from_str + '", "' + rel + '", "' + entity_to + '", "' + coord_to_str + '", DEFAULT, NULL, ' + i + ');\n')

        # Write SQL file
        outfile = '../mysql/initdb_{}.sql'.format(ds)
        with open(outfile, 'w') as f:
            for c in content:
                f.write(c)

if __name__ == "__main__":
    main()
