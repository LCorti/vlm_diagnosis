import mysql.connector
import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.sk_config_loader import SKConfig
from utils.data_io import make_dir, save_jsonl

# Assumption: data has been exported from docker container into local instance.
# If not, change the the connection configuration below


def get_query_annotations():
    return "SELECT \
            U.id AS id_user, optout, A.id as id_annotation, \
            correctly_identified, new_label_entity1 AS new_label_from_concept, \
            new_label_entity2 AS new_label_to_concept, \
            new_coords_entity1 AS new_coords_from_concept, \
            new_coords_entity2 AS new_coords_to_concept, \
            entities_in_relationship, correct_relationship, new_relationship, \
            answer_question, T.id AS id_triple, id_json, \
            entity1 AS from_concept, coords_entity1 AS coords_from_concept, \
            relationship, entity2 AS to_concept, \
            coords_entity2 AS coords_to_concept, new_user_annotation, \
            I.id as id_image, imgpath, question, dataset, category AS class \
            FROM user U \
            JOIN annotation A ON U.id = A.id_user \
            JOIN triple T ON T.id = A.id_triple \
            JOIN image I ON I.id = T.id_image \
            WHERE U.optout=0"


def get_query_new_triples():
    return "SELECT T.id AS id, U.id AS id_user, optout, \
            id_json, entity1, coords_entity1, relationship, \
            entity2, coords_entity2, new_user_annotation, T.id_image, \
            imgpath, question, dataset, category AS class \
            FROM user U \
            RIGHT JOIN triple T ON T.id_new_user_annotation = U.prolific_pid \
            JOIN image I ON I.id = T.id_image \
            WHERE new_user_annotation=1 AND U.optout=0;"


HOST = "localhost"
USER = "root"
PASSWORD = ""
DATABASE_PREFIX = "lmm_diag_"  # completed dynamically

if __name__ == "__main__":
    sk_config = SKConfig()
    sk_list = sk_config.get_sk_list()
    sk_list.remove("vqav2_holdout")

    for ds in sk_list:
        print("=" * 25)
        print("Connecting to {}...".format(ds))

        curr_out_paths = sk_config.get_sk_paths(ds)
        database = "{}{}".format(DATABASE_PREFIX, ds)
        try:
            mydb = mysql.connector.connect(
                host=HOST, user=USER, password=PASSWORD, database=database
            )
            cursor = mydb.cursor(dictionary=True)

            # Extract all data from validation step matched
            # with user, image, and triples
            query_anns = get_query_annotations()
            rows_val = []
            try:
                cursor.execute(query_anns)
                rows_val = cursor.fetchall()
            except Exception:
                print("Failed to execute query for {}".format(ds))
            print("Found {} rows.".format(len(rows_val)))

            # Extract all new triples (these are created and do not have
            # annotations associated to them, they are not found in 1st)
            query_triples = get_query_new_triples()
            rows_ann = []
            try:
                cursor.execute(query_triples)
                rows_ann = cursor.fetchall()
            except Exception:
                print("Failed to execute query for {}".format(ds))
            print("Found {} rows.".format(len(rows_ann)))

            # Save crowdsourced data in different files
            for ds_class in curr_out_paths:
                out_dir = f"../{curr_out_paths[ds_class]['dir']}"
                make_dir(out_dir)

                # Validation data
                out_file_name = curr_out_paths[ds_class]["val_step"]
                out_file_path = f"{out_dir}/{out_file_name}"
                curr_val_data = [r for r in rows_val if r["class"] == ds_class]
                save_jsonl(curr_val_data, out_file_path)

                # Annotation data
                out_file_name = curr_out_paths[ds_class]["ann_step"]
                out_file_path = f"{out_dir}/{out_file_name}"
                curr_ann_data = [r for r in rows_ann if r["class"] == ds_class]
                save_jsonl(curr_ann_data, out_file_path)

            # Close cursor and free up pooled connection
            cursor.close()
            mydb.close()

        except Exception as e:
            print(e)

    print("=" * 25)
