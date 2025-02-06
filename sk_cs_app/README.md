# Model Behaviour Annotation Tool

## Setting up database with custom images

1. Navigate to ```CRWD/prepareDB/datasets```
2. Copy the dataset data, making sure to have the following files
   1. ```questions.jsonl```
   2. ```triples.json```
3. Run ```db_converter.py``` to obtain the ```init.db``` file.
4. Copy the ```init.db``` to ```CRWD/mysql``` folder so that it is visible to ```docker compose``` later on.

### Expected data format

#### questions.jsonl
Each line follows this format
~~~~json
{
    "question_id": "...",   // ID of the question
    "image": "...",         // filename
    "text": "...",          // the question
    "category": "..."       // the class (if any)
}
~~~~

#### triples.json

~~~~json
{
    "imgs": [
        {
            "img_id": "...",                            // ID of the image
            "img_path": "...",                          // Filepath
            "relations": [{
                    "from_concept": {                       // Concept #1
                        "bb_label_idx": "...",
                        "bb_label_text": "...",
                        "bb_label_full": "..."
                    },
                    "to_concept": {                         // Concept #2
                        // Same fields as from_concept
                    },
                    "rel_label": {                          // Relation
                        "rel_label_idx": "...",
                        "rel_label_text": "..."
                    }
                }, ..., {...}
            ],
            "bboxes": [{
                    "top_left_x": "...",
                    "top_left_y": "...",
                    "bottom_right_x": "...",
                    "bottom_right_y": "...",
                    "length": "...",
                    "width": "...",
                    "bb_label": {                       // Concept label
                        "bb_label_idx": "...",
                        "bb_label_text": "...",         
                        "bb_label_full": "..."
                    }
                }, ..., {...}
            ]
        }
    ]
}
~~~~

## Running the annotation app with Docker
Once all the above is in place, run the following command to build and run the containers.
There are different docker-compose files, one for each dataset to be annotated.

~~~~bash
docker compose -p "crowd_llava" -f docker-compose-llava.yml up -d

docker compose -p "crowd_mmbench" -f docker-compose-mmbench.yml up -d

docker compose -p "crowd_seed" -f docker-compose-seed.yml up -d

docker compose -p "crowd_vqa" -f docker-compose-vqav2.yml up -d
~~~~

To access the webapp, go to ```http://localhost:8000/crwd/?PROLIFIC_PID=aaa&STUDY_ID=xxx```.

Change the port depending on the dataset
- LLaVa-Bench: 8000
- MMBench: 8001
- SEED Bench: 8002
- VQA v2: 8003

> Note that PROLIFIC_ID and STUDY_ID are given from Prolific. For testing locally, you can input random values for these.