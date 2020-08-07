import os
import shutil
import json
import h5py
from pathlib import Path
from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging

DATA_DIR = 'data/example_2'
GRAPH_PATH = DATA_DIR + '/edges.tsv'
MODEL_DIR = 'model_2'


if __name__ == '__main__':
    try:
        shutil.rmtree('data')
    except:
        pass
    try:
        shutil.rmtree('model_2')
    except:
        pass

    print('inside  MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM')
    # ==================================================================
    # 0. PREPARE THE GRAPH
    # the result of this step is a single file 'data/example_2/graph.tsv'
    # ==================================================================

    # This the graph we will be embedding.
    # It has 3 types of nodes - "user", "item", "merchant" - and 3 types of edges - "bought", "sold", "follows"
    # users can buy items, merchants sell items, users follow other users
    edges = [
        ['alice', 'bought', 'fridge'],
        ['alice', 'bought', 'bike'],
        ['bob', 'bought', 'laptop'],
        ['carol', 'bought', 'fridge'],
        ['carol', 'sold', 'laptop'],
        ['carol', 'sold', 'bike'],
        ['dave', 'sold', 'fridge'],
        ['alice', 'follows', 'bob'],
        ['bob', 'follows', 'carol'],
        ['bob', 'hates', 'dave'],
        ['dave', 'hates', 'carol']
    ]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(GRAPH_PATH, 'w') as f:
        for edge in edges:
            f.write('\t'.join(edge) + '\n')

    # ==================================================
    # 1. DEFINE CONFIG
    # this dictionary will be used in steps 2. and 3.
    # ==================================================

    raw_config = dict(
        # I/O data
        entity_path=DATA_DIR,
        edge_paths=[
            DATA_DIR + '/edges_partitioned',
        ],
        checkpoint_path=MODEL_DIR,
        # Graph structure
        entities={
            "user": {"num_partitions": 1},
            "item": {"num_partitions": 1},
            "merchant": {"num_partitions": 1}
        },
        relations=[
            {
                "name": "bought",
                "lhs": "user",
                "rhs": "item",
                "operator": "complex_diagonal",
            },
            {
                "name": "sold",
                "lhs": "merchant",
                "rhs": "item",
                "operator": "complex_diagonal",
            },
            {
                "name": "follows",
                "lhs": "user",
                "rhs": "user",
                "operator": "complex_diagonal",
            },
            {
                "name": "hates",
                "lhs": "user",
                "rhs": "user",
                "operator": "complex_diagonal",
            }
        ],
        dynamic_relations=False,
        dimension=4,  # silly graph, silly dimensionality
        global_emb=False,
        comparator="dot",
        num_epochs=7,
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.1,
        regularization_coef=1e-3,
        eval_fraction=0.,
    )

    # =================================================
    # 2. TRANSFORM GRAPH TO A BIGGRAPH-FRIENDLY FORMAT
    # This step generates the following metadata files:
    #
    # data/example_2/entity_count_item_0.txt
    # data/example_2/entity_count_merchant_0.txt
    # data/example_2/entity_count_user_0.txt
    # data/example_2/entity_names_item_0.json
    # data/example_2/entity_names_merchant_0.json
    # data/example_2/entity_names_user_0.json
    #
    # and this file with data:
    # data/example_2/edges_partitioned/edges_0_0.h5
    # =================================================
    setup_logging()
    config = parse_config(raw_config)
    subprocess_init = SubprocessInitializer()
    input_edge_paths = [Path(GRAPH_PATH)]

    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rel_col=1, rhs_col=2),
        dynamic_relations=config.dynamic_relations,
    )

    # ===============================================
    # 3. TRAIN THE EMBEDDINGS
    # files generated in this step:
    #
    # checkpoint_version.txt
    # config.json
    # embeddings_item_0.v7.h5
    # embeddings_merchant_0.v7.h5
    # embeddings_user_0.v7.h5
    # model.v7.h5
    # training_stats.json
    # ===============================================

    train(config, subprocess_init=subprocess_init)

    # =======================================================================
    # 4. LOAD THE EMBEDDINGS
    # The final output of the process consists of 3 dictionaries -
    # - one for users, items, merchants - mapping entity to its embedding
    # =======================================================================
    users_path = DATA_DIR + '/entity_names_user_0.json'
    items_path = DATA_DIR + '/entity_names_item_0.json'
    merchants_path = DATA_DIR + '/entity_names_merchant_0.json'

    user_emb_path = MODEL_DIR + "/embeddings_user_0.v{NUMBER_OF_EPOCHS}.h5" \
        .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])
    item_emb_path = MODEL_DIR + "/embeddings_item_0.v{NUMBER_OF_EPOCHS}.h5" \
        .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])
    merchant_emb_path = MODEL_DIR + "/embeddings_merchant_0.v{NUMBER_OF_EPOCHS}.h5" \
        .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

    with open(users_path, 'r') as f:
        users = json.load(f)

    with h5py.File(user_emb_path, 'r') as g:
        user_embeddings = g['embeddings'][:]

    user2embedding = dict(zip(users, user_embeddings))
    print('user embeddings')
    print(user2embedding)
    print()

    with open(items_path, 'r') as f:
        items = json.load(f)

    with h5py.File(item_emb_path, 'r') as g:
        item_embeddings = g['embeddings'][:]

    item2embedding = dict(zip(items, item_embeddings))
    print('item embeddings')
    print(item2embedding)
    print()

    with open(merchants_path, 'r') as f:
        merchants = json.load(f)

    with h5py.File(merchant_emb_path, 'r') as g:
        merchants_embeddings = g['embeddings'][:]

    merchant2embedding = dict(zip(merchants, merchants_embeddings))
    print('merchant embeddings')
    print(merchant2embedding)
