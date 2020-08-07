import os
import shutil
import json
import h5py
from pathlib import Path
from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging

DATA_DIR = 'data/example_1'
GRAPH_PATH = DATA_DIR + '/edges.tsv'
MODEL_DIR = 'model_1'


if __name__ == '__main__':
    try:
        shutil.rmtree('data')
    except:
        pass
    try:
        shutil.rmtree(MODEL_DIR)
    except:
        pass

    # ==================================================================
    # 0. PREPARE THE GRAPH
    # the result of this step is a single file 'data/example_1/graph.tsv'
    # ==================================================================

    # A simple graph with 4 vertices and 5 edges
    edges = [
        ["A", "B"],
        ["B", "C"],
        ["C", "D"],
        ["D", "B"],
        ["B", "D"]
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
            "WHATEVER": {"num_partitions": 1}
        },
        relations=[
            {
                "name": "doesnt_matter",
                "lhs": "WHATEVER",
                "rhs": "WHATEVER",
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
    # entity_names_WHATEVER_0.json
    # entity_count_WHATEVER_0.txt
    #
    # and this file with data:
    # data/example_1/edges_partitioned/edges_0_0.h5
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
        TSVEdgelistReader(lhs_col=0, rel_col=None, rhs_col=1),
        dynamic_relations=config.dynamic_relations,
    )

    # ===============================================
    # 3. TRAIN THE EMBEDDINGS
    # files generated in this step:
    #
    # checkpoint_version.txt
    # config.json
    # embeddings_WHATEVER_0.v7.h5
    # model.v7.h5
    # training_stats.json
    # ===============================================
    train(config, subprocess_init=subprocess_init)

    # =======================================================================
    # 4. LOAD THE EMBEDDINGS
    # The final output of the process is a dict mapping node names to embeddings
    # =======================================================================
    nodes_path = DATA_DIR + '/entity_names_WHATEVER_0.json'
    embeddings_path = MODEL_DIR + "/embeddings_WHATEVER_0.v{NUMBER_OF_EPOCHS}.h5" \
        .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

    with open(nodes_path, 'r') as f:
        node_names = json.load(f)

    with h5py.File(embeddings_path, 'r') as g:
        embeddings = g['embeddings'][:]

    node2embedding = dict(zip(node_names, embeddings))
    print('embeddings')
    print(node2embedding)
