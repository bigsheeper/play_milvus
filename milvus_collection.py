# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import time
import os

import signal
import sys
import random

from threading import Timer
import numpy as np

from pymilvus import (
    list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, Partition,
    connections,
)
from common import *

def connect_server(host):
    # configure milvus hostname and port
    print(f"\nCreate connection...")
    connections.connect(host=host, port=19530)

def prepare_collection(dataset):
    collection_name = COLLECTION_NAME
    if dataset == DATASET_DEEP:
        dim = 96
    elif dataset == DATASET_SIFT:
        dim = 128
    else:
        raise_exception("wrong dataset")

    # List all collection names
    print(f"\nList collections...")
    collection_list = list_collections()
    print(list_collections())

    if (collection_list.count(collection_name)):
        print(collection_name, " exist, and drop it")
        collection = Collection(collection_name)
        collection.drop()
        print("drop collection ", collection_name)

    field1 = FieldSchema(name="id", dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name="rand", dtype=DataType.DOUBLE, description="double", is_primary=False)
    field3 = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, description="float vector", dim=dim, is_primary=False)
    schema = CollectionSchema(fields=[field1, field2, field3], description="")
    collection = Collection(name=collection_name, data=None, schema=schema, shards_num=1)
    return collection

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Insert Data to Milvus user-defined ranges of .npy files")

    parser.add_argument("--host", type=str, nargs=1,
                        help="host:xx.xx.xx.xx", required=True)

    parser.add_argument('--dataset', type=str, nargs=1,
                        help="dataset: sift | deep", required=True)

    args = parser.parse_args()
    host = args.host[0]
    dataset = args.dataset[0]

    print("Host:", host)
    print("Collection:", COLLECTION_NAME)
    print("Dataset:", dataset)

    connect_server(host)
    collection = prepare_collection(dataset)
