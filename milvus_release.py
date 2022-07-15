# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import time
import os
import numpy as np
import signal
import sys

from threading import Timer

from pymilvus import (
    list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, Partition,
    connections, utility
)

from common import *

def connect_server(host):
    # configure milvus hostname and port
    print("connecting ...")
    connections.connect(host=host, port=19530)
    print("connected")


def release_collection():
    print("start to release")
    collection = Collection(name=COLLECTION_NAME)
    collection.release()
    print("release done")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Insert Data to Milvus user-defined ranges of .npy files")
    parser.add_argument("--host", type=str, nargs=1,
                        help="host:xx.xx.xx.xx", required=True)

    args = parser.parse_args()
    host = args.host[0]

    print("Host:", host)
    print("Collection:", COLLECTION_NAME)

    connect_server(host)
    release_collection()
