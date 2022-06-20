# This program demos how to connect to Milvus vector database, 
# create a vector collection,
# insert 10 vectors, 
# and execute a vector similarity search.

import random, time, argparse
import numpy as np

from milvus import Milvus, IndexType, MetricType, Status


# _HOST = '172.18.50.5'
_PORT = '19530'  # default value


# base params
_COLLECTION_NAME = "sift"
_PARTITION_NUM = 1
_NB = 1000000
_DIM = 128
_INDEX_FILE_SIZE = 4096


# dataset params
sift_dir_path = "/home/sheep/data-mnt/milvus/raw_data/sift10m/"
ID_COUNTER = 0
PER_FILE_ROWS = 100000


# index params
_INDEX_TYPE = "HNSW"
_EF_CONSTRUCTION = 150
_M = 12


# search params
# SEARCH_TIMES = 1000
# SEARCH_PARTITION_NUM = 10

_TOPK = 50
_EF = 50
# NQS = [1, 10]


def connect(host, port):
    milvus = Milvus(host, port)
    return milvus


def create_collection(milvus):
    status, ok = milvus.has_collection(_COLLECTION_NAME)
    if not status.OK():
        raise (Exception("has_collection failed"))
    if ok:
        drop_collection(milvus)

    param = {
        'collection_name': _COLLECTION_NAME,
        'dimension': _DIM,
        'index_file_size': _INDEX_FILE_SIZE,  # optional
        'metric_type': MetricType.L2  # optional
    }

    milvus.create_collection(param)
    print("create collection {}".format(_COLLECTION_NAME))

    # _, collections = milvus.list_collections()
    # print(collections)

    # _, collection = milvus.get_collection_info(_COLLECTION_NAME)
    # print(collection)


def get_partition_name(i):
    return "p" + str(i)    


def create_partitions(milvus):
    for i in range(_PARTITION_NUM):
        milvus.create_partition(_COLLECTION_NAME, get_partition_name(i))


def get_dataset_file_path(dataset_file_index):
    return sift_dir_path+"binary_128d_" + "%.5d"%dataset_file_index + ".npy"


def insert(milvus):
    insert_times_per_partition = int(_NB / _PARTITION_NUM / PER_FILE_ROWS)
    for i in range(_PARTITION_NUM):
        for j in range(insert_times_per_partition):
            dataset_file_index = i * insert_times_per_partition + j
            file_path = get_dataset_file_path(dataset_file_index)
            insert_file_to_partition(milvus, file_path, get_partition_name(i))


def insert_file_to_partition(milvus, fname, partition_name):
    global ID_COUNTER
    data = np.load(fname)
    block_size = PER_FILE_ROWS
    entities = data.tolist()

    print("Insert {} rows to Partition({}), ID start from {}, dataset_file = {}".format(block_size, partition_name, ID_COUNTER, fname))
    status, ids = milvus.insert(collection_name=_COLLECTION_NAME, records=entities, partition_tag=partition_name)
    if not status.OK():
        raise (Exception("insert failed"))
    if len(ids) != block_size:
        raise (Exception("insert failed, len(ids) = " + str(len(ids))))

    ID_COUNTER = ID_COUNTER + block_size


def flush(milvus):
    milvus.flush([_COLLECTION_NAME])
    status, _ = milvus.count_entities(_COLLECTION_NAME)
    if not status.OK():
        raise (Exception("count_entities failed"))
    # _, info = milvus.get_collection_stats(_COLLECTION_NAME)
    # print(info)


def create_index(milvus):
    index_param = {
        'M': _M,
        "efConstruction": _EF_CONSTRUCTION
    }

    print("Creating index: {}".format(index_param))
    status = milvus.create_index(_COLLECTION_NAME, IndexType.HNSW, index_param)
    if not status.OK():
        raise (Exception("create_index failed"))

    # describe index, get information of index
    status, index = milvus.get_index_info(_COLLECTION_NAME)
    print(index)


def gen_search_vectors(milvus, fname, nq):
    data = np.load(fname)
    vectors = data.tolist()
    return vectors[:nq]


def run_search(milvus, fname, search_times, search_partition_num, nqs):
    partition_names = []
    for i in range(search_partition_num):
        partition_names.append(get_partition_name(i))
    for nq in nqs:
        query_vectors = gen_search_vectors(milvus, fname, nq)
        search(milvus, query_vectors, search_times, partition_names)


def search(milvus, query_vectors, search_times, partition_names):
    # execute vector similarity search
    search_param = {
        "ef": _EF
    }

    param = {
        'collection_name': _COLLECTION_NAME,
        'query_records': query_vectors,
        'top_k': _TOPK,
        'partition_tags': partition_names,
        'params': search_param,
    }

    total_run_time = 0
    for _ in range(search_times):
        start = time.time()
        status, results = milvus.search(**param)
        search_time = time.time() - start
        # check_search_results(status, results)
        total_run_time = total_run_time + search_time
    avg_time = total_run_time * 1.0 / search_times
    print("TopK NQ AvgTime/{} SearchPartitions".format(search_times))
    print("{} {} {} {}".format(_TOPK, len(query_vectors), avg_time, partition_names))   


def check_search_results(status, results):
    if status.OK():
        # indicate search result
        # also use by:
        #   `results.distance_array[0][0] == 0.0 or results.id_array[0][0] == ids[0]`
        if results[0][0].distance == 0.0:
            print('Query result is correct')
        else:
            raise (Exception('Query result isn\'t correct'))

        # print results
        # print(results)
    else:
        raise (Exception("Search failed"))


def drop_collection(milvus):
    # Delete demo_collection
    status = milvus.drop_collection(_COLLECTION_NAME)
    if not status.OK():
        raise (Exception("drop_collection failed"))
    print("Drop collection {}".format(_COLLECTION_NAME))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Milvus 1.0 Tests")

    parser.add_argument("--host", type=str, nargs=1, help="xx.xx.xx.xx", required=True)

    args = parser.parse_args()
    host = args.host[0]

    print("host:{}".format(host))
    milvus = connect(host, _PORT)
    create_collection(milvus)
    create_partitions(milvus)
    insert(milvus)
    flush(milvus)
    create_index(milvus)
