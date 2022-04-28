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
_PARTITION_NUM = 10


# dataset params
sift_dir_path = "/home/sheep/data-mnt/milvus/raw_data/sift10m/"


_TOPK = 50
_EF = 50


def connect(host, port):
    milvus = Milvus(host, port)
    return milvus


def get_partition_name(i):
    return "p" + str(i)    


def get_query_file_path():
    return sift_dir_path+"query.npy"


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
    NQ = len(query_vectors)
    qps = NQ*1.0/avg_time
    print("TopK NQ AvgTime/{} QPS SearchPartitions".format(search_times))
    print("{} {} {} {} {}".format(_TOPK, NQ, avg_time, qps, partition_names))


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
    parser.add_argument("--runnums", type=int, nargs=1, help="search run times of each nq", required=True)
    parser.add_argument("--p", type=int, nargs=1, help="number of partitions to search", required=True)
    parser.add_argument("--nqs", type=int, nargs='+', help="nqs to search", required=True)

    args = parser.parse_args()
    host = args.host[0]
    runtimes = args.runnums[0]
    search_partition_num = args.p[0]
    nqs = args.nqs

    print("host:{}, runtimes:{}, search_partition_num:{}, nqs:{}".format(host, runtimes, search_partition_num, nqs))
    milvus = connect(host, _PORT)
    run_search(milvus, get_query_file_path(), runtimes, search_partition_num, nqs)
