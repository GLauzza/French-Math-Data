import argparse
import sys
import os 
import shutil

from semhash import SemHash
from datasets import load_dataset, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config
from prepare_data import *
from utils_model import *


def merge_same_answer(records):
    merged_clusters = []
    for rec in records:
        cluster_ind = None
        for i, cluster in enumerate(merged_clusters):
            if cluster[0]["answer"] == rec["answer"]:
                cluster_ind = i
        if cluster_ind is None:
            merged_clusters.append([rec])
        else:
            merged_clusters[cluster_ind].append(rec)
    return merged_clusters


def dedup(model, dataset, threshold, n, criterion):
    print("a")
    # Convert to record format
    records = [dict(sample) for sample in dataset]
    print("b")

    # Build SemHash index
    semhash_index = SemHash.from_records(
        records=records,
        columns=["question"],
        model=model,
        use_ann=True
    )

    print("FM - Deduplicating")
    dedup_result = semhash_index.self_deduplicate(threshold=threshold)

    dedup_samples = []

    for dup in dedup_result.filtered:
        all_records = [dup.record] + [sample[0] for sample in dup.duplicates]
        # Merge cluster if answers in duplicates match
        dup_clusters = merge_same_answer(all_records)
        for cluster in dup_clusters:
            if criterion == "shortest_cot":
                sorted_cluster = sorted(cluster, key=(lambda x: len(x["solution"])))
            else:
                raise Exception("Criterion not supported")
            dedup_samples.extend(sorted_cluster[:n])

    print(f"Filtered {100 * (len(records) - len(dedup_samples)) / len(records)}% of the dataset")
    print("FM - Deduplicated")
    return Dataset.from_list(dedup_samples)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs a task on a dataset using a model')
    parser.add_argument('--model', type=str, default="Qwen3-32B-FP8-dynamic", help='Model to use for embedding')
    parser.add_argument('--dataset', type=str, default="Fused-CoT", help='Dataset to dedup')
    parser.add_argument('--threshold', type=float, default=0.88, help='Threshold for self_deduplicate (default: 0.77)')
    parser.add_argument('--n', type=int, default=1, help='Number of samples to keep per duplicate group (default: 1)')
    parser.add_argument('--criterion', type=str, default="shortest_cot", help='Criterion to select which samples of duplicates to keep')
    args = parser.parse_args()

    model_path, _, _ = get_config(args.model)
    model = load_model(model_path)    

    dataset = load_data(args.dataset).shuffle(seed=0)
    
    new_dataset_name = args.dataset + "-Dedup"

    dataset = dedup(model, dataset, args.threshold, args.n, args.criterion)

    print("FM - Saving")
    dataset.save_to_disk(config.DATA_PATHS[1] + new_dataset_name)
    shutil.copytree(config.DATA_PATHS[1] + new_dataset_name, config.DATA_PATHS[2] + new_dataset_name, dirs_exist_ok=True)
    print("FM - Saved")