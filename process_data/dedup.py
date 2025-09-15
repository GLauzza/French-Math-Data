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


def dedup(model, dataset, threshold):
    # Convert to record format
    records = [dict(sample) for sample in dataset]

    # Build SemHash index
    semhash_index = SemHash.from_records(
        records=records,
        columns=["question"],
        model=model,
        use_ann=False
    )

    print("FM - Deduplicating")
    dedup_result = semhash_index.self_deduplicate(threshold=threshold)

    filtered_duplicates = []
    kept_duplicates = []

    for dup in dedup_result.duplicates:
        if dup.duplicates:
            rep_answer = dup.record.get("answer", "")
            if type(rep_answer) == str:
                rep_answer = rep_answer.strip()
            
            all_same_answer = all(
                rep_answer == (rec.get("answer", "").strip() if type(rec.get("answer", "")) == str else rec.get("answer", ""))
                for rec, _ in dup.duplicates
            )
            if all_same_answer:
                filtered_duplicates.append(dup)   
            else:
                kept_duplicates.append(dup)       

    print(f"Kept (unique + different answers): {len(dedup_result.selected) + len(kept_duplicates)}")
    print(f"Dropped (duplicates with same answer): {len(filtered_duplicates)}\n")
    # print("== Dropped ==")
    # for dup in filtered_duplicates:
    #     print("Rep:", dup.record)
    #     for rec, score in dup.duplicates:
    #         print(f"   Dup: {rec} (sim={score:.2f})")
    #     print()

    # print("== Kept ==")
    # for dup in kept_duplicates:
    #     print("Rep:", dup.record)
    #     for rec, score in dup.duplicates:
    #         print(f"   Dup: {rec} (sim={score:.2f})")
    #     print()
    print("FM - Deduplicated")

    print(kept_duplicates[:5])
    return Dataset.from_list(dedup_result.selected)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs a task on a dataset using a model')
    parser.add_argument('--model', type=str, default="Qwen3-32B-FP8-dynamic", help='Model to use for embedding')
    parser.add_argument('--dataset', type=str, default="Fused-CoT", help='Dataset to dedup')
    parser.add_argument('--threshold', type=float, default=0.88, help='Threshold for dedup')
    args = parser.parse_args()

    model_path, _, _ = get_config(args.model)
    model = load_model(model_path)    

    dataset = load_data(args.dataset).shuffle(seed=0)
    
    new_dataset_name = args.dataset + "-Dedup"

    dataset = dedup(model, dataset, args.threshold)

    print("FM - Saving")
    dataset.save_to_disk(config.DATA_PATHS[1] + new_dataset_name)
    shutil.copytree(config.DATA_PATHS[1] + new_dataset_name, config.DATA_PATHS[2] + new_dataset_name, dirs_exist_ok=True)
    print("FM - Saved")