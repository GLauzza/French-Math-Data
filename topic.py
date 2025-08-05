import os
import argparse

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

from utils_model import *
from process_data.prepare_data import *

curr_dir = os.path.dirname(os.path.abspath(__file__))


def download_topics():
    os.system(f'wget "https://en.wikipedia.org/w/index.php?title=Lists_of_mathematics_topics&action=raw" --output-document {curr_dir}/wiki_topic/Lists_of_mathematics_topics.txt')
    pages = []
    reading = True
    with open(curr_dir+"/wiki_topic/Lists_of_mathematics_topics.txt", "r") as f:
        for line in f.readlines():
            if line in ["==Mathematical objects==\n", "==Methodology==\n"]:
                reading = False
            if line in ["==General concepts==\n"]:
                reading = True
            if (
                reading and
                "* [[" in line and 
                "see also" not in line.lower() and
                "cryptography" not in line.lower() and
                "prime numbers" not in line.lower() and
                "wave" not in line.lower() and
                "catalog" not in line.lower()
            ):
                pages.append(line.replace("[","").replace("]","").replace("*","").strip().split("|")[0].replace(" ", "_"))

    print(len(pages))
    for page in pages:
        os.system(f'wget "https://en.wikipedia.org/w/index.php?title={page}&action=raw" --output-document {curr_dir}/wiki_topic/{page}.txt')


def get_topics():
    topics = []
    for file in os.listdir(curr_dir+"/wiki_topic/"):
        n_topic = 0
        
        if file.startswith("List_of"):
            topic = file[8:-4].replace("_", " ").replace(" topics", "").strip().capitalize()
        elif file.startswith("Outline_of"):
            topic = file[11:-4].replace("_", " ").strip().capitalize()
        elif file.startswith("Glossary_of"):
            topic = file[12:-4].replace("_", " ").strip().capitalize()
        else:
            topic = file[:-4].replace("_", " ").strip().capitalize()
        topics.append(topic)

        with open(curr_dir+"/wiki_topic/"+file, "r") as f:
            for line in f.readlines():
                if file == "Lists_of_mathematics_topics.txt":
                    pass
                else:
                    if (
                        "*[[" in line or 
                        "* [[" in line or 
                        "* '''[[" in line or
                        "*'''[[" in line or
                        "* {{" in line or 
                        "*{{" in line or
                        "*The [[" in line or 
                        "*The[[" in line
                    ):
                        topic = (
                            line
                            .replace("[[","")
                            .replace("{{","")
                            .replace("*The","")
                            .replace("*","")
                            .replace("'''","")
                            .split("]")[0]
                            .split("}")[0]
                            .split("|")[-1]
                            .strip()
                        )
                    elif line.startswith("{{term|"):    
                        topic = (
                            line
                            .split("}")[0]  
                            .split("]")[0]
                            .split("[")[-1]
                            .split("=")[-1]
                            .split("|")[-1]
                            .strip()
                        )
                    elif line.startswith("; "):
                        topic = (
                            line[2:]
                            .split(":")[0]
                            .replace("[", "")
                            .replace("]", "")
                            .replace("{", "")
                            .replace("}", "")
                            .strip()
                        )
                    else:
                        continue
                    topic = (
                        topic
                        .replace(":Category:", "")
                        .replace("''", "")
                        .replace(":", " ")
                        .replace("&", " ")
                        .replace(";", " ")
                        .replace("<!--", " ")
                        .strip()
                    )
                    if len(topic) > 4 and len(topic) < 75:  
                        if (topic[0] == "'" and topic[-1] == "'") or (topic[0] == '"' and topic[-1] == '"'):
                            topic = topic[1:-1]
                        topics.append(topic.strip().capitalize())
                        if topics[-1][0] < "A" or topics[-1][0] > "Z":
                            print(line, topics[-1], "\n")
                        n_topic += 1
        print(f"{file}: {n_topic} topics\n\n\n")
    topics = list(set(topics))
    print(f"{len(topics)} topics: {sorted(topics)}")
    return topics
  

def get_topic_grammar():
    topics = get_topics()
    return f"""
        root ::= topic_list

        topic_list ::= topics (", " topics)? (", " topics)? (", " topics)? (", " topics)? (".")?

        topics ::= "{'" | "'.join(topics)}"
    """


def plot_topic_embedding(model, dataset):
    topics = get_topics()
    embs = [result.outputs.embedding for result in model.embed(topics)]
    
    pca = PCA(n_components=2)    
    embs_2d = pca.fit_transform(np.array(embs))

    plt.scatter(embs_2d[:,0], embs_2d[:,1])
    plt.savefig(curr_dir+"/embeddings.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Allows to download or plot embedding of topics')
    parser.add_argument('--model', type=str, default="Qwen3-Embedding-8B", help='Model to use for getting the embedding')
    parser.add_argument('--dataset', type=str, default="Train-Math", help='Dataset to compute the embeddings on')
    parser.add_argument('--action', type=str, default="download", help='Which action to do (download or plot topics)')
    args = parser.parse_args()
    if args.action == "download":
        download_topics()
    elif args.action == "plot":
        dataset = load_data(args.dataset)
        model_path, _, _ = get_config(args.model)
        model = load_model(model_path, is_vllm=True)
        plot_topic_embedding(model, dataset)