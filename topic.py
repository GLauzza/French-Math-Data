import os
import argparse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
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
    to_2d = TSNE(n_components=2)   
    # to_2d = umap.UMAP(
    #     n_components=2,           # Number of dimensions to reduce to
    #     n_neighbors=40,           # Balance local vs global structure
    #     min_dist=0.5,             # Minimum distance between points
    #     metric='cosine',       # Distance metric
    #     random_state=0           # For reproducibility
    # )
    # to_2d = PCA(n_components=2)

    topics = get_topics()
    topic_to_id = {topic:i for i, topic in enumerate(topics)}

    embs_topics = [result.outputs.embedding for result in model.embed(topics)]
    embs_topic_2d = to_2d.fit_transform(np.array(embs_topics))
    # embs_data = [result.outputs.embedding for result in model.embed(dataset["question"])]
    # embs_data_2d = to_2d.transform(np.array(embs_data))
    # embs = to_2d.fit_transform(np.concatenate([embs_topics, embs_data], axis=0))
    # embs_topic_2d = embs[:len(embs_topics)]
    # embs_data_2d = embs[len(embs_topics):]


    embs_data_2d = []
    for sample in dataset:
        for topic in sample["question_topic"].split(", "):
            if topic[-1] == "." and topic not in topics:
                embs_data_2d.append(embs_topic_2d[topic_to_id[topic[:-1]]])
            elif topic in topics:
                embs_data_2d.append(embs_topic_2d[topic_to_id[topic]])
            else:
                print(f"{topic} not in topics.")
    embs_data_2d = np.array(embs_data_2d)
    
    plt.scatter(embs_topic_2d[:,0], embs_topic_2d[:,1], s=1, alpha=0.1, label="topic")
    plt.scatter(embs_data_2d[:,0], embs_data_2d[:,1], s=1, alpha=0.2, label="question")
    plt.savefig("embeddings.png", dpi=500)
    plt.show()

    kde = KernelDensity(bandwidth=0.1).fit(embs_topic_2d)
    log_density = np.exp(kde.score_samples(embs_topic_2d))
    topic_sampling = (1/(log_density))/sum(1/log_density)
    embs_uniform = []
    for idx in np.random.choice(len(embs_topic_2d), p=topic_sampling, size=10000, replace=True):
        embs_uniform.append(embs_topic_2d[idx])
    embs_uniform = np.array(embs_uniform)
    plt.scatter(embs_uniform[:,0], embs_uniform[:,1], s=20, alpha=0.05)
    plt.savefig("topic_sampling.png", dpi=500)
    plt.show()

    for idx in np.random.choice(len(embs_topic_2d), p=topic_sampling, size=300, replace=False):
        x, y = embs_topic_2d[idx, 0], embs_topic_2d[idx, 1]
        plt.scatter(x, y, c='red', s=0)  # Highlight labeled points
        plt.annotate(
            topics[idx],
            (x, y),
            textcoords="offset points",
            xytext=(0, 0),
            fontsize=1,
            color='red',
            alpha=0.4,
            weight='normal'
        )
    plt.savefig("topic_dist.png", dpi=500)
    plt.show()


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