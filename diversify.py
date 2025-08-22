import os
import argparse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.feature_extraction.text import TfidfVectorizer
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
  

def get_grammar(grammar_name):
    if grammar_name == "topic":
        topics = get_topics()
        return f"""
            root ::= topic_list

            topic_list ::= topics (", " topics)? (", " topics)? (", " topics)? (", " topics)? (".")?

            topics ::= "{'" | "'.join(topics)}"
        """
    elif grammar_name == "difficulty":
        return f"""
            root ::= (number_1_to_12 " grade") | (number_1_to_4 " year undergrad") | "Master" | "PhD" | ("AMC " amc_number) | "AIME" | "USAJMO" | "USAMO" | "MOP" | "IMO" | "Putnam"
            number_1_to_4 ::= "1st" | "2nd" | "3rd" | "4th"
            number_1_to_12 ::= number_1_to_4 | "5th" | "6th" | "7th" | "8th" | "9th" | "10th" | "11th" | "12th"
            amc_number ::= "8" | "10" | "12"
        """
    elif grammar_name == "knowledge" or grammar_name == "steps":
        return f"""
            root ::= non_null_digit (digit)*
            non_null_digit ::= "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
            digit ::= "0" | non_null_digit
        """
    elif grammar_name == "quality":
        return f"""
            root ::= "Correct" | "Incorrect"
        """
        # return f"""
        #     root ::= "Upvote" | "Keep" | "Downvote" | "Remove"
        # """
    else:
        raise Exception(f"Grammar {grammar_name} not supported")



def plot_embedding(model, dataset):
    # to_2d = TSNE(n_components=2)   
    to_2d = umap.UMAP(
        n_components=2,
        n_neighbors=40,
        min_dist=0.5,
        metric='cosine',
        random_state=0
    )
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


    embs_topics_data_2d = []
    for sample in dataset:
        for topic in sample["question_topic"].split(", "):
            if topic[-1] == "." and topic not in topics:
                embs_topics_data_2d.append(embs_topic_2d[topic_to_id[topic[:-1]]])
            elif topic in topics:
                embs_topics_data_2d.append(embs_topic_2d[topic_to_id[topic]])
            else:
                print(f"{topic} not in topics.")
    embs_topics_data_2d = np.array(embs_topics_data_2d)
    
    plt.scatter(embs_topic_2d[:,0], embs_topic_2d[:,1], s=1, alpha=0.1, label="topic")
    plt.scatter(embs_topics_data_2d[:,0], embs_topics_data_2d[:,1], s=1, alpha=0.2, label="topic data")
    plt.legend()
    plt.savefig(curr_dir+"/topic_embeddings.png", dpi=500)
    plt.close()

    kde = KernelDensity(bandwidth=0.1).fit(embs_topic_2d)
    density = np.exp(kde.score_samples(embs_topic_2d))
    topic_sampling = (1/(density))/sum(1/density)
    embs_uniform = []
    for idx in np.random.choice(len(embs_topic_2d), p=topic_sampling, size=10000, replace=True):
        embs_uniform.append(embs_topic_2d[idx])
    embs_uniform = np.array(embs_uniform)
    plt.scatter(embs_uniform[:,0], embs_uniform[:,1], s=20, alpha=0.05)
    plt.savefig(curr_dir+"/topic_sampling.png", dpi=500)
    plt.close()

    for idx in np.random.choice(len(embs_topic_2d), p=topic_sampling, size=100, replace=False):
        x, y = embs_topic_2d[idx, 0], embs_topic_2d[idx, 1]
        plt.scatter(x, y, c='red', s=0)  # Highlight labeled points
        plt.annotate(
            topics[idx],
            (x, y),
            textcoords="offset points",
            xytext=(0, 0),  
            fontsize=2,
            color='red',
            alpha=0.4,
            weight='normal'
        )
    plt.savefig(curr_dir+"/topic_dist.png", dpi=500)
    plt.close()

    # to_2d = TSNE(n_components=2)   
    to_2d = umap.UMAP(
        n_components=2,
        n_neighbors=40,
        min_dist=0.5,
        metric='cosine',
        random_state=0
    )
    # to_2d = PCA(n_components=2)

    embs_question = [result.outputs.embedding for result in model.embed(dataset["question"])]
    embs_question_2d = to_2d.fit_transform(np.array(embs_question))
    
    plt.scatter(embs_question_2d[:,0], embs_question_2d[:,1], s=1, alpha=0.2, label="question")
    plt.legend()
    plt.savefig(curr_dir+"/question_embeddings.png", dpi=500)
    plt.close()

    kde = KernelDensity(bandwidth=0.1).fit(embs_question_2d)
    density = np.exp(kde.score_samples(embs_question_2d))
    topic_sampling = (1/(density))/sum(1/density)
    embs_uniform = []
    for idx in np.random.choice(len(embs_question_2d), p=topic_sampling, size=10000, replace=True):
        embs_uniform.append(embs_question_2d[idx])
    embs_uniform = np.array(embs_uniform)
    plt.scatter(embs_uniform[:,0], embs_uniform[:,1], s=20, alpha=0.05)
    plt.savefig(curr_dir+"/question_sampling.png", dpi=500)
    plt.close()

    for idx in np.random.choice(len(embs_question_2d), p=topic_sampling, size=50, replace=False):
        x, y = embs_question_2d[idx, 0], embs_question_2d[idx, 1]
        plt.scatter(x, y, c='red', s=0)  # Highlight labeled points
        plt.annotate(
            dataset['question'][idx].replace('$', ''),
            (x, y),
            textcoords="offset points",
            xytext=(0, 0),  
            fontsize=2,
            color='red',
            alpha=0.4,
            weight='normal'
        )
    plt.savefig(curr_dir+"/question_dist.png", dpi=500)
    plt.close()


def tok_dist(tokenizer, dataset):
    vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize, lowercase=False)
    freq = vectorizer.fit_transform(dataset["solution"])
    to_2d = TSNE(n_components=2)   
    freq_2d = to_2d.fit_transform(np.array(freq))
    plt.scatter(freq_2d[:,0], freq_2d[:,1], s=1, alpha=0.1)
    plt.savefig(curr_dir+"/tok_dist.png", dpi=500)
    plt.close()



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
        plot_embedding(model, dataset)
    elif args.action == "dist":
        dataset = load_data(args.dataset)
        model_path, _, _ = get_config(args.model)
        model, tokenizer = load_model(model_path)
        tok_dist(tokenizer, dataset)