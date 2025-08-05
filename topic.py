import os

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
                "cryptography" not in line and
                "prime numbers" not in line and
                "wave" not in line
            ):
                pages.append(line.replace("[","").replace("]","").replace("*","").strip().split("|")[0].replace(" ", "_"))

    print(len(pages))
    # for page in pages:
    #     os.system(f'wget "https://en.wikipedia.org/w/index.php?title={page}&action=raw" --output-document {curr_dir}/wiki_topic/{page}.txt')


def get_topics():
    topics = []
    with open(curr_dir+"/wiki_topic/Glossary_of_areas_of_mathematics.txt", "r") as f:
        for line in f.readlines():
            if line.startswith("{{term|"):
                topics.append(line[9:-3].replace("]","").split("|")[-1])
    for file in os.listdir(curr_dir+"/wiki_topic/"):
        n_topic = 0
        with open(curr_dir+"/wiki_topic/"+file, "r") as f:
            for line in f.readlines():
                if "*[[" in line or  "* [[" in line:
                    topic = line.replace("[","").replace("*","").split("]")[0].strip().split("|")[-1]
                    if len(topic) > 4 and len(topic) < 75:
                        topics.append(topic)
                        print(topics[-1])
                        n_topic += 1
        print(f"{file}: {n_topic} topics\n\n\n")
    print(f"{len(topics)} topics")
    return topics
  

def get_topic_grammar():
    topics = get_topics()
    return f"""
        root ::= topic_list

        topic_list ::= topics (", " topics)? (", " topics)? (", " topics)? (", " topics)? (".")?

        topics ::= "{'" | "'.join(topics)}"
    """


if __name__ == "__main__":
    # download_topics()
    get_topics()