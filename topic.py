import pandas as pd

import config


def get_topics():
    pass
  

def get_topic_grammar():
    topics = get_topics()
    return f"""
        root ::= topic_list

        topic_list ::= {{topics, }}[topics]

        topics ::= "{'" | "'.join(topics)}"
    """
