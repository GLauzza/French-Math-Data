def get_topics():
    return ["Algebra", "Boolean Theory", "Complex numbers", "Discrete maths"]
  

def get_topic_grammar():
    topics = get_topics()
    return f"""
        root ::= topic_list

        topic_list ::= topics (", " topics)? (", " topics)? (", " topics)? (", " topics)? (".")?

        topics ::= "{'" | "'.join(topics)}"
    """