topics = [
    "Linear Algebra", "Inequalities", "Proof"
]


topic_grammar = f"""
    root ::= topic_list

    topic_list ::= {{topics, }}[topics]

    topics ::= "{'" | "'.join(topics)}"
"""

print(topic_grammar)
