def extract_boxed_text(x):
    splitted = x.split("\\boxed{")
    if len(splitted) == 1:
        return ""
    last_boxed = splitted[-1]
    n_left = 1
    n_right = 0
    output = ""
    for char in last_boxed:
        if char == "}":
            n_right += 1
        elif char == "{":
            n_left += 1
        if n_left == n_right:
            return output
        output += char
    return ""


def remove_str(text):
    if len(text) < 2:
        return text
    if (text[0] == "'" and text[-1] == "'") or (text[0] == "'" and text[-1] == "'"):
        return text[1:-1]
    else:
        return text


def to_latex(text):
    if text is None:
        return "$$"
    if len(text) >= 2 and ((text[0] == "$" and text[-1] == "$") or (text[0] == "[" and text[-1] == "]")):
        base_text = text[1:-1]
    elif len(text) >= 4 and ((text[:2] == "\\[" and text[-2:] == "\\]") or (text[:2] == "\\(" and text[-2:] == "\\)")):
        base_text = text[2:-2]
    elif len(text) >= 8 and ((text[:7] == "\\boxed{" and text[-1] == "}")):
        base_text = text[7:-1]
    else:
        base_text = text
    return "$" + remove_str(base_text) + "$"