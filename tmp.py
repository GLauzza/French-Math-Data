import datasets

import config
from process_data.utils_data import *

math500 = load_data("HuggingFaceH4/MATH-500", split="test")
math500 = math500.rename_column("problem", "question")
math500 = math500.add_column("source", ["math500" for _ in range(len(math500))])
math500.save_to_disk(config.DATA_PATHS[1] + "tmp")
