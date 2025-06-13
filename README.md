# French-Math-Data

### Instructions
You can configure your environnement by calling ```conda env create -f math.yml; conda activate vllmath```

You can choose where your models and datasets will be located in ```config.py```. There is 2 locations for each, allowing to have first Jean Zay DSDIR and second a custom location.

You can download the math datasets by calling ```python ./process_data/get_data.py```.

Once all the datasets are downloaded you can create the CoT and Eval datasets by calling ```python ./create_datasets.py```

To evaluate your models, edit the ```eval.py``` to add the model you want to evaluate along with their chat template function and sampling params in ```model_configs.py```. This will generate an ```eval.json``` file containing the results of the evaluation.

You can explore those results using the ```plot_eval.ipynb``` notebook.

### TODOs

- [ ] check llama-nemotron answer correctly extracted
- [ ] check sourcing and models
- [ ] process pensez subsets
- [ ] handle fact that am deepseek distill contains non math interesting data (llama-nemotron)
- [ ] filter too easy according to Lucie
- [ ] filter sources we don't want
- [ ] put hyperparams of filters in config
- [ ] keep categories features of some datasets
- [ ] handle openr1 finish reason, s1 cot type, numina/openr1 question type filtering
- [ ] evaluate if answer length / question lenght / CoT length is correlated with accuracy
- [ ] format the training data such that CoT ends by \\boxed{}
- [ ] do I need to train to predict end of text token ?
- [ ] add custom response template
- [ ] add SFT checkpointing
- [ ] see if math verify not supporting french is a problem
- [ ] check how NEMO handles chat templates for finetuning
- [ ] think on how to pad during training (padding free doesn't work with custom datacollator)
- [ ] fix eval of n samples (not sure about the output format)
- [ ] qwen3 disable thinking for translation
- [ ] add variable chat templates depending on config type
- [ ] add answer and training on french dataset (check if answers need to be translated)
- [ ] add language forcing
- [ ] do we keep learning signal from failed samples (contrastive / imitating RL with SFT)
- [ ] check extraction of boxed works perfectly
- [ ] get sampling params this way: "Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`"
- [ ] comment gerer les balises think ?
- [ ] budget control