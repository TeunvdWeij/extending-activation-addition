# Extending Activation Steering to Broad Skills and Multiple Behaviours
You can find the paper here: https://arxiv.org/abs/2403.05767 and the [tweet thread here](https://twitter.com/Teun_vd_Weij/status/1767728559416570049)!
## Abstract
Current large language models have dangerous capabilities, which are likely to become more problematic in the future. Activation steering techniques can be used to reduce risks from these capabilities. In this paper, we investigate the efficacy of activation steering for broad skills and multiple behaviours. First, by comparing the effects of reducing performance on general coding ability and Python-specific ability, we find that steering broader skills is competitive to steering narrower skills. Second, we steer models to become more or less myopic and wealth-seeking, among other behaviours. In our experiments, combining steering vectors for multiple different behaviours into one steering vector is largely unsuccessful. On the other hand, injecting individual steering vectors at different places in a model simultaneously is promising.


![image](https://github.com/TeunvdWeij/output_control/assets/57399756/b2b03f96-62bf-43e7-bb4b-504da33e8cb1)

## Repository outline
```
├── bash_scripts (various scripts to run on high-performance cluster)
├── data
│   ├── activations (stored activations of different runs)
│   ├── datasets (behaviour dataset and how to download them)
│   └── skip_tokens (tokens to skip, not used in paper)
├── notebooks
│   ├── make_coding_plots.ipynb 
│   ├── misc_investigations.ipynb (activation pattern of steering vector)
│   ├── multi_steering.ipynb (run the multi steering experiments)
│   └── plot_multiple_concept_steering.ipynb
├── plots (various plots, most important ones are in the paper)
├── results (mostly json files of experimental results, not all are in paper and some are old)
├── src
│   ├── activation_tensor.py (class to save activations and its meta data)
│   ├── evaluate.py (main evaluation workflow)
│   ├── evaluation.py (class to store and process evaluation results)
│   ├── generate_activations.py (main activation generation workflow)
│   ├── model.py (wrapper aroudn the Llama2 models)
│   ├── multi_steering_activation_tensor.py (not used for paper)
│   ├── multi_steering_generate_activations.py (not used for paper)
│   ├── skip_tokens.py (not used for paper)
│   └── utils.py 
```
Due to a lack of time and other priorities, not all of the code is at neat as it maybe should be. Please reach out to me (mailvanteun@gmail.com) if you have any questions about the code!
