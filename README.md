# Compositional Structured Explanation Generation with Dynamic Modularized Reasoning

Code and Data for the \*SEM paper 


We propose a new task, compositional structured explanation generation (CSEG), to facilitate research on compositional generalization in reasoning. CSEG tests a model's ability to generalize from generating entailment trees with a limited number of inference steps to more steps. 

We propose a new dynamic modularized reasoning model, MORSE, that factorizes the inference process into modules, where each module represents a functional unit. 


## Code
1. Prepare the dataset
2. Do pre-training with primitve data *args.do_finetuning = "p1"*
3. Do fine-tuning with composition data *args.do_finetuning = "p2"*
3. Run the script
```
sh run-test-module.sh
```

## Data Download (OneDrive)
We reset the data for compositional generalization tests. Details can be found in our paper.

**Link**: https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/fuxiyan_mail_nankai_edu_cn/EoU7dqOMkldIlYPbL1kw9E8B6VErBodogMQjEXAg5JfeyQ?e=M3Sw7Z

**Files**: data for DBPedia and EnatailmentBank

## Evaluation
For evaluation, we follow ['Explaining Answers with Entailment Trees'](https://github.com/allenai/entailment_bank):
1. Download the bleurt-large-512 model from https://github.com/google-research/bleurt/blob/master/checkpoints.md under 'scorer folder/'
2. Set the evaluation parameter 'evaluation_root_dir'
3. Run the script


## Citations
````
@inproceedings{fu-frank-2024-compositional,
    title = "Compositional Structured Explanation Generation with Dynamic Modularized Reasoning",
    author = "Fu, Xiyan  and  Frank, Anette",
    editor = "Bollegala, Danushka  and  Shwartz, Vered",
    booktitle = "Proceedings of the 13th Joint Conference on Lexical and Computational Semantics (*SEM 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.starsem-1.31/",
    doi = "10.18653/v1/2024.starsem-1.31",
    pages = "385--401",
}
```
