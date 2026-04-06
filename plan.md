# Task

I want to redo two RL runs:
- RL with no prefill
- RL with craft prefill
You may look at wandb to see what parameterws we used for these runs.

Crucially, this is a runpod with 6 H100 GPUs. Adapt the hyperparameters accordingly from these runs accordingly. If the choice is between num_generations or speed of training, prioritise speed of training. In particular, I don't want to change the training dynamics; I just want to rerun the same RL runs, simply faster.

# Evaluation

This is essentially a redo of experiments we have already done. However, the evaluations showed only mild improvement on the evaluation set. I was thinking of using the APPS dataset to evaluate the two models we obtain after these RL runs. The point is that there should be a significant difference in the performance of the two models (as measured by %correctness).