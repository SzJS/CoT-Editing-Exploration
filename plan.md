# Multi-GPU training
This is a 2xH100 pod. I want to utilise both GPUs for RL training on gpt-oss-20b. However, this may require some changes in the codebase; I'm not sure. See this guide on how to do multi-GPU with TRL: https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp

## Requirements:
- The model we RL train should be the one we have SFT'd to reward hack sometimes, (see: https://huggingface.co/Thessalonican17/gpt-oss-20b-sft-synthetic-cp15-bnb4bit and https://huggingface.co/Thessalonican17/gpt-oss-20b-sft-synthetic-cp15-mxfp4)

## Questions:
Before making any plans, look into the following problems and get back to me with your answers.
- Are two H100s going to be enough for this? Does DDP help?
- Ideally, with the multi-GPU set up, I would like to use medium reasoning effort. Would this be possible?

# Easier RL environment
I want to create an RL environment composed of simpler coding problems. The aim of this is so that model can learn to solve most problems during RL. Right now, the model learns only get only slightly better at the problems during the RL run, which makes it harder to derive statistical significance. If we could widen the gap from pre-RL to post-RL, we would need fewer reruns to show statistical significance.

## Requirements:
- I want to be able to reuse the current training pipeline. In other words, I would like to keep the current structure. (Including the above implemented multi-GPU training)
- Currently, the training and eval sets overlap. Therefore, I would also like the new environment to have enough samples so that we can have a separate eval set.

## Possibilities
I think we could use the APPS dataset (link: https://huggingface.co/datasets/codeparrot/apps). I think we could use the inputoutput column for the creation of assert statements. I believe that some of the code is already implemented. What I remember is that the easiest problems may be too easy to solve. We would need to test by running for 100 steps (on benign problems only).