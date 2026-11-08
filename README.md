# rm-blindspot-forecast

Tools for auditing a reward model against a given LLM before RL training. The audit samples the LLM's answers to math problems, grades them with the reward model, checks them against the ground truth, and surfaces answers that the reward model grades highly although they are wrong.

Status: early development; the code is being added.
