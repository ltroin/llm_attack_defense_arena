# Baseline Defenses for Adversarial Attacks Against Aligned Language Models
Official Code for "Baseline Defenses for Adversarial Attacks Against Aligned Language Models"

## Overview
We evaluate several baseline defense strategies against leading adversarial attacks on LLMs, discussing the various settings in which each is feasible and effective. Particularly, we look at three types of defenses: detection (perplexity based), input preprocessing (paraphrase and retokenization), and adversarial training. The paper can be found [here](https://arxiv.org/abs/2309.00614).

The repository only contains the code for the perplexity filter and paraphrase attack. The retokenization defenses is conducted directly via altering tokenizer via BPE-dropout. For LLaMA model, see the `tokenizer.sp_model.encode(input_text, alpha=bt_alpha, enable_sampling=True)` function, and for other models, BPE-dropout is set by `tokenizer._tokenizer.model.dropout=bt_alpha`, where `bt_alpha` is the dropout rate.

## Perplexity Filter

The perplexity filter in the code consists of two filters, a perplexity filter which as also been proposed in concurrent work by [Alon et al.](https://arxiv.org/abs/2308.14132) and a windowed perplexity filter, which consists of checking the perplexity of a window of $n$ tokens.

## Paraphrase Defense

The paraphrase defense is rewriting the prompt. For our experiments, we used ChatGPT. Note while this defense is effective it might come at high performance cost.

## Limitations
As in all research work, we were limited to the settings we explored in the paper. 
