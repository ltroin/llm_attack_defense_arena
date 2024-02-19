# Certifying LLM Safety against Adversarial Prompting

This is the official repository for the code accompanying the paper [Certifying LLM Safety against Adversarial Prompting](https://arxiv.org/abs/2309.02705).

## Introduction
Large language models (LLMs) released for public use are often fine-tuned to "align" their behavior to ensure they do not produce harmful or objectionable content. When prompted to produce inappropriate content, a well-aligned LLM should decline the user's request.
Following is an example of how an aligned LLM would respond when prompted with a harmful prompt:

<p align="center">
  <img src="figures/harmful_prompt.png" width="600"/>
</p>

However, such safety measures have been shown to be vulnerable to **adversarial prompts**, which add maliciously designed sequences of tokens to a harmful prompt aimed at making an LLM produce harmful content despite being well-aligned.
Given a harmful prompt P, which should be rejected by the LLM, it is possible to generate an adversarial prompt P + [adv] that can deceive the LLM into thinking that the prompt is safe and make the LLM comply with it.
Following is an example of an adversarial attack on the above prompt:

<p align="center">
  <img src="figures/adversarial_attack.png" width="600"/>
</p>

Moreover, the generation of such prompts can be entirely automated, creating an endless supply of quick and easy attacks.

The goal of this project is to design a *certified* safety procedure to defend against adversarial prompts. Given a harmful prompt P, the objective is to guarantee that the prompt is detected as harmful even if it is modified with an adversarial sequence [adv] as long as the length of [adv] is bounded.
We design a framework, **erase-and-check**, that erases tokens one by one from the input prompt and evaluates the erased subsequences with a safety filter.
Our procedure labels the input prompt as harmful if any subsequences or the input prompt itself are detected as harmful by the filter. The safety certificate of our procedure guarantees that harmful prompts are not misclassified as safe under an adversarial attack up to a certain size.
Following is an illustration of erase-and-check for a harmful prompt with an adversarial suffix:

<p align="center">
  <img src="figures/erase-and-check.png" width="600"/>
</p>


## This Repository
The file `defenses.py` implements the safety filter and different versions of the erase-and-check procedure. The file `main.py` is the main evaluation script for the experiments in the paper. The `data` directory contains the safe and the harmful prompts used for the experiments.

As shown in the paper, the accuracy of erase-and-check is lower bounded by the performance of the safety filter.
Thus, to compute the certified accuracy of erase-and-check onharmful prompts, we only need the accuracy of the safety filter on the harmful prompts.
To evaluate the performance of the Llama 2-based safety filter on the harmful prompts, run:
```
python main.py --num_prompts 520 --eval_type harmful --harmful_prompts data/harmful_prompts.txt
```
This accuracy is around **92%**.

To generate the results for the different attack modes, run the following scripts from the directory named `bash scripts`:

Adversarial Suffix: `jobs_suffix.sh`

Adversarial Insertion: `jobs_insert.sh`

Adversarial Infusion: `jobs_infuse.sh`

Smoothing-based Certificates: `jobs_smoothing.sh`

Each of these scripts will produce a JSON file in the `results` directory. Use the following plotting scripts from `plot scripts` to generate the plots in the paper:

For accuracy on safe prompts, run:
```
python plot\ scripts/plot_acc.py results/[result file].json
```

For plotting running time, run:
```
python plot\ scripts/plot_time.py results/[result file].json
```

For comparison plot for the smoothing-based certificates, run:
```
python plot\ scripts/plot_smoothing.py results/[result file].json
```

## Performance on Safe Prompts
In our work, we seek to certify the performance of erase-and-check only on the harmful prompts and not the safe prompts.
Attacking a safe prompt makes little sense in practice as it is unlikely that a user will seek to make their safe prompts look harmful to an aligned LLM only to get their request rejected.
However, it is important to ensure that the quality of service does not degrade significantly for a non-malicious, non-adversarial user.
To that end, we evaluate the empirical performance of the erase-and-check procedure on clean safe prompts.

Following is the detection accuracy and the average running time per prompt w.r.t. the maximum erase length for the suffix mode on 520 safe prompts from `safe_prompts.txt`:

<p align="center">
  <img src="results/safe_suffix_520_acc.png" width="300" style="margin-right: 20px;"/>
  <img src="results/safe_suffix_520_time.png" width="300"/>
</p>

Example command:
```
python main.py --num_prompts 520 --mode suffix --eval_type safe --max_erase 20 --safe_prompts data/safe_prompts.txt
```
Use the script `jobs_suffix.sh` in `bash scripts` to produce the results in the above plots.

Similarly, for the insertion mode (100 prompts, one adversarial insertion):
<p align="center">
  <img src="results/safe_insertion_100_acc.png" width="300" style="margin-right: 20px;"/>
  <img src="results/safe_insertion_100_time.png" width="300"/>
</p>

Example command:
```
python main.py --num_prompts 100 --mode insertion --eval_type safe --max_erase 8 --num_adv 1 --safe_prompts data/safe_prompts.txt
```
Use the script `jobs_insertion.sh` in `bash scripts` to produce the results in the above plots.

Similarly, for the infusion mode (100 prompts):
<p align="center">
  <img src="results/safe_infusion_100_acc.png" width="300" style="margin-right: 20px;"/>
  <img src="results/safe_infusion_100_time.png" width="300"/>
</p>

Example command:
```
python main.py --num_prompts 100 --mode infusion --eval_type safe --max_erase $max_erase --safe_prompts data/safe_prompts.txt
```
Use the script `jobs_infusion.sh` in `bash scripts` to produce the results in the above plots.

## Trained Safety Classifier
While we can obtain good performance by simply prompting Llama 2 to classify safe and harmful prompts, running a large language model is computationally expensive and requires significant amounts of processing power and storage capacity.
Furthermore, since Llama 2 is not specifically trained to recognize safe and harmful prompts, its accuracy decreases against longer adversarial sequences.
We show that it is possible to improve the performance of erase-and-check by replacing the safety filter with a text classifier trained to distinguish safe and harmful prompts.
We download a pre-trained DistilBERT model from Hugging Face and fine-tune it on our safety dataset. DistilBERT is a faster and lightweight version of the BERT language model.

Following is a comparison between the performance of erase-and-check with Llama 2 vs. DistilBERT as the safety filter for the insertion mode:
<p align="center">
  <img src="results/comparison_safe_suffix_acc.png" width="300" style="margin-right: 20px;"/>
  <img src="results/comparison_safe_suffix_time.png" width="300"/>
</p>

Similarly, for the insertion mode:
<p align="center">
  <img src="results/comparison_safe_insertion_acc.png" width="300" style="margin-right: 20px;"/>
  <img src="results/comparison_safe_insertion_time.png" width="300"/>
</p>


To reproduce the results for the DistilBERT safety classifier, first train the classifier using the script `distil_bert_toxic_classifier.py` in the `safety_classifier` directory. We train the classifier with erased versions of the safe prompts, available in data files with the tag `_erased`, to help them recognize erased safe promtps as safe as well. Use the `main.py` script with the option `--use_classifier` to evaluate on safe and harmful prompts:

```
python main.py --num_prompts 120 --eval_type harmful --use_classifier --model_wt_path models/[model-weights.pt] --harmful_prompts data/harmful_prompts_test.txt

python main.py --num_prompts 120 --eval_type safe --mode suffix --max_erase 20 --use_classifier --model_wt_path models/[model-weights.pt] --safe_prompts data/safe_prompts_test.txt
```

The accuracy of the DistilBERT classifier on the harmful prompts test set is **100%**, which is the certified accuracy of erase-and-check on this set.

## Attacking the Safety Classifier
`gcg.py` implements the Greedy Coordinate Gradient attack in [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) for the DistilBERT safety classifier. It takes a model and a set of harmful prompts as input and produces adversarial suffixes of the specified length for that model. To generate attacks of a sequence length of 20, run:

```
python gcg.py --num_adv 20 --model_wt_path models/[model-weights.pt] --prompts_file data/harmful_prompts_test.txt
```
We do not include the adversarial prompts in this repository as some of them might bypass the safety guardrails of publicly available LLMs due to their transferability.

To evaluate the safety filter on the adversarial prompts, run:
```
python main.py --num_prompts 120 --eval_type harmful --use_classifier --model_wt_path models/[model-weights.pt] --harmful_prompts [path-to-adversarial-prompts]
```

## RandEC: Randomized Erase-and-Check
The erase-and-check procedure performs an exhaustive search over the set of erased subsequences to check whether an input prompt is harmful or not. Evaluating the safety filter on all erased subsequences is necessary to certify the accuracy of erase-and-check against adversarial prompts. However, in many practical applications, certified guarantees may not be needed. Relaxing this requirement could allow us to reduce the number of filter evaluations and lower the computational cost of erase-and-check while maintaining good detection performance.

The radomized version of the erase-and-check procedure evaluates the safety filter on a randomly sampled subset of the erased subsequences. The fration of the subsequences sampled is controlled by the `--sampling_ratio` parameter.
To evaluate the empirical performance of this version, first generate adversarial prompts with sequence lengths 0, 2, 4, ..., 20. Then run the following command:

```
python main.py --num_prompts 120 --eval_type empirical --mode suffix --max_erase 20 --use_classifier --model_wt_path models/[model-weights.pt] --randomize --sampling_ratio 0.2
```

Following is the empirical performance of RandEC for different values of the sampling ratio on adversarial prompts of different lengths:
<p align="center">
  <img src="results/empirical_suffix_120_clf_rand.png" width="400"/>
</p>

By checking only 20% of the erased subsequences (sampling ratio = 0.2), it achieves an accuracy of over 90%.

## GradEC: Gradient-based Erase-and-Check
We propose another emppirical defense, GradEC, that uses the gradients of the safety filter with respect to the input prompt to optimize the erased tokens to minimize the loss for the harmful class. We use the DistilBERT safety classifier for this optimization. Please see the corresponding section in the paper for more details on the objective function and the overall procedure.

This propcedure is implemented in `grad_ec.py`.
We vary the number of iterations in the optimization to control the strength and running time of the defense.
Below is the empirical performance of GradEC on adversarial prompts of different lengths. Accuracy is above 90% in just 6 iterations.
<p align="center">
  <img src="results/grad_ec_120_clf.png" width="400"/>
</p>

Following is the command for producing the above results:
```
python main.py --num_prompts 120 --eval_type grad_ec --use_classifier --model_wt_path models/[model-weights.pt] --num_iters [# iterations]
```


## Installation
Follow the instructions below to set up the environment for the experiments.

1. Install Anaconda:
    - Download .sh installer file from https://www.anaconda.com/products/distribution
    - Run: 
        ```
        bash Anaconda3-2023.03-Linux-x86_64.sh
        ```
2. Create Conda Environment with Python:
    ```
    conda create -n [env] python=3.10
    ```
3. Activate environment:
    ```
    conda activate [env]
    ```
4. Install PyTorch with CUDA from: https://pytorch.org/
	```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
5. Install transformers from Huggingface:
    ```
    conda install -c huggingface transformers
    ```
6. Install accelerate:
    ```
    conda install -c conda-forge accelerate
    ```
7. Install `scikit-learn` (required for training safety classifiers):
    ```
    conda install -c anaconda scikit-learn
    ```