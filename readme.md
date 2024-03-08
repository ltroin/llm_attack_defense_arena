# A Comprehensive Study of Jailbreak Attack versus Defense for Large Language Models

## Overview

This toolkit is designed for conducting attacks on and applying defenses to various AI models. It supports a range of attack and defense mechanisms, providing a flexible framework for assessing model robustness. The script allows users to specify the model to target, the type of attack or defense to apply, and manages model downloads if necessary.

## Requirements

- Linux
- Python 3.9 or later,Cuda 12.1
- Access to two 40GB memory GPU resources is recommended for certain attack types, especially when using models like vicuna-13b as attackrer for Tap and Pair attacks or mistral for BergeonMethod defense.
- Dependencies listed in `install.ipynb` file. Running every block.
- If you encounter nccl error, make sure you have one nccl version with `pip uninstall nvidia-nccl-cu11, pip uninstall nvidia-nccl-cu12, pip install nvidia-nccl-cu12==2.18.1`.

## Additional Requirements
1. Run the following command in your terminal, replacing <Your key> with your API key. 

```
echo "export OPENAI_API_KEY='<Your key>'" >> ~/.zshrc
echo "export AEGIS_API_KEY='<Your key>'" >> ~/.zshrc
```

2. Update the shell with the new variable:
```
source ~/.zshrc

```
3. Confirm that you have set your environment variable using the following command. 

```
echo $OPENAI_API_KEY
```
## Supported Models

- gpt-3.5-turbo
- llama
- vicuna

## Supported Attacks

- AutoDAN
- GPTFuzz
- DeepInception
- Tap
- Pair
- Jailbroken
- TemplateJailbreak
- Parameters
- GCG

## Supported Defenses

- RALLM
- Baseline
- Aegis
- LLMGuard
- Smooth
- Moderation
- BergeonMethod

## 🚀 Features

- **Various SOTA Attacks and Defenses**: Implements a variety of attack strategies and defense mechanisms tailored for LLMs.
- **Model Compatibility**: Supports multiple LLM architectures, including GPT-3.5 Turbo, LLaMA, and Vicuna.
- **Ease of Use**: Streamlined command-line interface for straightforward experimentation with different models and strategies.
- **Extendability**: Modulely Design facilitates the extension with new methods to this project.


## 🛠 Usage

1. **Download Models**: The script can automatically download the necessary models based on the selected attack or defense. This feature can be toggled off if models are already available locally.

    ```bash
    python main.py --model [Model to be downloaded] --need-download "true"
    ```
    After downloading model, you will find it under `./models/<your model>`

2. **Running an Attack**:
    If you are running Tap or Pair, Please also downlaod `vicuna13`
    
    ```bash
    python main.py --model [Model Name] --mode attack --type [Attack Type] 
    ```
    
    After running the attack, you should see the results in `./Results/<your model>`

3. **Analyze the Attack**:
    This step requires download huggingface model `zhx123/ftrobertallm`.
    ```bash
    python main.py --model [Model Name] --mode process
    ```
    After running the analysis, you should see the results in `./Results/<your model>`
4. **Applying a Defense**:
    If you are running bergeron, Please also download `mistral`
    ```bash
    python main.py --model [Model Name] --mode defense --type [Defense Type]
    ```
    After running the analysis, you should see the results in `./Results/defense/<your model>`

5. **Customizing the Toolkit**: Users can modify `Attacks` and `attack.py` or `Defense` and `defence.py` to add new attack or defense types or to change the default parametors for specific models. The existing ones are the optimal based on the original papers.

## Important Notes

- When running Tap and Pair attacks, vicuna13(Vicuna-13b) is used by default and requires substantial GPU resources.
- The BergeonMethod defense defaults to using the mistral model. Adjustments for model preference should be made in the `defence.py` script.

## Contributing

Contributions to extend the toolkit's capabilities, improve efficiency, or add new features are welcome. Please submit a pull request with a clear description of your changes. Additionally, we provide templates for attack and defense methods for integration with your own method. 

First, modify `attack_template.py` and `defence_template.py`. There are seven steps in total; detailed comments can be found inside the file. 

Secondly, create an attack class and defense class in `attack.py` and `defence.py`, respectively. Demo code is provided in these files. 

Thirdly, add your method to `main.py`.



## 📝 License

This project is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Citation
```
@article{xu2024llm,
  title={LLM Jailbreak Attack versus Defense Techniques--A Comprehensive Study},
  author={Xu, Zihao and Liu, Yi and Deng, Gelei and Li, Yuekang and Picek, Stjepan},
  journal={arXiv preprint arXiv:2402.13457},
  year={2024}
}
```