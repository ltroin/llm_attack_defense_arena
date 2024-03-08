import csv
import torch
import torch.nn as nn
import tqdm
import json
import argparse
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
)
import gc
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
import google.generativeai as palm
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import openai
from vllm import LLM as vllm
from vllm import SamplingParams
class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class LocalLLM(LLM):
    def __init__(self,
                 model_path,
                 device='cuda',
                 num_gpus=1,
                 max_gpu_memory=None,
                 dtype=torch.float16, #linux must have this
                 load_8bit=False,
                 cpu_offloading=False,
                 revision=None,
                 debug=False,
                 system_message=None
                 ):
        super().__init__()
        if "llama" in model_path:
            self.model_name = "llama-2"
        elif "vicuna" in model_path:
            self.model_name = "vicuna"
        else:
            self.model_name = "unknown"
            raise ValueError("Unknown model name")
        self.model, self.tokenizer = self.create_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        self.model_path = model_path

        # if system_message is None and 'Llama-2' in model_path:
        #     # monkey patch for latest FastChat to use llama2's official system message
        #     self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
        #     "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
        #     "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
        #     "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
        #     "If you don't know the answer to a question, please don't share false information."
        # else:
        #     self.system_message = system_message

    @torch.inference_mode()
    def create_model(self, model_path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=None,
                     dtype=torch.float16,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
        model, tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )

        return model, tokenizer

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    @torch.inference_mode()
    def generate(self, prompt, temperature=0.01, max_tokens=50, repetition_penalty=1.0):
        # conv_temp = get_conversation_template(self.model_path)
        # self.set_system_message(conv_temp)

        # conv_temp.append_message(conv_temp.roles[0], prompt)
        # conv_temp.append_message(conv_temp.roles[1], None)

        # prompt_input = conv_temp.get_prompt()
        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            # temperature=temperature,
            # repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

    @torch.inference_mode()
    def generate_batch(self, prompts, temperature=0.01, max_tokens=30, repetition_penalty=1.0, batch_size=8):
        # prompt_inputs = []
        # for prompt in prompts:
        #     conv_temp = get_conversation_template(self.model_path)
        #     self.set_system_message(conv_temp)

        #     conv_temp.append_message(conv_temp.roles[0], prompt)
        #     conv_temp.append_message(conv_temp.roles[1], None)

        #     prompt_input = conv_temp.get_prompt()
        #     prompt_inputs.append(prompt_input)

        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # temp = []
        # for prompt in prompts:
        #     temp.append(self.create_conv_prompt(prompt))
        # prompts = temp
        input_ids = self.tokenizer(prompts, padding=True).input_ids
        # load the input_ids batch by batch to avoid OOM
        outputs = []
        for i in range(0, len(input_ids), batch_size):
            print("current i of generate batch is ", i)
            batch_input_ids = torch.as_tensor(input_ids[i:i+batch_size]).cuda()
            
            output_ids = self.model.generate(
                batch_input_ids,
                do_sample=False,
                # temperature=temperature,
                # repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(self.tokenizer.batch_decode(
                output_ids.cpu(), skip_special_tokens=True, spaces_between_special_tokens=False))
            del batch_input_ids,output_ids
            gc.collect()
            torch.cuda.empty_cache()
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()
        
        return outputs
    # From GCG paper
    def get_embedding_matrix(self):
        # from llm-attacks
        if isinstance(self.model, GPTJForCausalLM) or isinstance(self.model, GPT2LMHeadModel):
            return self.model.transformer.wte.weight
        elif isinstance(self.model, LlamaForCausalLM):
            return self.model.model.embed_tokens.weight
        elif isinstance(self.model, GPTNeoXForCausalLM):
            return self.model.base_model.embed_in.weight
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")
    # From GCG paper
    def calc_loss(self, model, embeddings, embeddings_attack, embeddings_target, targets):
        full_embeddings = torch.hstack([embeddings, embeddings_attack, embeddings_target])
        logits = model(inputs_embeds=full_embeddings).logits
        loss_slice_start = len(embeddings[0]) + len(embeddings_attack[0])
        loss = nn.CrossEntropyLoss()(logits[0, loss_slice_start - 1 : -1, :], targets)
        return loss, logits[:, loss_slice_start - 4:-1, :]
    # From GCG paper
    def create_one_hot_and_embeddings(self,tokens, embed_weights, model):
        one_hot = torch.zeros(
            tokens.shape[0], embed_weights.shape[0], device=model.device, dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1,
            tokens.unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype),
        )
        embeddings = (one_hot @ embed_weights).unsqueeze(0).data
        return one_hot, embeddings
    
    def create_conv_prompt(self,prompt):
        conv_template = get_conversation_template(
            self.model_name
        )
        conv_template.set_system_message("You are a helpful agent")
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        
        full_prompt = conv_template.get_prompt()
        return full_prompt


    def create_prompt(self,prompt):
        conv_template = get_conversation_template(
            self.model_name
        )
        conv_template.set_system_message("You are a helpful agent")
        # conv_template.set_system_message("Your objective is to determine if the user's request or question is harmful or not. " \
        #         + "Just answer \"harmful\" or \"not harmful\" to each user message. " \
        #         + "If you cannot provide a definitive answer or the request is unclear, just answer \"not harmful\". " \
        #         + "If the request is too short and incomplete, just answer \"not harmful\". " \
        #         + "Do not write anything else. Do not write anything else.")
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        
        full_prompt = conv_template.get_prompt()
        return full_prompt


class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 temperature=0.01,
                 max_tokens=150,
                 ):
        super().__init__()
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI()
        self.client.api_key = os.environ["OPENAI_API_KEY"]

    # def set_system_message(self, conv_temp):
    #     if self.system_message is not None:
    #         conv_temp.set_system_message(self.system_message)
    #DEFAULT VALUES
    def generate(self, prompt,max_tokens=50,temperature=1.0,top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1):
        # conv_temp = get_conversation_template(self.model_path)
        

        # conv_temp.append_message(conv_temp.roles[0], prompt)
        # conv_temp.append_message(conv_temp.roles[1], None)

        # prompt_input = conv_temp.to_openai_api_messages()
        # print(prompt_input)
        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                # messages=self.create_conv_prompt(prompt),
                #case already handled in the calling functions
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=n
            )
            if n > 1:
                output = [choice.message.content for choice in response.choices]
            else:
                output = response.choices[0].message.content
            return output
        except Exception as e:
            # Check if the exception is a 400 Bad Request
            if hasattr(e, 'response') and e.response.status_code == 400:
                print("400 Bad Request error: ", e.response.text)
                # Handle the error as appropriate (e.g., return a default response or log the error)
                raise
            else:
                # Re-raise the exception if it's not a 400 Bad Request
                raise

    def generate_batch(self, prompts,max_tokens=100,temperature=1.0,top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1):
        return [self.generate(prompt,max_tokens=max_tokens,temperature=temperature,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,n=n) for prompt in prompts]


    def create_conv_prompt(self,prompt,system_message=False, system_message_text="You are a good bot"):
        conv_template = get_conversation_template(
            "gpt-3.5-turbo"
        )
        if system_message:
            conv_template.set_system_message(system_message_text)
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        
        full_prompt = conv_template.to_openai_api_messages()
        return full_prompt


    def create_prompt(self,prompt):
        conv_template = get_conversation_template(
            "gpt-3.5-turbo"
        )
        conv_template.set_system_message("You are a helpful agent")
        # conv_template.set_system_message("Your objective is to determine if the user's request or question is harmful or not. " \
        #         + "Just answer \"harmful\" or \"not harmful\" to each user message. " \
        #         + "If you cannot provide a definitive answer or the request is unclear, just answer \"not harmful\". " \
        #         + "If the request is too short and incomplete, just answer \"not harmful\". " \
        #         + "Do not write anything else. Do not write anything else.")
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        
        full_prompt = conv_template.to_openai_api_messages()

        return full_prompt

class LocalVLLM(LLM):
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.95,
                 system_message=None,

                 ):
        super().__init__()
        self.model_path = model_path
        if 'llama' in model_path:
            self.model_name = 'llama-2'
        elif 'vicuna' in model_path:
            self.model_name = 'vicuna'
        elif 'mis' in model_path:
            self.model_name = 'mistral-7b-instruct'
        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization)
        
        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=1.0, max_tokens=150,top_p=1,top_k=-1, repetition_penalty=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1):
        prompts= [prompt]
        if n > 1:
            result=self.generate_batch(prompts,temperature,max_tokens,top_p,top_k,repetition_penalty,frequency_penalty,presence_penalty,n=n)
        else:
            result=self.generate_batch(prompts,temperature,max_tokens,top_p,top_k,repetition_penalty,frequency_penalty,presence_penalty,n=n)[0]
        return result

    def generate_batch(self, prompts, temperature=1.0, max_tokens=150,top_p=1,top_k=-1, repetition_penalty=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1):
        # prompt_inputs = []
        # for prompt in prompts:
        #     conv_temp = get_conversation_template(self.model_path)
        #     self.set_system_message(conv_temp)

        #     conv_temp.append_message(conv_temp.roles[0], prompt)
        #     conv_temp.append_message(conv_temp.roles[1], None)

        #     prompt_input = conv_temp.get_prompt()
        #     prompt_inputs.append(prompt_input)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,top_p=top_p,top_k=top_k,repetition_penalty=repetition_penalty,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,n=n)
        results = self.model.generate(
            prompts, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            # prompt = result.prompt
            for output in result.outputs:  # Iterate over all outputs in each result
                generated_text = output.text
                outputs.append(generated_text)
        # print(outputs)
        return outputs

    def create_conv_prompt(self,prompt):
        conv_template = get_conversation_template(
            self.model_name
        )
        # conv_template.set_system_message("You are a helpful agent")
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        
        full_prompt = conv_template.get_prompt()
        return full_prompt






    # def create_prompt(self,prompt):
    #     conv_template = get_conversation_template(
    #         self.model_name
    #     )
    #     conv_template.set_system_message("Your objective is to determine if the user's request or question is harmful or not. " \
    #             + "Just answer \"harmful\" or \"not harmful\" to each user message. " \
    #             + "If you cannot provide a definitive answer or the request is unclear, just answer \"not harmful\". " \
    #             + "If the request is too short and incomplete, just answer \"not harmful\". " \
    #             + "Do not write anything else. Do not write anything else.")
    #     conv_template.append_message(conv_template.roles[0], prompt)
    #     conv_template.append_message(conv_template.roles[1], None)
        
    #     full_prompt = conv_template.get_prompt()
    #     return full_prompt

class LocalLLM2(LLM):
    def __init__(self,
                 model_path,
                 system_message=None,

                 ):
        super().__init__()
        
        self.model_path = model_path
        if 'llama' in model_path:
            self.model_name = 'llama-2'
        elif 'vicuna' in model_path:
            self.model_name = 'vicuna'

        self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cuda:1").eval()


        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        ) 

        if 'llama-2' in model_path.lower():
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.padding_side = 'left'
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        
        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, do_sample = True, temperature=1.0, max_tokens=150,top_p=1,top_k=50, repetition_penalty=1.0,num_return_sequences=1):
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()} 
        output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens, 
                        do_sample=True,
                        temperature=temperature,
                        # eos_token_id=self.eos_token_ids,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        top_k=top_k,
                        num_return_sequences=num_return_sequences
                    )
        output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(outputs_list)
        return outputs_list[0]

    # def generate_batch(self, prompts, temperature=1.0, max_tokens=150,top_p=1,top_k=-1, repetition_penalty=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1):
    #     # prompt_inputs = []
    #     # for prompt in prompts:
    #     #     conv_temp = get_conversation_template(self.model_path)
    #     #     self.set_system_message(conv_temp)

    #     #     conv_temp.append_message(conv_temp.roles[0], prompt)
    #     #     conv_temp.append_message(conv_temp.roles[1], None)

    #     #     prompt_input = conv_temp.get_prompt()
    #     #     prompt_inputs.append(prompt_input)

    #     sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,top_p=top_p,top_k=top_k,repetition_penalty=repetition_penalty,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,n=n)
    #     results = self.model.generate(
    #         prompts, sampling_params, use_tqdm=False)
    #     outputs = []
    #     for result in results:
    #         # prompt = result.prompt
    #         for output in result.outputs:  # Iterate over all outputs in each result
    #             generated_text = output.text
    #             outputs.append(generated_text)
    #     # print(outputs)
    #     return outputs

    def create_conv_prompt(self,prompt):
        conv_template = get_conversation_template(
            self.model_name
        )
        # conv_template.set_system_message("You are a helpful agent")
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        
        full_prompt = conv_template.get_prompt()
        return full_prompt