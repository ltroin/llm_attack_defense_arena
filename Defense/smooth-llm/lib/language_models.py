import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import os
from fastchat.model import load_model, get_conversation_template
import gc
import torch
from vllm import LLM as vllm
from vllm import SamplingParams

class LLM:

    """Forward pass through a LLM."""

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

        if 'vicuna' in model_path:
            self.model_name = "vicuna"
        elif 'llama' in model_path:
            self.model_name = "llama-2"
        else:
            raise ValueError("Unknown model name")
        self.conv_template = get_conversation_template(
            self.model_name
        ) 
        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

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

    # @torch.inference_mode()
    # def __call__(self, prompt, temperature=0.1, max_tokens=30, repetition_penalty=1.0):
    #     # conv_temp = get_conversation_template(self.model_path)
    #     # self.set_system_message(conv_temp)

    #     # conv_temp.append_message(conv_temp.roles[0], prompt)
    #     # conv_temp.append_message(conv_temp.roles[1], None)

    #     # prompt_input = conv_temp.get_prompt()
    #     input_ids = self.tokenizer([prompt]).input_ids
    #     output_ids = self.model.generate(
    #         torch.as_tensor(input_ids).cuda(),
    #         do_sample=True,
    #         temperature=temperature,
    #         repetition_penalty=repetition_penalty,
    #         max_new_tokens=max_tokens
    #     )

    #     if self.model.config.is_encoder_decoder:
    #         output_ids = output_ids[0]
    #     else:
    #         output_ids = output_ids[0][len(input_ids[0]):]

    #     return self.tokenizer.decode(
    #         output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        # )

    @torch.inference_mode()
    def __call__(self, batch, temperature=0.01, max_tokens=30, repetition_penalty=1.0, batch_size=2):
        torch.set_grad_enabled(False)
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(batch, padding=True).input_ids
        # load the input_ids batch by batch to avoid OOM
        outputs = []
        # print("checker1",torch.cuda.memory_summary())
        # print("language model py 134",torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = torch.as_tensor(input_ids[i:i+batch_size]).cuda()  # Move a batch to GPU
            output_ids = self.model.generate(
                batch_input_ids,
                do_sample=False,
                # temperature=temperature,
                # repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            # print("language model py 144",torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(self.tokenizer.batch_decode(
                output_ids.cpu(), skip_special_tokens=True, spaces_between_special_tokens=False))
        #     print("language model py 148",torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        # print("language model py 149",torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
            del batch_input_ids, output_ids  # Free up GPU memory
            gc.collect()
            torch.cuda.empty_cache()
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()
        # print("checker2",torch.cuda.memory_summary())
        # print("language model py 154",torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        return outputs

class LocalVLLM():
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.90,
                 system_message=None
                 ):
        super().__init__()
        self.model_path = model_path
        if 'llama' in model_path:
            self.model_name = 'llama-2'
        elif 'vicuna' in model_path:
            self.model_name = 'vicuna'

        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization)
        
    #     if system_message is None and 'Llama-2' in model_path:
    #         # monkey patch for latest FastChat to use llama2's official system message
    #         self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
    #         "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
    #         "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
    #         "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
    #         "If you don't know the answer to a question, please don't share false information."
    #     else:
    #         self.system_message = system_message

    # def set_system_message(self, conv_temp):
    #     if self.system_message is not None:
    #         conv_temp.set_system_message(self.system_message)
        self.conv_template = get_conversation_template(
            self.model_name
        )
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
    
    def __call__(self, batch,temperature=0.1,max_tokens=100,params=None):
        # inputs = []
        # for prompt in batch:
        #     inputs.append(self.create_conv_prompt(prompt))
        if params:
            param,value = params.split(",")
            if param == "temperature":
                value = float(value)
                outputs = self.generate_batch(batch, max_tokens=max_tokens, temperature=value)
            elif param == "top_p":
                value = float(value)
                if value == 0.0:
                    value = 0.001
                outputs = self.generate_batch(batch, max_tokens=max_tokens, top_p=value)
            elif param == "top_k":
                value = float(value)
                outputs = self.generate_batch(batch, max_tokens=max_tokens, top_k=value)
            # elif param=="presence":
            #     value = float(value)
            #     outputs = self.llm.generate_batch(batch, max_tokens=max_new_tokens, presence_penalty=value)
            # elif param=="frequency":
            #     value = float(value)
            #     outputs = self.llm.generate_batch(batch, max_tokens=max_new_tokens, frequency_penalty=value)
        else:
            outputs= self.generate_batch(batch,temperature=temperature,max_tokens=max_tokens)
        return outputs


class OpenAILLM():
    def __init__(self,
                 model_path,
                 temperature=0.01,
                 max_new_tokens=30,
                 ):
        super().__init__()
        self.model_name = "gpt-3.5-turbo" if "gpt-3.5" in model_path else "gpt4"
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_new_tokens
        self.client = openai.OpenAI()
        self.client.api_key = os.environ["OPENAI_API_KEY"]



    # def set_system_message(self, conv_temp):
    #     if self.system_message is not None:
    #         conv_temp.set_system_message(self.system_message)
        self.conv_template = get_conversation_template(
            self.model_path
        )

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

    def __call__(self, batch,max_tokens=100,temperature=1.0,top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1,params=None):
        if params:
            param,value = params.split(",")
            if param == "temperature":
                value = float(value)
                outputs = [self.generate(prompt,max_tokens=max_tokens,temperature=value,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,n=n) for prompt in batch]
            elif param == "top_p":
                value = float(value)
                if value == 0.0:
                    value = 0.001
                outputs = [self.generate(prompt,max_tokens=max_tokens,temperature=temperature,top_p=value,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,n=n) for prompt in batch]
            elif param=="presence":
                value = float(value)
                outputs = [self.generate(prompt,max_tokens=max_tokens,temperature=temperature,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=value,n=n) for prompt in batch]
            elif param=="frequency":
                value = float(value)
                outputs = [self.generate(prompt,max_tokens=max_tokens,temperature=temperature,top_p=top_p,frequency_penalty=value,presence_penalty=presence_penalty,n=n) for prompt in batch]
        else:
            outputs= [self.generate(prompt,max_tokens=max_tokens,temperature=temperature,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,n=n) for prompt in batch]
        return outputs
