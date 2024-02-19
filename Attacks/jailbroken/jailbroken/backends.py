import concurrent.futures
import logging
import os
import pprint
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Generic, Optional, Sequence, TypeVar
from fastchat.model import load_model, get_conversation_template
import anthropic
import openai
import tiktoken
import torch
import jailbroken.settings  # Loads environment variables
from jailbroken.utils import Timeout
from vllm import LLM as vllm
from vllm import SamplingParams

_OPENAI_TOKEN_LIMIT = os.getenv("OPENAI_TOKEN_LIMIT", 8000)
_ANTHROPIC_TOKEN_LIMIT = os.getenv("OPENAI_TOKEN_LIMIT", 8000)
_LOCAL_TOKEN_LIMIT = os.getenv("Local_TOKEN_LIMIT", 8000)
REPEAT_TIME_PER_QUESTION = 5
@dataclass
class Message:
    """Message class for text-based agents/backends."""

    sender: str
    content: str


PromptType = TypeVar("PromptType")


@dataclass
class Request(Generic[PromptType]):
    prompt: PromptType
    stop: Optional[Sequence[str]] = None


class Backend:
    def run(self, request: Request) -> str:
        raise NotImplementedError

    def batch_run(self, requests: list[Request]) -> list[str]:
        return [self.run(request) for request in requests]


class Terminal(Backend):
    def run(self, request: Request[str]) -> str:
        print(f"\n================ PROMPT ================\n{request.prompt}", end="")
        response_text = input()
        return response_text


class OpenAI(Backend):
    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        token_limit_str = _OPENAI_TOKEN_LIMIT
        self.token_limit = int(token_limit_str) if token_limit_str else None

        openai.api_key = os.getenv("OPENAI_API_KEY")

    def _count_tokens(self, prompt: str) -> int:
        """
        Counts the total number of tokens prompted/requested in a text completion request.
        (See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb.)
        """
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(prompt))

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncates a prompt to fit within the token limit."""
        while (
            self.token_limit is not None
            and self._count_tokens(prompt) >= self.token_limit - self.max_tokens
        ):
            prompt = prompt[prompt.find("\n") + 1 :]
        return "...\n" + prompt

    def run(self, request: Request[str]) -> str:
        logging.info(f"Sending request to OpenAI API:\n{pprint.pformat(request)}")
        prompt = self._truncate_prompt(request.prompt)
        with Timeout(180, "OpenAI API timed out!"):
            response = openai.Completion.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                prompt=prompt,
                stop=request.stop,
            )
        response_text = response["choices"][0].text  # type: ignore
        logging.info(f'Received response from OpenAI API: "{response_text}"')
        return response_text


class ChatBackend(Backend):
    def _condense_messages(self, messages: Sequence[Message]) -> Sequence[Message]:
        """Condense messages from the same sender into a single message and strip whitespace."""
        condensed_messages = []
        current_message = ""
        current_sender = ""
        for message in messages:
            assert message.sender != ""
            if message.sender != current_sender:
                if current_message:
                    condensed_message = Message(current_sender, current_message.strip())
                    condensed_messages.append(condensed_message)
                current_message = message.content
                current_sender = message.sender
            else:
                current_message += message.content
        if current_sender != "":
            condensed_message = Message(current_sender, current_message.strip())
            condensed_messages.append(condensed_message)
        return condensed_messages

    def run(self, request: Request[Sequence[Message]]) -> str:
        raise NotImplementedError


class OpenAIChat(ChatBackend):
    _ROLE_LOOKUP = {"system": "system", "environment": "user", "agent": "assistant"}
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # client = openai.OpenAI(api_key="sn-k-sda1")
    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        token_limit_str = _OPENAI_TOKEN_LIMIT
        self.token_limit = int(token_limit_str) if token_limit_str else None

        openai.api_key = ""

    def _count_tokens(self, messages: Sequence[dict]) -> int:
        """
        Counts the total number of tokens prompted/requested in a chat completion request.
        (See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb.)
        """
        encoding = tiktoken.encoding_for_model(self.model)
        # There are 4 metadata tokens per message and 2 prompt tokens to prime the model response
        num_tokens = len(messages) * 4 + 2
        for m in messages:  # Count the message content tokens
            num_tokens += len(encoding.encode(m["content"]))
        return num_tokens

    def _truncate_messages(self, messages: list[dict]) -> Sequence[dict]:
        """Truncates messages to fit within the token limit."""
        while (
            self.token_limit is not None
            and self._count_tokens(messages) >= self.token_limit - self.max_tokens
        ):
            pop_index = 0
            while messages[pop_index]["role"] == "system":
                pop_index += 1
            messages.pop(pop_index)
        return messages

    def run(self, request: Request[Sequence[Message]],from_user=False) -> str:
        messages = []
        for message in self._condense_messages(request.prompt):
            role, content = self._ROLE_LOOKUP[message.sender], message.content
            messages.append({"role": role, "content": content})
        messages = self._truncate_messages(messages)
        # logging.info(
        #     f"Sending request to OpenAI chat completion API:\n"
        #     f"{pprint.pformat(messages, sort_dicts=False)}"
        # )
        if from_user:
            temp = REPEAT_TIME_PER_QUESTION
        else:
            temp = 1
        with Timeout(360, "OpenAI API timed out!"):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
                stop=request.stop,
                n=temp
            )
        if temp > 1:
            response_text = [r.message.content for r in response.choices]  # type: ignore
        else:
            response_text = response.choices[0].message.content  # type: ignore
        logging.info(f'Received response from OpenAI API: "{response_text}"')
        return response_text

    def _send_request(
        self, messages: list[dict], request: Request[Sequence[Message]]
    ) -> str:
        logging.info(
            f"Sending request to OpenAI chat completion API:\n"
            f"{pprint.pformat(messages, sort_dicts=False)}"
        )
        # Retry up to 3 times if the request errors
        for _ in range(3):
            try:
                with Timeout(180, "OpenAI API timed out!"):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=messages,
                        stop=request.stop,
                    )
                response_text = response.choices[0].message.content # type: ignore
                logging.info(f'Received response from OpenAI API: "{response_text}"')
                return response_text
            except Exception as e:
                logging.warning(f"OpenAI API request failed: {e}")
                time.sleep(10)
                continue
        raise RuntimeError("OpenAI API request failed 3 times!")

    def batch_run(self, requests: list[Request[Sequence[Message]]]) -> list[str]:
        list_messages = []
        for request in requests:
            messages = []
            for message in self._condense_messages(request.prompt):
                role, content = self._ROLE_LOOKUP[message.sender], message.content
                messages.append({"role": role, "content": content})
            messages = self._truncate_messages(messages)
            list_messages.append(messages)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            response_texts = executor.map(self._send_request, list_messages, requests)

        response_texts = list(response_texts)
        return response_texts


class TerminalChat(ChatBackend):
    _PREFIX = "  | "

    def run(self, request: Request[Sequence[Message]]) -> str:
        print(f"\n================ CHAT HISTORY ================\n")
        for message in self._condense_messages(request.prompt):
            formatted_sender = f"\033[1m{message.sender.upper()}\033[0m"
            formatted_message = f"{formatted_sender}:\n{message.content}"
            lines = []
            for line in formatted_message.splitlines():
                wrapped_lines = textwrap.fill(line, 100).splitlines()
                if not wrapped_lines:
                    wrapped_lines = [""]
                lines.extend(wrapped_lines)
            formatted_message = ("\n" + self._PREFIX).join(lines)

            print(f"{formatted_message}\n")
        print(f"\033[1mAGENT\033[0m:\n{self._PREFIX}", end="")
        response_text = input()
        return response_text


class RawTerminalChat(ChatBackend):
    """Terminal chat interface for copy/pasting for testing purposes"""

    def __init__(self, use_input: bool = True):
        self._use_input = use_input

    def run(self, request: Request[Sequence[Message]]) -> str:
        print(f"\n================ CHAT HISTORY ================\n")
        for message in self._condense_messages(request.prompt)[-2:]:
            formatted_sender = f"\033[1m{message.sender.upper()}\033[0m"
            formatted_message = f"=== {formatted_sender} ===\n{message.content}"
            print(f"{formatted_message}\n")
        print(f"=== \033[1mAGENT\033[0m ===\n", end="")
        response_text = input() if self._use_input else sys.stdin.read()
        return response_text


class AnthropicChat(ChatBackend):
    _ROLE_LOOKUP = {"system": "system", "environment": "Human", "agent": "Assistant"}

    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        token_limit_str = _ANTHROPIC_TOKEN_LIMIT
        self.token_limit = int(token_limit_str) if token_limit_str else None

        self.client = anthropic.Client(os.getenv("ANTHROPIC_API_KEY", ""))

    def _format(self, messages: Sequence[dict]) -> str:
        """Formats a list of messages for Anthropic API. Ignores system messages."""
        return (
            "".join(
                f"\n\n{m['role']}: {m['content']}"
                for m in messages
                if m["role"] != "system"
            )
            + "\n\nAssistant:"
        )

    def _count_tokens(self, messages: Sequence[dict]) -> int:
        """
        Counts the total number of tokens prompted/requested in a chat completion request.
        (See https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/count_tokens.py.)
        """
        return anthropic.count_tokens(self._format(messages))

    def _truncate_messages(self, messages: list[dict]) -> Sequence[dict]:
        """Truncates messages to fit within the token limit."""
        while (
            self.token_limit is not None
            and self._count_tokens(messages) >= self.token_limit - self.max_tokens
        ):
            pop_index = 0
            messages.pop(pop_index)
        return messages

    def run(self, request: Request[Sequence[Message]], from_user= False) -> str:
        messages = []
        for message in self._condense_messages(request.prompt):
            role, content = self._ROLE_LOOKUP[message.sender], message.content
            messages.append({"role": role, "content": content})
        prompt = self._format(self._truncate_messages(messages))
        logging.info(
            f"Sending request to Anthropic chat completion API:\n"
            f"{pprint.pformat(prompt, sort_dicts=False)}"
        )
        with Timeout(180, "Anthropic API timed out!"):
            stop_sequences = request.stop or []
            response = self.client.completion(
                model=self.model,
                temperature=self.temperature,
                max_tokens_to_sample=self.max_tokens,
                prompt=prompt,
                stop_sequences=stop_sequences,
            )
        response_text = response["completion"]
        logging.info(f'Received response from Anthropic API: "{response_text}"')
        return response_text.strip()


class LocalLLMChat(ChatBackend):
    _ROLE_LOOKUP = {"system": "system", "environment": "user", "agent": "assistant"}

    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 256):
        self.model_path = ""
        if model == 'vicuna':
            self.model_path = '../../../models/lmsys/vicuna-7b-v1.5'
            self.model_name = "vicuna"
        elif model == 'llama':
            self.model_path = '../../../models/meta-llama/Llama-2-7b-chat-hf'
            self.model_name = "llama-2"
        # self.model, self.tokenizer = load_model(
        #     self.model_path,
        #     device='cuda'

        # )
        self.model = vllm(self.model_path, gpu_memory_utilization=0.9,dtype=torch.float16)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # token_limit_str = _LOCAL_TOKEN_LIMIT
        # self.token_limit = int(token_limit_str) if token_limit_str else None


    # def _format(self, messages: Sequence[dict]) -> str:
    #     """Formats a list of messages for Anthropic API. Ignores system messages."""
    #     return (
    #         "".join(
    #             f"\n\n{m['role']}: {m['content']}"
    #             for m in messages
    #             if m["role"] != "system"
    #         )
    #         + "\n\nAssistant:"
    #     )

    # def _count_tokens(self, messages: Sequence[dict]) -> int:
    #     """
    #     Counts the total number of tokens prompted/requested in a chat completion request.
    #     (See https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/count_tokens.py.)
    #     """
    #     return anthropic.count_tokens(self._format(messages))

    # def _truncate_messages(self, messages: list[dict]) -> Sequence[dict]:
    #     """Truncates messages to fit within the token limit."""
    #     while (
    #         self.token_limit is not None
    #         and self._count_tokens(messages) >= self.token_limit - self.max_tokens
    #     ):
    #         pop_index = 0
    #         messages.pop(pop_index)
    #     return messages

    
    def prepare_inputs(self, messages):
        prompt_inputs = []
        # print("the messages are: ", messages)
        for prompt in messages:
            conv_temp = get_conversation_template(self.model_name)
            if prompt['role']== 'system':
                conv_temp.set_system_message(prompt['content'])
                continue
            elif prompt['role']== 'user':
                conv_temp.append_message(conv_temp.roles[0], prompt['content'])
            elif prompt['role']== 'assistant':
                conv_temp.append_message(conv_temp.roles[1], None)
            
            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)
        # print("the prompt_inputs are: ", prompt_inputs)
        return prompt_inputs
    @torch.inference_mode()
    def batch_generate(self, temperature=1.0, max_tokens=256, prompt_inputs=None,n=1):
        sampling_params = SamplingParams(n=n,temperature=temperature, max_tokens=max_tokens)
        # print(full_prompts_list[0])
        results = self.model.generate(
            prompt_inputs, sampling_params, use_tqdm=True)
        outputs = []
        for result in results:
            # prompt = result.prompt
            for output in result.outputs:  # Iterate over all outputs in each result
                generated_text = output.text
                outputs.append(generated_text)
        return outputs
    # @torch.inference_mode()
    # def batch_generate(self, prompt_inputs, temperature=0.01, max_tokens=256, repetition_penalty=1.0, batch_size=16):
    #     if self.tokenizer.pad_token == None:        
    #         self.tokenizer.pad_token = self.tokenizer.eos_token
    #         self.tokenizer.padding_side = "left"
    #     input_ids = self.tokenizer(prompt_inputs, padding=True).input_ids
    #     outputs = []
    #     for i in range(0, len(input_ids), batch_size):
    #         output_ids = self.model.generate(
    #             torch.as_tensor(input_ids[i:i+batch_size]).cuda(),
    #             do_sample=False,
    #             temperature=temperature,
    #             repetition_penalty=repetition_penalty,
    #             max_new_tokens=max_tokens,
    #         )
    #         output_ids = output_ids[:, len(input_ids[0]):]
    #         outputs.extend(self.tokenizer.batch_decode(
    #             output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
    #     return outputs
    
    def run(self, request: Request[Sequence[Message]],from_user=False) -> str:
        messages = []
        for message in self._condense_messages(request.prompt):
            role, content = self._ROLE_LOOKUP[message.sender], message.content
            messages.append({"role": role, "content": content})
        prompt_inputs = self.prepare_inputs(messages)
        # logging.info(
        #     f"Sending request to LocalLLM chat completion API:\n"
        #     f"{pprint.pformat(prompt_inputs, sort_dicts=False)}"
        # )
        if from_user:
            temp = REPEAT_TIME_PER_QUESTION
        else:
            temp = 1
        with Timeout(180, "LocalLLM API timed out!"):
            stop_sequences = request.stop or []
            response = self.batch_generate(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                prompt_inputs=prompt_inputs,
                n=temp
                )

        # load the input_ids batch by batch to avoid OOM
        # print("the response is: ", response)
        if from_user:
            response_text = response
        else:
            response_text = response[0]
        logging.info(f'Received response from LocalLLM API and length is : "{len(response_text)}, is it from user?: {from_user}"')
        return response_text




class DummyChat(ChatBackend):
    def run(self, request: Request[Sequence[Message]]) -> str:
        return "Moo!"
