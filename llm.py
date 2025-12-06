# Code adapted from https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py.
import json
import os
import re

import anthropic
import backoff
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MAX_OUTPUT_TOKENS = 4096
AVAILABLE_LLMS = [
    # Anthropic models
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI models
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    # OpenRouter models
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # DeepSeek models
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
]

def create_hf_client(repo_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(repo_or_path, use_fast=True)
    # If CUDA is available, prefer bfloat16/float16; otherwise CPU with float32
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(repo_or_path, device_map="auto", torch_dtype=dtype)
    if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        try:
            model.config.pad_token_id = tokenizer.eos_token_id
        except Exception:
            pass
    text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return {"pipeline": text_gen, "tokenizer": tokenizer, "model": model, "repo_id": repo_or_path}


def create_client(model: str):
    """
    Create and return an LLM client based on the specified model.
    Args:
        model (str): The name of the model to use.
    Returns:
        Tuple[Any, str]: A tuple containing the client instance and the client model name.
    """
    if model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )
        return client, client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif 'gpt' in model or model.startswith("o1-") or model.startswith("o3-"):
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model.startswith("deepseek-"):
        print(f"Using OpenAI API with {model}.")
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )
        return client, model
    elif model == "llama3.1-405b":
        print(f"Using OpenAI API with {model}.")
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        )
        return client, model
    elif model.startswith("hf-local:") or model.startswith("hf-local/") or model.startswith("hf/"):
        if model.startswith("hf-local:"):
            repo_or_path = model.split(":", 1)[1]
        else:
            repo_or_path = model.split("/", 1)[1]
        print(f"Using Hugging Face transformers model {repo_or_path} (local if path).")
        # Prefer local cache under HF_HOME
        hf_home = os.getenv("HF_HOME")
        if hf_home and os.path.isdir(hf_home):
            os.environ["HF_HOME"] = hf_home
            # If a relative path like 'hf_models/...' is provided inside container, rewrite to HF_HOME
            norm = repo_or_path.replace("\\", "/")
            if not os.path.isabs(repo_or_path) and norm.startswith("hf_models/"):
                suffix = norm[len("hf_models/"):]
                repo_or_path = os.path.join(hf_home, suffix)
        # Resolve relative local path on host to absolute path if it exists
        if not os.path.isabs(repo_or_path):
            local_candidate = os.path.abspath(repo_or_path)
            if os.path.exists(local_candidate):
                repo_or_path = local_candidate
        client = create_hf_client(repo_or_path)
        return client, model
    else:
        raise ValueError(f"Model {model} not supported.")

# Get N responses from a single message, used for ensembling.
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
        n_responses=1,
):
    if msg_history is None:
        msg_history = []

    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, anthropic.RateLimitError, anthropic.APIStatusError),
    max_time=120,
)
def get_response_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.3,
):
    if msg_history is None:
        msg_history = []

    if "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif model.startswith("gpt-4o-"):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model.startswith("o1-") or model.startswith("o3-"):
        new_msg_history = msg_history + [{"role": "user", "content": system_message + msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                # {"role": "user", "content": system_message},
                *new_msg_history,
            ],
            temperature=1,
            # max_completion_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            # stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["deepseek-chat", "deepseek-coder"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["deepseek-reasoner"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        reasoning_content = response.choices[0].message.reasoning_content
    elif model.startswith("llama3.1-"):
        llama_size = model.split("-")[-1]
        client_model = f"meta-llama/llama-3.1-{llama_size}-instruct"
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=client_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        resoning_content = response.choices[0].message.reasoning_content
    elif model.startswith("hf/") or model.startswith("hf-local:") or model.startswith("hf-local/"):
        # Normalize to use the instantiated HF client regardless of prefix
        # Build prompt from system + history + user
        def _to_text(val):
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                texts = []
                for block in val:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
                    elif hasattr(block, "text"):
                        texts.append(getattr(block, "text"))
                return "\n".join(texts)
            return str(val)
        hist = ""
        for m in (msg_history or []):
            role = m.get("role", "user") if isinstance(m, dict) else "user"
            content_val = m.get("content") if isinstance(m, dict) else m
            hist += f"\n[{role.upper()}]\n{_to_text(content_val)}\n"
        # Build chat prompt using tokenizer chat_template if available
        messages = (
            (msg_history or []) + [{"role": "user", "content": msg}]
        )
        prompt = None
        tokenizer = client["tokenizer"]
        try:
            if getattr(tokenizer, "chat_template", None):
                # Normalize to list of {role, content: string}
                norm_msgs = [{"role": "system", "content": system_message}]
                for m in messages:
                    role = m.get("role", "user")
                    content_val = m.get("content")
                    if isinstance(content_val, list):
                        pieces = []
                        for block in content_val:
                            if isinstance(block, dict) and "text" in block:
                                pieces.append(block["text"])
                            elif hasattr(block, "text"):
                                pieces.append(getattr(block, "text"))
                        content_val = "\n".join(pieces)
                    norm_msgs.append({"role": role, "content": content_val})
                prompt = tokenizer.apply_chat_template(norm_msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = None
        if not prompt:
            # Fallback simple SFT-style prompt
            def _to_text(val):
                if isinstance(val, str):
                    return val
                if isinstance(val, list):
                    texts = []
                    for block in val:
                        if isinstance(block, dict) and "text" in block:
                            texts.append(block["text"])
                        elif hasattr(block, "text"):
                            texts.append(getattr(block, "text"))
                    return "\n".join(texts)
                return str(val)
            hist = ""
            for m in (msg_history or []):
                role = m.get("role", "user") if isinstance(m, dict) else "user"
                content_val = m.get("content") if isinstance(m, dict) else m
                hist += f"\n[{role.upper()}]\n{_to_text(content_val)}\n"
            prompt = f"[SYSTEM]\n{system_message}\n{hist}\n[USER]\n{msg}\n[ASSISTANT]\n"
        # Truncate prompt to model context window
        model_max = getattr(tokenizer, "model_max_length", 2048)
        if not isinstance(model_max, int) or model_max <= 0 or model_max > 32768:
            model_max = 2048
        # Use more room for output for better JSON/toolfulness
        max_new_tokens = min(512, MAX_OUTPUT_TOKENS)
        max_input_tokens = max(768, model_max - max_new_tokens - 32)
        toks = tokenizer(prompt, add_special_tokens=False)
        input_ids = toks["input_ids"]
        if len(input_ids) > max_input_tokens:
            input_ids = input_ids[-max_input_tokens:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        gen = client["pipeline"](
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        )
        full_text = gen[0].get("generated_text", "") if isinstance(gen, list) else str(gen)
        content = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        new_msg_history = (msg_history or []) + [{"role": "user", "content": msg}, {"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        print(f'User: {new_msg_history[-2]["content"]}')
        print(f'Assistant: {new_msg_history[-1]["content"]}')
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    return content, new_msg_history

def extract_json_between_markers(llm_output):
    inside_json_block = False
    json_lines = []
    
    # Split the output into lines and iterate
    for line in llm_output.split('\n'):
        striped_line = line.strip()
        
        # Check for start of JSON code block
        if striped_line.startswith("```json"):
            inside_json_block = True
            continue
        
        # Check for end of code block
        if inside_json_block and striped_line.startswith("```"):
            # We've reached the closing triple backticks.
            inside_json_block = False
            break
        
        # If we're inside the JSON block, collect the lines
        if inside_json_block:
            json_lines.append(line)
    
    # If we never found a JSON code block, fallback to any JSON-like content
    if not json_lines:
        # Fallback: Try a regex that finds any JSON-like object in the text
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
        for candidate in matches:
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Attempt to clean control characters and re-try
                    candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(candidate_clean)
                    except json.JSONDecodeError:
                        continue
        return None

    # Join all lines in the JSON block into a single string
    json_string = "\n".join(json_lines).strip()
    
    # Try to parse the collected JSON lines
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Attempt to remove invalid control characters and re-parse
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None
