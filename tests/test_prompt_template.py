import pytest

from src.models.prompt_template import (
    SYSTEM_PROMPT,
    format_completion,
    format_prompt,
    format_training_example,
    load_model_config,
)

MODEL_NAMES = ["qwen2.5-1.5b", "llama3.2-1b", "granite-guardian-2b"]

# a distinctive role marker that must appear in each model's rendered prompt
ROLE_MARKER = {
    "qwen2.5-1.5b": "<|im_start|>user",
    "llama3.2-1b": "<|start_header_id|>user<|end_header_id|>",
    "granite-guardian-2b": "<|start_of_role|>user<|end_of_role|>",
}


@pytest.fixture(params=MODEL_NAMES)
def config(request):
    return load_model_config(request.param)


def test_config_loads_all_fields(config):
    assert config.hf_id
    assert config.tokenizer
    assert config.lora_target_modules
    assert config.labels == ("benign", "injection")  # benign first: p_safe = P(labels[0])


def test_prompt_structure(config):
    text = "Summarize the attached quarterly report."
    prompt = format_prompt(config, text)
    t = config.chat_template
    assert prompt.count(SYSTEM_PROMPT) == 1
    assert ROLE_MARKER[config.name] in prompt
    assert text in prompt
    # scoring/generation position: prompt must end exactly at the assistant opener
    assert prompt.endswith(t.assistant_open)
    # roles are ordered system -> user -> assistant
    assert prompt.index(t.system_open) < prompt.index(text) < prompt.rindex(t.assistant_open)


def test_prompt_is_deterministic(config):
    text = "hello world"
    assert format_prompt(config, text) == format_prompt(config, text)


def test_text_with_braces_and_template_markers_survives(config):
    # payload-ish structure (benign content): str.format must not eat braces,
    # and user text containing role markers must be embedded verbatim
    text = '{"a": 1} <|im_start|> {text} <|eot_id|>'
    prompt = format_prompt(config, text)
    assert text in prompt


def test_completion(config):
    t = config.chat_template
    assert format_completion(config, "benign") == "benign" + t.assistant_close
    assert format_completion(config, "injection") == "injection" + t.assistant_close
    with pytest.raises(ValueError):
        format_completion(config, "safe")


def test_training_example_concatenates_cleanly(config):
    prompt, completion = format_training_example(config, "some text", "injection")
    full = prompt + completion
    # the label must directly follow the assistant opener (loss boundary)
    assert config.chat_template.assistant_open + "injection" in full


def test_load_by_path_matches_load_by_name():
    from src.models.prompt_template import CONFIG_DIR

    by_name = load_model_config("qwen2.5-1.5b")
    by_path = load_model_config(CONFIG_DIR / "qwen2.5-1.5b.yaml")
    assert by_name == by_path
