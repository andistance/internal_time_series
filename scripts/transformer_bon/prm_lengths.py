"""Token-only length preflight; never load PRM weights or retain input batches."""

from __future__ import annotations

from .prms.registry import get_prm_backend


def create_length_counter(config):
    from transformers import AutoConfig, AutoTokenizer
    from .prm import HFPRMScorer

    backend = get_prm_backend(config.prm_name)
    module = backend.implementation()
    trust = config.prm_trust_remote_code
    qwen = backend.name == "qwen2.5-math-prm-7b"
    kind = "base"
    if qwen:
        try:
            model_config = AutoConfig.from_pretrained(config.prm_model_path, trust_remote_code=trust)
        except Exception:
            if trust:
                raise
            model_config = AutoConfig.from_pretrained(config.prm_model_path, trust_remote_code=True)
            trust = True
        if any(str(name).lower() == "qwen2forprocessrewardmodel"
               for name in (getattr(model_config, "architectures", None) or [])):
            kind, trust = "process_reward", True
    tokenizer = AutoTokenizer.from_pretrained(config.prm_model_path, trust_remote_code=trust)
    cls = HFPRMScorer if qwen else module.scorer_class()
    renderer = object.__new__(cls)
    renderer.tokenizer = tokenizer
    renderer.kind = kind
    renderer.system_prompt = (config.prm_system_prompt if config.prm_system_prompt is not None else
                              "Please reason step by step, and put your final answer within \\boxed{}.")
    if "rlhflow" in backend.name:
        renderer.plus_id = renderer._single_token_id("+")

    def length(value):
        if qwen:
            encoded = tokenizer([renderer._render_input(value)], padding=False, truncation=False,
                                return_tensors="pt")
            return int(encoded["attention_mask"].sum())
        if "rlhflow" in backend.name:
            return len(renderer._prepare_row(value)[0])
        if backend.name == "qwen-2.5-7b-fover-prm-2026":
            conversation = renderer._render_conversation(value)
            # Same template call and compatibility fallback as _prepare_one.
            try:
                ids = tokenizer.apply_chat_template(conversation, return_tensors="pt",
                                                     add_generation_prompt=False)
            except TypeError:
                ids = tokenizer.apply_chat_template(conversation, return_tensors="pt",
                                                     add_generation_prompt=False)
            if getattr(ids, "ndim", None) != 2:
                raise RuntimeError("FoVer chat template must return a [1, seq_len] tensor")
            return int(ids.shape[-1])
        problem = "" if value.get("problem") is None else str(value["problem"])
        if backend.name == "versa-prm":
            steps = value.get("solution_steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError("VersaPRM input has no solution_steps")
            text = problem + " \n\n" + " \n\n\n\n".join(str(step) for step in steps) + " \n\n\n\n"
        else:
            assistant = "" if value.get("assistant") is None else str(value["assistant"])
            if not assistant:
                raise ValueError("Math-Shepherd PRM input has no assistant solution")
            text = problem + " " + assistant
        return len(tokenizer.encode(text))

    return length
