"""Authentic OPI document-embedded prompt-injection record constructor.

This module builds poisoned *documents* exactly the way the vendored
Open-Prompt-Injection (OPI) framework does: by string assembly through its
Attacker classes. No LLM / model is instantiated and no GPU is touched --
``Application.query`` is deliberately avoided because it requires an LLM.

Construction recipe (traced from the OPI source):

* ``create_task(config, num)`` splits a HuggingFace dataset into ``num`` clean
  *target* documents and ``num`` *injected* samples. The clean document is
  ``target_task[idx][0]``.
* The injected instruction is ``injected_task.get_instruction()`` and the
  injected data is ``injected_task[idx][0]``.
* ``create_attacker(strategy, injected_task)`` produces an attacker whose
  ``inject(clean_data, idx[, target_task_name])`` returns the poisoned document
  string -- this is the ``text`` a detector would score.

To make the injected task genuinely *different* from the target task (as a
real cross-task injection), the injected instruction is drawn from a *different*
task config than the target. See ``INJECTION_PAIRS``.

The OPI framework reads ``./data/system_prompts/*.txt`` and writes dataset
caches under ``./data/`` using RELATIVE paths, so ``build`` temporarily
``chdir``s into the OPI repo root and restores the CWD afterwards.
"""

from __future__ import annotations

import os
import random
import sys
import warnings
from pathlib import Path

# Repository root
_CASCADE_ROOT = Path(__file__).resolve().parents[3]
_OPI_ROOT = _CASCADE_ROOT / "data" / "raw" / "Open-Prompt-Injection"
_CONFIG_DIR = _OPI_ROOT / "configs" / "task_configs"

# The 7 document-oriented target tasks (math500 / compromise excluded: not
# document-embedded tasks).
_TARGET_CONFIGS = [
    "sst2",       # sentiment_analysis
    "mrpc",       # duplicate_sentence_detection
    "rte",        # natural_language_inference
    "hsol",       # hate_detection
    "jfleg",      # grammar_correction
    "gigaword",   # summarization
    "sms_spam",   # spam_detection
]

# For each target task (by config name), which config to draw the INJECTED
# instruction/data from. Chosen so the injected task differs from the target
# task, giving an authentic cross-task injection.
INJECTION_PAIRS = {
    "sst2":     "sms_spam",   # sentiment doc  <- inject spam detection
    "mrpc":     "sst2",       # dup-sentence   <- inject sentiment
    "rte":      "hsol",       # NLI            <- inject hate detection
    "hsol":     "sst2",       # hate detection <- inject sentiment
    "jfleg":    "gigaword",   # grammar        <- inject summarization
    "gigaword": "sst2",       # summarization  <- inject sentiment
    "sms_spam": "hsol",       # spam           <- inject hate detection
}

_ATTACKS = ["naive", "escape", "ignore", "fake_comp", "combine"]

# fake_comp / combine only support these target task names.
_FAKE_COMP_SUPPORTED = {
    "sentiment_analysis",
    "spam_detection",
    "hate_detection",
    "summarization",
    "grammar_correction",
    "duplicate_sentence_detection",
    "natural_language_inference",
    "math",
}

# How many clean/injected samples to materialize per task. num*2 must be <= the
# split size; a few hundred comfortably clears every split we use and yields
# num * len(_ATTACKS) candidate records per task.
_DATA_NUM = 300


def _import_opi_factories():
    """Load OPI's ``create_task`` and ``create_attacker`` without running the
    package ``__init__``.

    The real ``OpenPromptInjection/__init__.py`` eagerly imports ``models``,
    which pulls in optional heavy LLM SDKs (google.generativeai, fastchat,
    peft, ...) that are neither installed nor needed for pure string-assembly
    document construction. We register a lightweight placeholder package in
    ``sys.modules`` whose ``__path__`` points at the real source dir, so the
    ``tasks`` and ``attackers`` subpackages import normally via their own
    (models-free) ``__init__`` files, but the top-level ``__init__`` never runs.
    """
    import importlib
    import types

    # `datasets` >= 3 removed `datasets.tasks` (TextClassification etc.). OPI's
    # vendored sms_spam.py imports it at module load time. Provide a minimal
    # placeholder so the import resolves; the templates are metadata-only and
    # unused by our string-assembly path. We must not edit the vendored source.
    if "datasets.tasks" not in sys.modules:
        try:
            import datasets.tasks  # noqa: F401  (present on old versions)
        except Exception:
            shim_tasks = types.ModuleType("datasets.tasks")

            class _TaskTemplateShim:  # noqa: D401 - metadata placeholder
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

            for _name in ("TextClassification", "Summarization",
                          "QuestionAnsweringExtractive",
                          "LanguageModeling", "AutomaticSpeechRecognition"):
                setattr(shim_tasks, _name, _TaskTemplateShim)
            sys.modules["datasets.tasks"] = shim_tasks

    pkg_name = "OpenPromptInjection"
    pkg_dir = _OPI_ROOT / pkg_name

    existing = sys.modules.get(pkg_name)
    if existing is None or getattr(existing, "__opi_shim__", False) is False:
        # Drop a fully-initialized real package if present, replace with shim.
        for mod_name in list(sys.modules):
            if mod_name == pkg_name or mod_name.startswith(pkg_name + "."):
                del sys.modules[mod_name]
        shim = types.ModuleType(pkg_name)
        shim.__path__ = [str(pkg_dir)]
        shim.__opi_shim__ = True
        sys.modules[pkg_name] = shim

    tasks_mod = importlib.import_module(f"{pkg_name}.tasks")
    attackers_mod = importlib.import_module(f"{pkg_name}.attackers")
    return tasks_mod.create_task, attackers_mod.create_attacker


def _load_config(name: str) -> dict:
    import json

    with open(_CONFIG_DIR / f"{name}_config.json") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Modern-HF-hub loader overrides.
#
# OPI's vendored dataset builder scripts (OpenPromptInjection/tasks/*.py) break
# under `datasets` v5: sst2/sms_spam pass the removed `task_templates=` kwarg to
# DatasetInfo, and jfleg/gigaword download from dead Google-Drive/archive URLs
# and generate 0 examples. Rather than editing the vendored package (and without
# downgrading `datasets`), we override the `get_<dataset>` loader that OPI's
# Task base class calls, returning the equivalent modern hub dataset shaped so
# the *existing* `preprocess_func` (see tasks/utils.py) consumes it unchanged.
#
# Field contracts expected by tasks/utils.py:
#   process_sst2 -> dp['sentence'], dp['label']            (0=neg, 1=pos)
#   process_sms_spam -> dp['sms'], dp['label']             (0=ham, 1=spam)
#   process_jfleg -> dp['sentence'], dp['corrections']
#   process_gigaword -> dp['document'], dp['summary']
#
# Only these 4 are overridden; hsol/mrpc/rte already load fine via OPI's own
# loaders and are left untouched.
# --------------------------------------------------------------------------- #

# For summarization we bound the ICL/train slice: the hub gigaword train split
# is ~3.8M rows, but OPI only needs a handful of ICL examples from it. A modest
# slice keeps len() available (needed by OPI's split assertion) without pulling
# the whole corpus.
_GIGAWORD_SLICE = 5000


def _hub_loader_overrides():
    """Return ``{opi_get_func_name: replacement_loader}``.

    Each replacement has signature ``loader(split) -> iterable-of-dicts`` with a
    working ``len()``, matching the OPI ``load_raw_data_func`` contract.
    """
    from datasets import load_dataset

    def get_sst2(split="validation"):
        # stanfordnlp/sst2 already exposes 'sentence' and 'label'.
        return load_dataset("stanfordnlp/sst2", split=split)

    def get_sms_spam(split="train"):
        # ucirvine/sms_spam exposes 'sms' and 'label' (0=ham, 1=spam).
        return load_dataset("ucirvine/sms_spam", split=split)

    def get_jfleg(split="validation"):
        # jhu-clsp/jfleg exposes 'sentence' and 'corrections'.
        return load_dataset("jhu-clsp/jfleg", split=split)

    def get_gigaword(split="validation"):
        # SalmanFaroz/gigaword exposes 'article'/'summary'; OPI's
        # process_gigaword wants 'document'/'summary', so rename 'article'.
        # Bound the (huge) train split used only for ICL sampling.
        load_split = split
        if split == "train":
            load_split = f"train[:{_GIGAWORD_SLICE}]"
        ds = load_dataset("SalmanFaroz/gigaword", split=load_split)
        return ds.rename_column("article", "document")

    return {
        "get_sst2": get_sst2,
        "get_sms_spam": get_sms_spam,
        "get_jfleg": get_jfleg,
        "get_gigaword": get_gigaword,
    }


class _patched_opi_loaders:
    """Temporarily swap OPI's ``get_*`` loaders for modern-hub replacements.

    OPI's ``Task`` base class builds its ``load_raw_data_func`` dispatch dict
    from module-level names (``get_sst2`` etc.) imported into
    ``OpenPromptInjection.tasks.Task``. We patch those names on the ``Task``
    module for the duration of task construction, then restore them.
    """

    def __init__(self, tasks_pkg):
        self._task_mod = sys.modules.get(f"{tasks_pkg}.Task")
        self._overrides = _hub_loader_overrides()
        self._saved = {}

    def __enter__(self):
        if self._task_mod is None:
            return self
        for name, fn in self._overrides.items():
            if hasattr(self._task_mod, name):
                self._saved[name] = getattr(self._task_mod, name)
                setattr(self._task_mod, name, fn)
        return self

    def __exit__(self, *exc):
        if self._task_mod is None:
            return False
        for name, fn in self._saved.items():
            setattr(self._task_mod, name, fn)
        return False


def _normalize(text: str) -> str:
    """Light whitespace normalization that keeps the injection intact.

    Newlines are preserved on purpose: the ``escape`` and ``combine`` attacks
    use ``\\n`` as their distinguishing separator, so collapsing all whitespace
    would make ``escape`` identical to ``naive`` and destroy that attack. We
    only collapse runs of spaces/tabs per line and trim leading/trailing
    whitespace of the whole string.
    """
    lines = str(text).split("\n")
    lines = [" ".join(line.split(" ")).strip() for line in lines]
    # Collapse internal tabs/multiple spaces within each line.
    lines = [" ".join(line.split()) for line in lines]
    return "\n".join(lines).strip()


def _build_for_target(target_cfg_name: str, create_task, create_attacker):
    """Yield (attack, poisoned_text, injected_instruction) for one target task.

    Returns a dict: attack -> list[(text, injected_instruction)].
    Raises on any failure so the caller can skip this task cleanly.
    """
    target_config = _load_config(target_cfg_name)
    injected_cfg_name = INJECTION_PAIRS[target_cfg_name]
    injected_config = _load_config(injected_cfg_name)

    target_task = create_task(target_config, _DATA_NUM, for_injection=False)
    injected_task = create_task(injected_config, _DATA_NUM, for_injection=True)

    target_task_name = target_config["task_info"]["task"]
    injected_instruction = _normalize(injected_task.get_instruction())

    n = min(len(target_task), len(injected_task))

    per_attack: dict[str, list[tuple[str, str]]] = {a: [] for a in _ATTACKS}
    for attack in _ATTACKS:
        attacker = create_attacker(attack, injected_task)
        for idx in range(n):
            clean_data = target_task[idx][0]
            if attack in ("fake_comp", "combine"):
                if target_task_name not in _FAKE_COMP_SUPPORTED:
                    break
                poisoned = attacker.inject(clean_data, idx, target_task_name)
            else:
                poisoned = attacker.inject(clean_data, idx)
            per_attack[attack].append((_normalize(poisoned), injected_instruction))

    return per_attack, target_task_name


def build(cap: int, seed: int = 3131) -> list[dict]:
    """Return up to ``cap`` authentic OPI document-embedded injection records.

    Each record::

        {"text": <poisoned document str>,
         "channel": "document",
         "source": "openpromptinjection",
         "meta": {"target_task": str, "injected_task": str, "attack": str}}

    Deterministic given ``(cap, seed)``. No LLM/model instantiation, no GPU.
    """
    if cap <= 0:
        return []

    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)

    # Records grouped by (target_task, attack) so we can interleave evenly.
    # bucket key -> list[dict]
    buckets: dict[tuple[str, str], list[dict]] = {}
    succeeded: list[str] = []
    skipped: list[str] = []

    try:
        os.chdir(_OPI_ROOT)
        if not (_OPI_ROOT / "data" / "system_prompts").is_dir():
            raise RuntimeError(
                f"OPI system_prompts dir not found under {_OPI_ROOT}; "
                "cannot build authentic OPI records."
            )
        if str(_OPI_ROOT) not in sys.path:
            sys.path.insert(0, str(_OPI_ROOT))

        # Optionally load .env so HF_TOKEN is available for gated datasets.
        env_path = _CASCADE_ROOT / ".env"
        if env_path.is_file():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_path)
            except Exception:  # dotenv optional
                pass

        create_task, create_attacker = _import_opi_factories()

        # Swap OPI's broken vendored loaders for modern-hub equivalents (only
        # affects sst2/sms_spam/jfleg/gigaword; hsol/mrpc/rte untouched).
        loader_patch = _patched_opi_loaders("OpenPromptInjection.tasks")

        for target_cfg_name in _TARGET_CONFIGS:
            try:
                with loader_patch:
                    per_attack, target_task_name = _build_for_target(
                        target_cfg_name, create_task, create_attacker
                    )
            except Exception as exc:  # network/gated/other -> skip this task
                warnings.warn(
                    f"[opi_document] skipping target task '{target_cfg_name}': "
                    f"{type(exc).__name__}: {exc}"
                )
                skipped.append(target_cfg_name)
                continue

            injected_task_name = _load_config(
                INJECTION_PAIRS[target_cfg_name]
            )["task_info"]["task"]

            got_any = False
            for attack, items in per_attack.items():
                if not items:
                    continue
                got_any = True
                key = (target_task_name, attack)
                bucket = buckets.setdefault(key, [])
                for text, _instr in items:
                    bucket.append(
                        {
                            "text": text,
                            "channel": "document",
                            "source": "openpromptinjection",
                            "meta": {
                                "target_task": target_task_name,
                                "injected_task": injected_task_name,
                                "attack": attack,
                            },
                        }
                    )
            if got_any:
                succeeded.append(target_cfg_name)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path

    if succeeded:
        print(
            f"[opi_document] target tasks succeeded: {sorted(succeeded)}; "
            f"skipped: {sorted(skipped)}"
        )

    # Deterministically dedup within each bucket, shuffle each bucket, then
    # round-robin interleave across buckets to balance (target_task, attack).
    rng = random.Random(seed)

    ordered_keys = sorted(buckets.keys())
    prepared: dict[tuple[str, str], list[dict]] = {}
    for key in ordered_keys:
        seen: set[str] = set()
        unique = []
        for rec in buckets[key]:
            if rec["text"] in seen:
                continue
            seen.add(rec["text"])
            unique.append(rec)
        rng.shuffle(unique)
        prepared[key] = unique

    # Round-robin across buckets, with global exact-dedup across the whole set.
    result: list[dict] = []
    global_seen: set[str] = set()
    pointers = {key: 0 for key in ordered_keys}
    active = [key for key in ordered_keys if prepared[key]]

    while active and len(result) < cap:
        next_active = []
        for key in active:
            if len(result) >= cap:
                break
            lst = prepared[key]
            i = pointers[key]
            # advance to next not-yet-globally-seen record
            while i < len(lst) and lst[i]["text"] in global_seen:
                i += 1
            if i < len(lst):
                rec = lst[i]
                global_seen.add(rec["text"])
                result.append(rec)
                pointers[key] = i + 1
                if pointers[key] < len(lst):
                    next_active.append(key)
        active = next_active

    return result[:cap]
