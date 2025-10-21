#!/usr/bin/env python3
"""Resilient P&ID TikZ pipeline tailored for Kaggle notebooks.

Highlights:
    * Mirrors the local notebook workflow without requiring CLI arguments.
    * Downloads Hugging Face models into the working directory with retry logic
      so partial snapshots resume cleanly when CAS service errors occur.
    * Handles gated repositories by surfacing clear guidance on requesting access.
    * Loads large causal language models with 4-bit quantisation when GPUs are
      available, falling back to CPU on out-of-memory events.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # bitsandbytes is optional on CPU-only hardware
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:  # pragma: no cover - Kaggle CPU runtimes may land here
    BitsAndBytesConfig = None  # type: ignore
    BNB_AVAILABLE = False

try:  # huggingface_hub is expected in Kaggle; fall back when symbols move
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError as exc:  # pragma: no cover - install missing dependency via pip
    raise RuntimeError("huggingface_hub package is required. Install it via pip before running.") from exc

try:
    from huggingface_hub import HfHubHTTPError  # type: ignore[attr-defined]
except Exception:
    try:  # Older releases keep it under huggingface_hub.utils
        from huggingface_hub.utils import HfHubHTTPError  # type: ignore
    except Exception:
        class HfHubHTTPError(Exception):
            """Fallback raised when huggingface_hub omits HfHubHTTPError."""

try:
    from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore
except Exception:  # pragma: no cover - older releases reuse HfHubHTTPError
    RepositoryNotFoundError = HfHubHTTPError  # type: ignore


# ---------------------------------------------------------------------------
# Configuration knobs (edit directly inside Kaggle)
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "tikz_path": "/kaggle/input/pidisltts/main.tex",
    "artifacts_dir": "pipeline_artifacts",
    "models_dir": "local_models",
    "tikz_to_text_model": "meta-llama/Llama-3.1-8B-Instruct",
    "text_to_tikz_model": "nllg/detikzify-v2.5-8b",
    "hf_token": os.environ.get("HF_TOKEN"),
    "compile_pdf": False,
    "job_name": "pid_kaggle_run",
    "max_tokens_description": 1024,
    "temperature_description": 0.3,
    "top_p_description": 0.9,
    "repetition_penalty_description": 1.05,
    "max_tokens_regen": 768,
    "temperature_regen": 0.2,
    "top_p_regen": 0.95,
    "repetition_penalty_regen": 1.0,
    "max_download_retry": 4,
}

GATED_REPO_HINT = (
    "Access to this model is gated on Hugging Face. Visit {url} to request access, then set"
    " CONFIG['hf_token'] or the HF_TOKEN environment variable so downloads succeed."
)

PROJECT_ROOT = Path.cwd()
ARTIFACTS_DIR = PROJECT_ROOT / CONFIG["artifacts_dir"]
MODELS_DIR = PROJECT_ROOT / CONFIG["models_dir"]
for directory in (ARTIFACTS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MODEL_CACHE: Dict[str, Path] = {}
RUN_STATE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")


def _safe_repo_dirname(repo_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", repo_id)


def _clean_incomplete_files(target_dir: Path, cache_key: str) -> None:
    partials = list(target_dir.rglob("*.incomplete"))
    if not partials:
        return
    for partial in partials:
        try:
            partial.unlink()
        except OSError as unlink_err:
            logging.warning("[%s] Could not remove partial file %s: %s", cache_key, partial, unlink_err)


def _download_missing_shards(
    repo_id: str,
    cache_key: str,
    target_dir: Path,
    token: Optional[str],
    *,
    max_retry: int,
) -> None:
    index_path = target_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return

    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as err:
        logging.warning("[%s] Unable to parse safetensor index: %s", cache_key, err)
        return

    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        return

    required_files = {Path(filename).name for filename in weight_map.values()}
    missing_files = [name for name in required_files if not (target_dir / name).exists()]
    if not missing_files:
        return

    logging.info("[%s] Detected %s missing shard(s); downloading individually.", cache_key, len(missing_files))
    for filename in missing_files:
        for attempt in range(1, max_retry + 1):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="model",
                    local_dir=target_dir.as_posix(),
                    local_dir_use_symlinks=False,
                    token=token,
                    resume_download=True,
                )
                break
            except HfHubHTTPError as err:
                status_code = getattr(getattr(err, "response", None), "status_code", None)
                if status_code == 401:
                    raise PermissionError(
                        f"Unauthorized when downloading shard '{filename}' from '{repo_id}'."
                    ) from err
                raise
            except RuntimeError as err:
                if "CAS service error" in str(err):
                    logging.warning(
                        "[%s] Shard %s attempt %s/%s hit CAS service error; retrying...",
                        cache_key,
                        filename,
                        attempt,
                        max_retry,
                    )
                    if attempt == max_retry:
                        raise
                    time.sleep(min(30, 5 * attempt))
                else:
                    raise
        else:  # pragma: no cover - defensive safeguard
            raise RuntimeError(f"Failed to download shard '{filename}' after {max_retry} retries.")


def ensure_model_local(repo_id: str, cache_key: str, token: Optional[str] = None, *, max_retry: int = 3) -> Path:
    target_dir = MODELS_DIR / _safe_repo_dirname(repo_id)
    marker_file = target_dir / ".completed"
    if marker_file.exists() and target_dir.exists():
        logging.info("[%s] Using cached weights at %s", cache_key, target_dir)
        MODEL_CACHE[cache_key] = target_dir
        return target_dir

    logging.info("[%s] Downloading snapshot for %s", cache_key, repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retry + 1):
        _clean_incomplete_files(target_dir, cache_key)
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=target_dir.as_posix(),
                local_dir_use_symlinks=False,
                token=token,
                resume_download=True,
            )
            _download_missing_shards(
                repo_id,
                cache_key,
                target_dir,
                token,
                max_retry=max_retry,
            )
            marker_file.touch()
            MODEL_CACHE[cache_key] = target_dir
            logging.info("[%s] Download complete at %s", cache_key, target_dir)
            return target_dir
        except RepositoryNotFoundError as err:
            auth_url = f"https://huggingface.co/{repo_id}"
            hint = GATED_REPO_HINT.format(url=auth_url)
            raise PermissionError(f"Repository '{repo_id}' unavailable. {hint}") from err
        except HfHubHTTPError as err:
            status_code = getattr(getattr(err, "response", None), "status_code", None)
            if status_code == 401:
                raise PermissionError(
                    f"Unauthorized when fetching '{repo_id}'. Provide a valid HF token with access."
                ) from err
            raise
        except RuntimeError as err:
            if "CAS service error" in str(err):
                logging.warning(
                    "[%s] Snapshot attempt %s/%s failed due to CAS service error; retrying soon...",
                    cache_key,
                    attempt,
                    max_retry,
                )
                if attempt == max_retry:
                    logging.error("[%s] Exhausted retries due to persistent CAS service error.", cache_key)
                    raise
                time.sleep(min(30, 5 * attempt))
            else:
                raise

    raise RuntimeError(f"Unable to download '{repo_id}' after {max_retry} attempts.")


@dataclass
class GenerationDefaults:
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float


class SafeCausalLM:
    def __init__(self, label: str, local_dir: Path, generation_defaults: GenerationDefaults) -> None:
        self.label = label
        self.local_dir = Path(local_dir)
        self.generation_defaults = generation_defaults
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._load_model(force_cpu=False)

    def _quant_config(self) -> Optional[Any]:
        if self.device.type != "cuda" or not BNB_AVAILABLE:
            return None
        return BitsAndBytesConfig(  # type: ignore[arg-type]
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    def _load_model(self, force_cpu: bool) -> None:
        target_device = torch.device("cpu") if force_cpu or self.device.type != "cuda" else torch.device("cuda")
        quant_config = None if target_device.type == "cpu" else self._quant_config()
        torch_dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_dir,
                device_map="auto" if target_device.type == "cuda" else None,
                torch_dtype=torch_dtype,
                quantization_config=quant_config,
            )
            self.device = target_device
            logging.info("[%s] Loaded on %s (4-bit=%s)", self.label, self.device, quant_config is not None)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if target_device.type == "cuda" and "out of memory" in str(err).lower():
                self._recover_from_oom(f"{self.label} loading", err)
                self._load_model(force_cpu=True)
            else:
                raise

    def _recover_from_oom(self, stage: str, err: BaseException) -> None:
        logging.warning("[OOM] %s encountered: %s", stage, err)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def generate(self, prompt: str, overrides: Optional[Dict[str, Any]] = None) -> str:
        params = {
            "max_new_tokens": self.generation_defaults.max_new_tokens,
            "temperature": self.generation_defaults.temperature,
            "top_p": self.generation_defaults.top_p,
            "repetition_penalty": self.generation_defaults.repetition_penalty,
        }
        if overrides:
            params.update(overrides)

        while True:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_length = inputs["input_ids"].shape[-1]
                if self.device.type == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        do_sample=params["temperature"] > 0,
                        temperature=params["temperature"],
                        top_p=params["top_p"],
                        max_new_tokens=params["max_new_tokens"],
                        repetition_penalty=params["repetition_penalty"],
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                generated = output[0, input_length:]
                return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                if "out of memory" in str(err).lower() and self.device.type == "cuda":
                    self._recover_from_oom(f"{self.label} inference", err)
                    self.model.to("cpu")
                    self.device = torch.device("cpu")
                    continue
                raise


# ---------------------------------------------------------------------------
# Core pipeline stages
# ---------------------------------------------------------------------------


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 500) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    stride = max(max_chars - overlap, 1)
    return [text[start : start + max_chars] for start in range(0, len(text), stride)]


def build_description_prompt(tikz_snippet: str, ordinal: int) -> str:
    return textwrap.dedent(
        f"""
        <s>[INST]\nYou are a senior plant instrumentation engineer. Provide meticulous natural-language instructions to recreate a process and instrumentation diagram.\n\nTikZ snippet #{ordinal}:\n```tikz\n{tikz_snippet}\n```\n\nInclude:\n1. Numbered reconstruction steps covering all equipment, piping, instrumentation, and safety elements.\n2. Connectivity, flow direction, and signal semantics for each tag.\n3. Relative layout cues (e.g., left/right/up/down) to aid sketching.\n4. A concise summary paragraph at the end.\n[/INST]
        """
    ).strip()


def ensure_tikz_source(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")

    sample_tikz = textwrap.dedent(
        r"""
        \begin{tikzpicture}[>=stealth]
          \draw[thick] (0,0) rectangle (2,2);
          \draw[thick,->] (2,1) -- (3.5,1) node[right]{Process flow};
          \node[draw,circle,minimum size=0.8cm] at (1,1) {P101};
          \node[draw,diamond,minimum size=0.8cm] at (0.5,1.8) {FIC-01};
          \draw[dashed] (0.5,1.8) -- (1,1.4);
        \end{tikzpicture}
        """
    ).strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample_tikz, encoding="utf-8")
    logging.warning("No TikZ input found at %s; created a sample diagram.", path)
    return sample_tikz


def compile_pdf_if_requested(tikz_code: str, job_name: str) -> Optional[Path]:
    if not CONFIG["compile_pdf"]:
        logging.info("PDF compilation skipped (set CONFIG['compile_pdf'] = True to enable).")
        return None

    tex_document = """\\documentclass[tikz,border=5pt]{standalone}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
\\begin{document}
%s
\\end{document}
""" % tikz_code
    tex_path = ARTIFACTS_DIR / f"{job_name}.tex"
    tex_path.write_text(tex_document, encoding="utf-8")
    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={ARTIFACTS_DIR.as_posix()}",
                tex_path.as_posix(),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        logging.warning("pdflatex not installed. Skipping PDF generation.")
        return None
    except subprocess.CalledProcessError as err:
        logging.error("pdflatex failed: %s", err.stdout)
        raise

    pdf_path = ARTIFACTS_DIR / f"{job_name}.pdf"
    if not pdf_path.exists():
        raise RuntimeError("Expected PDF not produced by pdflatex.")
    logging.info("PDF generated at %s", pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    configure_logging()
    logging.info("Starting Kaggle pipeline with job '%s'", CONFIG["job_name"])

    if torch.cuda.is_available():
        logging.info("CUDA available: %s", torch.cuda.get_device_name(0))
    else:
        logging.warning("CUDA device not detected; falling back to CPU execution.")

    tikz_path = Path(CONFIG["tikz_path"])
    tikz_source = ensure_tikz_source(tikz_path)
    RUN_STATE["tikz_path"] = str(tikz_path)
    RUN_STATE["tikz_source"] = tikz_source

    hf_token = CONFIG.get("hf_token") or os.environ.get("HF_TOKEN")
    max_retry = int(CONFIG.get("max_download_retry", 3))

    MODEL_CACHE.clear()
    tikz_to_text_dir = ensure_model_local(
        CONFIG["tikz_to_text_model"],
        "tikz_to_text",
        hf_token,
        max_retry=max_retry,
    )
    text_to_tikz_dir = ensure_model_local(
        CONFIG["text_to_tikz_model"],
        "text_to_tikz",
        hf_token,
        max_retry=max_retry,
    )

    tikz_to_text_runner = SafeCausalLM(
        label="TikZ->Text",
        local_dir=tikz_to_text_dir,
        generation_defaults=GenerationDefaults(
            max_new_tokens=CONFIG["max_tokens_description"],
            temperature=CONFIG["temperature_description"],
            top_p=CONFIG["top_p_description"],
            repetition_penalty=CONFIG["repetition_penalty_description"],
        ),
    )

    snippets = chunk_text(tikz_source)
    logging.info("Processing %s TikZ chunk(s)", len(snippets))
    description_parts: List[str] = []
    for idx, snippet in enumerate(snippets, start=1):
        prompt = build_description_prompt(snippet, idx)
        response = tikz_to_text_runner.generate(prompt)
        description_parts.append(textwrap.dedent(response).strip())
        logging.info("Chunk %s/%s processed", idx, len(snippets))

    description = "\n\n".join(part for part in description_parts if part)
    RUN_STATE["description"] = description
    description_path = ARTIFACTS_DIR / "generated_description.txt"
    description_path.write_text(description, encoding="utf-8")

    text_to_tikz_runner = SafeCausalLM(
        label="Text->TikZ",
        local_dir=text_to_tikz_dir,
        generation_defaults=GenerationDefaults(
            max_new_tokens=CONFIG["max_tokens_regen"],
            temperature=CONFIG["temperature_regen"],
            top_p=CONFIG["top_p_regen"],
            repetition_penalty=CONFIG["repetition_penalty_regen"],
        ),
    )

    regenerated_tikz = text_to_tikz_runner.generate(
        description,
        overrides={
            "max_new_tokens": CONFIG["max_tokens_regen"],
            "temperature": CONFIG["temperature_regen"],
            "top_p": CONFIG["top_p_regen"],
            "repetition_penalty": CONFIG["repetition_penalty_regen"],
        },
    )
    RUN_STATE["regenerated_tikz"] = regenerated_tikz
    regen_path = ARTIFACTS_DIR / "regenerated_tikz.tex"
    regen_path.write_text(regenerated_tikz, encoding="utf-8")

    extracted_path = ARTIFACTS_DIR / "extracted_tikz.tex"
    extracted_path.write_text(tikz_source, encoding="utf-8")

    pdf_path = compile_pdf_if_requested(regenerated_tikz, CONFIG["job_name"])
    if pdf_path is not None:
        RUN_STATE["compiled_pdf"] = str(pdf_path)

    summary = {
        "tikz_source": str(tikz_path),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "description_file": str(description_path),
        "regenerated_tikz_file": str(regen_path),
        "compiled_pdf": RUN_STATE.get("compiled_pdf"),
        "tikz_to_text_model": CONFIG["tikz_to_text_model"],
        "text_to_tikz_model": CONFIG["text_to_tikz_model"],
        "model_cache": {key: str(val) for key, val in MODEL_CACHE.items()},
    }
    summary_path = ARTIFACTS_DIR / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Pipeline completed successfully. Summary saved to %s", summary_path)


if __name__ == "__main__":
    run_pipeline()
