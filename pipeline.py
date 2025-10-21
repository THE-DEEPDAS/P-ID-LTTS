#!/usr/bin/env python
"""End-to-end P&ID digitalisation pipeline for Kaggle notebooks.

Steps:
    1. Read LaTeX/TikZ describing a P&ID diagram from disk.
    2. Use a foundation LLM (Llama 3.1 8B Instruct) to translate TikZ into detailed natural-language
       reconstruction instructions.
    3. Feed the natural-language instructions into AutomaTikZ to regenerate/clean TikZ code.
    4. Optionally compile the regenerated TikZ into a PDF using pdflatex.

The script is intentionally modular so individual stages can be swapped or inspected during
experimentation in Kaggle notebooks.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ---------------------------------------------------------------------------
# Default configuration (modifiable in-code for Colab/Kaggle convenience)
# ---------------------------------------------------------------------------

GATED_REPO_HINT = (
    "Access to this model is restricted on Hugging Face. Request access at {url} "
    "and provide an auth token via USER_PIPELINE_OVERRIDES['hf_token'] or the HF_TOKEN env var."
)


DEFAULT_PIPELINE_SETTINGS = {
    "tex_path": "/kaggle/input/tex-tikz/main.tex",
    "output_dir": "pipeline_artifacts",
    "tikz_to_text_model": "meta-llama/Llama-3.1-8B-Instruct",
    "text_to_tikz_model": "potamides/AutomaTikZ",
    "hf_token": None,
    "job_name": "pid_digitalised",
    "compile": False,
    "keep_intermediates": False,
    "full_precision": False,
    "verbose": True,
}


# Optional user overrides; edit these in Kaggle without touching pipeline internals
USER_PIPELINE_OVERRIDES: Dict[str, Any] = {}


@dataclass
class PipelineSettings:
    tex_path: str = DEFAULT_PIPELINE_SETTINGS["tex_path"]
    output_dir: str = DEFAULT_PIPELINE_SETTINGS["output_dir"]
    tikz_to_text_model: str = DEFAULT_PIPELINE_SETTINGS["tikz_to_text_model"]
    text_to_tikz_model: str = DEFAULT_PIPELINE_SETTINGS["text_to_tikz_model"]
    hf_token: Optional[str] = DEFAULT_PIPELINE_SETTINGS["hf_token"]
    job_name: str = DEFAULT_PIPELINE_SETTINGS["job_name"]
    compile: bool = DEFAULT_PIPELINE_SETTINGS["compile"]
    keep_intermediates: bool = DEFAULT_PIPELINE_SETTINGS["keep_intermediates"]
    full_precision: bool = DEFAULT_PIPELINE_SETTINGS["full_precision"]
    verbose: bool = DEFAULT_PIPELINE_SETTINGS["verbose"]


def build_settings(overrides: Optional[Dict[str, Any]] = None) -> PipelineSettings:
    data = {**DEFAULT_PIPELINE_SETTINGS, **(overrides or {})}
    return PipelineSettings(**data)


def _handle_gated_repo_error(model_name: str, err: Exception) -> None:
    lower_msg = str(err).lower()
    if "gated repo" in lower_msg or "401" in lower_msg or "access to model" in lower_msg:
        auth_url = f"https://huggingface.co/{model_name}"
        hint = GATED_REPO_HINT.format(url=auth_url)
        raise PermissionError(
            f"Authentication required for '{model_name}'. {hint}"
        ) from err
    raise err

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def chunk_text(text: str, max_chars: int) -> List[str]:
    """Split text into overlapping windows to comply with LLM context limits."""

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    step = max_chars - 500  # add a bit of overlap for context continuity
    for start in range(0, len(text), step):
        chunk = text[start : start + max_chars]
        chunks.append(chunk)
    return chunks


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# TikZ -> natural language using Llama 3.1 8B
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class TikzToTextModel:
    def __init__(
        self,
        model_name: str,
        hf_token: Optional[str] = None,
        generation_config: GenerationConfig | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_name = model_name
        self.hf_token = hf_token
        self.generation_config = generation_config or GenerationConfig()

        logging.info("Loading TikZ->text model: %s", model_name)
        tokenizer_kwargs = {"token": hf_token} if hf_token else {}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **tokenizer_kwargs)
        except Exception as err:  # catch gated repo auth issues
            _handle_gated_repo_error(model_name, err)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"token": hf_token} if hf_token else {}

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            model_kwargs["quantization_config"] = bnb_config

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype if not load_in_4bit else None,
                device_map="auto",
                **model_kwargs,
            )
        except Exception as err:
            _handle_gated_repo_error(model_name, err)

    def describe(self, tikz_code: str) -> str:
        all_chunks = chunk_text(tikz_code, max_chars=6000)
        outputs: List[str] = []

        for idx, chunk in enumerate(all_chunks, start=1):
            logging.info("Generating description for chunk %s/%s", idx, len(all_chunks))
            prompt = self._build_prompt(chunk, idx)
            outputs.append(self._generate(prompt))

        final_text = "\n\n".join(textwrap.dedent(out).strip() for out in outputs if out.strip())
        return final_text

    def _build_prompt(self, tikz_chunk: str, ordinal: int) -> str:
        system = (
            "You are a senior plant instrumentation engineer. "
            "Provide meticulous natural-language instructions to recreate a P&ID diagram." \
        )
        user = textwrap.dedent(
            f"""
            TikZ snippet #{ordinal}:
            ```tikz
            {tikz_chunk}
            ```

            Produce an exhaustive numbered procedure plus a separate summary section covering:
            - All equipment, piping, control elements, and instrumentation symbols.
            - Connectivity, flow direction, signal types, valves, and tag identifiers.
            - Relative positioning (left/right/up/down) and approximate proportions.
            - Any loops, redundancies, or safety features.
            """
        ).strip()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_cfg = self.generation_config
        output = self.model.generate(
            **inputs,
            do_sample=gen_cfg.temperature > 0,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            max_new_tokens=gen_cfg.max_new_tokens,
            repetition_penalty=gen_cfg.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()


# ---------------------------------------------------------------------------
# AutomaTikZ natural language -> TikZ regeneration
# ---------------------------------------------------------------------------


class TextToTikzModel:
    def __init__(
        self,
        model_name: str = "potamides/AutomaTikZ",
        hf_token: Optional[str] = None,
        generation_config: GenerationConfig | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_name = model_name
        self.hf_token = hf_token
        self.generation_config = generation_config or GenerationConfig(max_new_tokens=768, temperature=0.2, top_p=0.95)

        logging.info("Loading AutomaTikZ model: %s", model_name)
        tokenizer_kwargs = {"token": hf_token} if hf_token else {}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **tokenizer_kwargs)
        except Exception as err:
            _handle_gated_repo_error(model_name, err)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"token": hf_token} if hf_token else {}
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            model_kwargs["quantization_config"] = bnb_config

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype if not load_in_4bit else None,
                device_map="auto",
                **model_kwargs,
            )
        except Exception as err:
            _handle_gated_repo_error(model_name, err)

    def generate(self, description: str) -> str:
        prompt = self._build_prompt(description)
        return self._generate(prompt)

    def _build_prompt(self, description: str) -> str:
        system = (
            "You are AutomaTikZ, an assistant that converts natural-language instructions into precise TikZ code. "
            "Return only compilable TikZ code without additional commentary."
        )
        user = textwrap.dedent(
            f"""
            Natural-language specification:
            {description}
            """
        ).strip()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_cfg = self.generation_config
        output = self.model.generate(
            **inputs,
            do_sample=gen_cfg.temperature > 0,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            max_new_tokens=gen_cfg.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()


# ---------------------------------------------------------------------------
# Compilation utilities
# ---------------------------------------------------------------------------


TEX_DOCUMENT_TEMPLATE = textwrap.dedent(
    r"""
    \documentclass[tikz,border=5pt]{standalone}
    \usepackage{tikz}
    \usepackage{pgfplots}
    \pgfplotsset{compat=1.18}
    \begin{document}
    % TikZ code generated by AutomaTikZ
    % P&ID digitalisation pipeline output
    {tikz_body}
    \end{document}
    """
)


def compile_tikz(tikz_code: str, output_dir: Path, job_name: str, keep_intermediates: bool = False) -> Path:
    ensure_directory(output_dir)
    tex_path = output_dir / f"{job_name}.tex"
    pdf_path = output_dir / f"{job_name}.pdf"
    document = TEX_DOCUMENT_TEMPLATE.format(tikz_body=tikz_code)
    write_text(tex_path, document)

    logging.info("Compiling TikZ to PDF using pdflatex")
    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={output_dir.as_posix()}",
                tex_path.as_posix(),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as err:
        raise RuntimeError("pdflatex not found. Install TeX Live (apt-get install texlive-full).") from err
    except subprocess.CalledProcessError as err:
        logging.error("pdflatex failed:\n%s", err.stdout.decode("utf-8", errors="ignore"))
        raise
    finally:
        if not keep_intermediates:
            for ext in (".aux", ".log", ".out"):
                candidate = output_dir / f"{job_name}{ext}"
                if candidate.exists():
                    candidate.unlink()

    if not pdf_path.exists():
        raise RuntimeError("Expected PDF not produced by pdflatex")
    logging.info("PDF generated at %s", pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# Orchestration CLI
# ---------------------------------------------------------------------------


def run_pipeline(settings: PipelineSettings) -> None:
    tex_input_path = Path(settings.tex_path or "").resolve()
    if not tex_input_path.exists():
        raise FileNotFoundError(f"TikZ source not found at: {tex_input_path}")

    output_dir = Path(settings.output_dir).resolve()
    ensure_directory(output_dir)

    logging.info("Using TikZ source from %s", tex_input_path)
    tikz_code = tex_input_path.read_text(encoding="utf-8")
    write_text(output_dir / "extracted_tikz.tex", tikz_code)

    hf_token = settings.hf_token or os.environ.get("HF_TOKEN")

    tikz_to_text = TikzToTextModel(
        model_name=settings.tikz_to_text_model,
        hf_token=hf_token,
        load_in_4bit=not settings.full_precision,
    )
    description = tikz_to_text.describe(tikz_code)
    write_text(output_dir / "generated_description.txt", description)

    text_to_tikz = TextToTikzModel(
        model_name=settings.text_to_tikz_model,
        hf_token=hf_token,
        load_in_4bit=not settings.full_precision,
    )
    regenerated_tikz = text_to_tikz.generate(description)
    write_text(output_dir / "regenerated_tikz.tex", regenerated_tikz)

    if settings.compile:
        compile_tikz(
            regenerated_tikz,
            output_dir=output_dir,
            job_name=settings.job_name,
            keep_intermediates=settings.keep_intermediates,
        )

    summary = {
        "tikz_source": tex_input_path.as_posix(),
        "extracted_tikz": (output_dir / "extracted_tikz.tex").as_posix(),
        "description": (output_dir / "generated_description.txt").as_posix(),
        "regenerated_tikz": (output_dir / "regenerated_tikz.tex").as_posix(),
        "compiled_pdf": (output_dir / f"{settings.job_name}.pdf").as_posix() if settings.compile else None,
        "tikz_to_text_model": settings.tikz_to_text_model,
        "text_to_tikz_model": settings.text_to_tikz_model,
    }
    write_text(output_dir / "pipeline_summary.json", json.dumps(summary, indent=2))
    logging.info("Pipeline completed successfully. Summary saved to %s", output_dir / "pipeline_summary.json")


# ---------------------------------------------------------------------------
# Entrypoint wiring
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)8s | %(message)s",
    )


def main() -> None:
    settings = build_settings(USER_PIPELINE_OVERRIDES)
    configure_logging(settings.verbose)
    logging.info("Pipeline starting with job: %s", settings.job_name)
    try:
        run_pipeline(settings)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.exception("Pipeline terminated with an error: %s", exc)
        raise


if __name__ == "__main__":
    main()
