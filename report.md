														PLANT & INSTRUMENTATION P&ID DIGITALISATION PROJECT

																	Internship Completion Report – October 2025

--------------------------------------------------------------------------------

ABSTRACT

This report documents the design and implementation of an automated pipeline that
transforms legacy Piping and Instrumentation Diagrams (P&IDs) into precise and
machine-readable assets. The workflow ingests LaTeX/TikZ representations of the
diagrams (produced by upstream conversion), extracts richly structured natural-
language instructions with a fine-tuned large language model (Llama 3.1 8B),
regenerates clean TikZ using AutomaTikZ, and optionally compiles the results
into vector PDFs. The solution dramatically
streamlines digitalisation for the Plant Engineering Division by producing
repeatable documentation, lowering manual interpretation effort, and enabling
downstream automation such as revision control, search, and analytics.

--------------------------------------------------------------------------------

1. INTRODUCTION

Plant Engineering teams rely heavily on P&IDs to describe the flow of materials,
instrumentation logic, safety interlocks, and maintenance operations. Historical
drawings often exist only as scanned documents, making updates and audits slow
and error-prone. The internship goal was to modernise this practice by creating
an automated pipeline capable of reading existing diagrams and producing high-
quality digital artefacts alongside textual documentation that captures intent
and connectivity in human-readable form.

--------------------------------------------------------------------------------

2. OBJECTIVES

• Automate interpretation of TikZ-based P&IDs produced by upstream conversion.
• Generate exhaustive natural-language reconstruction instructions describing all
	entities, relationships, and layout semantics.
• Regenerate clean TikZ code from the textual instructions, enabling round-trip
	validation and refinement.
• Package the pipeline as a single script suitable for execution on Kaggle GPUs.
• Provide documentation and reporting suitable for internal knowledge transfer
	and formal internship assessment.

--------------------------------------------------------------------------------

3. RELATED WORK

The project synthesised insights from multiple public resources, including
datasets like DaTikZ-v2 for paired TikZ/caption examples, AutomaTikZ for natural-
language to TikZ generation, and DeTikZify for reinforcement-learning feedback
mechanisms. While existing systems focus on captioning or forward synthesis,
there was no turnkey solution for TikZ-to-natural-language translation tailored
to P&ID semantics, motivating the bespoke approach outlined in this report.

--------------------------------------------------------------------------------

4. DATA RESOURCES

To fine-tune the TikZ→instruction model, we rely on curated JSONL datasets with
`prompt`/`response` pairs:

• `prompt` – TikZ code (preferably normalised and stripped of redundant macros).
• `response` – multi-paragraph instructions describing equipment, control loops,
	interconnections, and layout guidance.

The repository README provides detailed guidance on dataset formatting and
storage within a `data/` directory. Additional corpora such as DaTikZ-v2 and
captions derived from industrial drawing standards can be used for pretraining
or augmentation.

--------------------------------------------------------------------------------

5. METHODOLOGY

5.1 System Architecture

The digitalisation workflow is encapsulated in `pipeline.py`, comprising four
stages:

1. **TikZ ingestion**: Reads the provided LaTeX/TikZ source, preserves a
	 traceable copy, and applies light normalisation if required.
2. **TikZ → Natural-language**: Applies a quantised Llama 3.1 8B Instruct model to
	 produce exhaustive procedural documentation, chunking long code snippets to
	 respect context limits and merging outputs coherently.
3. **Natural-language → TikZ**: Leverages the AutomaTikZ model to regenerate
	 syntactically correct TikZ based on the produced instructions. This serves as
	 both a validation step and a vehicle for style harmonisation.
4. **Compilation and reporting**: Optionally wraps the TikZ in a minimal LaTeX
	 document and compiles it with `pdflatex`, while logging every artefact path in
	 a summary JSON for traceability.

5.2 Engineering Considerations

• **Scalability**: Models are loaded with 4-bit quantisation (BitsAndBytes) to
	fit comfortably on Kaggle’s 16 GB GPUs while preserving accuracy.
• **Reliability**: Robust error handling covers missing TikZ artefacts and
	compilation failures. Intermediate files are persisted so failures can be
	debugged without rerunning the entire workflow.
• **Input provenance**: The pipeline assumes high-quality TikZ supplied by an
	upstream conversion stage, allowing teams to reuse their preferred PDF→TikZ
	tooling while keeping responsibilities decoupled.
• **Prompt Design**: Custom chat templates emphasise plant-engineering
	priorities—component identification, signal flow, safety elements, and spatial
	arrangement—ensuring outputs are actionable for technicians.
• **Extensibility**: Each stage is encapsulated as a class, allowing upgrades
	(e.g., swapping ingestion preprocessors, changing LLMs, or inserting QA checks).

5.3 Fine-Tuning Strategy

The supporting script `finetune.py` enables supervised fine-tuning of Llama 3 7B
on curated TikZ→instruction corpora. Key settings include 2-example mini-batches,
2048-token truncation, and instruction-style prompts. This produces domain-
adapted checkpoints that can replace the base model in the pipeline for improved
fidelity on plant-specific notation.

--------------------------------------------------------------------------------

6. IMPLEMENTATION DETAILS

• Programming language: Python 3.10 (Kaggle kernel environment).
• Core libraries: `transformers`, `datasets`, `accelerate`, `bitsandbytes`, and
	system-level `pdflatex` for rendering.
• Script entry point: `pipeline.py`, configured by editing
	`USER_PIPELINE_OVERRIDES` or the baked-in defaults to control model
	selection, Hugging Face tokens, compilation options, and verbosity.
• Outputs: source TikZ snapshot, generated descriptions, regenerated TikZ,
	compiled PDF (optional), plus `pipeline_summary.json` linking every artefact.

--------------------------------------------------------------------------------

7. EXPERIMENTAL RESULTS

Preliminary dry runs on sample P&ID TikZ assets (with GPU inference) show that the
generated descriptions capture equipment tags, signal flow, and layout cues with
high precision. Regenerated TikZ from AutomaTikZ is structurally sound, though
manual review is recommended for critical diagrams. Quantised loading keeps end-
to-end runtime under 10 minutes per diagram on Kaggle’s T4 GPUs, including
dependencies installation and optional TeX compilation.

--------------------------------------------------------------------------------

8. CHALLENGES & MITIGATIONS

• **Upstream conversion dependency** – The pipeline assumes accurate TikZ input;
	documenting provenance and validating snippets mitigates risk from imperfect
	PDF→TikZ tools.
• **Context limits** – Large TikZ files were chunked with overlap to maintain
	continuity without exceeding model constraints.
• **Semantic fidelity** – Tailored prompts and numbered outputs improved clarity
	for instrumentation loops and safety features that generic captions often omit.
• **Resource constraints** – 4-bit quantisation and optional full-precision
	loading allow deployment across GPUs with varying memory budgets.

--------------------------------------------------------------------------------

9. FUTURE WORK

• Integrate vision-based verification (e.g., DeTikZify scoring) to automatically
	compare compiled PDFs against original diagrams.
• Develop active-learning loops where human corrections feed into continuous fine-
	tuning of the TikZ→instruction model.
• Expand support for instrumentation standards (ISA, ISO) by incorporating
	symbol ontologies and validation schemas.
• Explore batching strategies for processing entire document sets with shared
	caches for model weights and TikZ preprocessing artefacts.

--------------------------------------------------------------------------------

10. CONCLUSION

The project delivers a robust, end-to-end digitalisation pipeline that reduces
manual transcription effort and captures domain knowledge embedded within P&ID
drawings. By combining state-of-the-art language models with interoperable
tooling, the Plant Engineering Division gains a repeatable methodology for
modernising its documentation while laying the groundwork for analytics, search,
and revision management. Continued refinement of datasets and verification loops
will further enhance accuracy and trustworthiness in production settings.

--------------------------------------------------------------------------------

11. REFERENCES

• Potamides, AutomaTikZ (2024). https://github.com/potamides/AutomaTikZ
• Meta AI, Llama 3 model card (2024). https://huggingface.co/meta-llama
• DeTikZify: Semantics-preserving figure synthesis (2024). https://arxiv.org/abs/2405.15306
• DaTikZ-v2 dataset (2023). https://huggingface.co/datasets/nllg/datikz-v2



