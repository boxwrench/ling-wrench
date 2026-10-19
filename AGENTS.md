# Ling Wrench for agents

This repository holds Ling quants' documentation, measurements and benchmark scripts.
Weights live on Hugging Face; the runtime is installed separately.

**Installing for a user:** follow [docs/AI_SETUP.md](docs/AI_SETUP.md), steps 0–9.
Check hardware, agree on the file and download size, then install in an isolated environment.
Linux/ROCm is measured; CUDA instructions are untested. Explain that before proceeding.

**Editing this repository:** keep the model pages concise. Put setup and reproduction details in [docs/](docs/INSTALL.md).
The results JSON files are the measurement sources. Never edit rendered marker blocks by hand;
use [tools/fill_results.py](tools/fill_results.py). Checks are listed in [REPRODUCE](docs/REPRODUCE.md#document-checks).
Every new command needs an execution receipt or an explicit “untested” label.
Keep measured facts separate from estimates. Do not include local paths, credentials, personal prompts or internal records.

**Reporting:** send the human the model revision, hardware, runtime version and local changes, exact command/environment,
actual token counts, logs around the failure, and what remains untested.
Use [COMMUNITY_BENCHMARKS](docs/COMMUNITY_BENCHMARKS.md); review logs before sharing them.
