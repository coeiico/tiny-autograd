# CLAUDE.md

## Commits

- Never commit unless the user explicitly instructs you to. A request to make a change is
  not a request to commit it, and an instruction to commit covers that commit only, not
  later work.
- Every commit message follows `.agents/instructions/commit-messages.md`. Read it before
  writing one.

## Plans

- User requested plans go in `.agents/plans/`.
- You may read plans.
- You may modify plans when requested to do so by a user, or when receiving permission from the user to do so.
- Do not execute plans unless explicitly instructed to.
- Every plan opens with this header: the title, a short paragraph saying what the plan does
  and why, a line recording the date and the commit it was written against
  (`git rev-parse --short HEAD`), and a line naming its log. The commit tells a later reader
  which code the plan's findings, line references and file lists describe.

  ```md
  # Plan — <Plan Name>

  <One short paragraph: what the plan changes, and why.>

  Written <YYYY-MM-DD> against `<short commit hash>`.
  Log: @.agents/plans/<plan-name>.log.md
  ```

### Plan logs

- A plan's log, `<plan-name>.log.md` beside it, records how implementing the plan is going:
  where it stands, what was found, what was decided and what is still open. The plan says
  what to do; the log says what happened.
- **Whenever you read a plan, read its log too**, before acting on the plan. Where they
  disagree, a decision the user made that is recorded in the log overrides the plan. A
  change the log only proposes does not, until the user approves it.
- Create the log when implementation starts, not when the plan is written. When you are
  instructed to execute a plan, you may write its log freely; that permission does not
  extend to the plan itself.
- Update the log at the end of every phase, and before stopping mid-phase: the next
  session, possibly another agent, starts from it.
- A discovery that changes the plan (a wrong assumption, a step that cannot work, a better
  route) goes in the log with a proposed plan change, and the user is asked. Do not edit the
  plan to match what you did.
- Format:

  ```md
  # Log — <Plan Name>

  Plan: @.agents/plans/<plan-name>.md

  ## Status

  <Kept current, overwritten each update: the phase in progress, what is done in it, the
  next step, anything blocking.>

  ## Open questions

  <Live questions for the user. When one is answered, move it to an entry with its answer.>

  ## Entries

  ### <YYYY-MM-DD> — <phase or topic> (`<short commit hash>`)

  <Append-only, newest last: what was done, what was discovered, what was decided and why,
  and where the work departs from the plan.>
  ```

- When a plan is done, rename it to `<plan-name>.done.md` and its log to
  `<plan-name>.done.log.md`, and move both into `.agents/plans/archived/`, in the same
  change (e.g. `foo.md` → `archived/foo.done.md`, `foo.log.md` → `archived/foo.done.log.md`).
  Update the `Log:` and `Plan:` lines to match, and move any other file belonging to the
  plan with it. `.agents/plans/` holds only plans that are still live.

## Inbox

`INBOX.md` at the repository root is the user's own list of ideas, todos and suggestions.
Read it when the user refers to it. Do not act on an item unless the user asks you to, and
change the file only when the user asks you to, for example to add a suggestion or to
remove an item that is done.

## Python Formatting

- Use `./.agents/instructions/python-formatting.md` for local formatting rules.

## General Formatting

- Use `./.agents/instructions/general-formatting.md` for the formatting rules that apply to
  every file, whatever the language.
- Whenever a language allows, use 2 spaces for indentation.
- If there is no semantic difference between single quotes and double quotes then prefer
  single quotes over double quotes.

## Code Design and Project Structure

- Use `./.agents/instructions/code-design.md` for how ML code itself is written: the rules
  that always hold, readable computation, silent failures, data, trustworthy results,
  reproducibility, configuration and testing.
- Use `./.agents/instructions/project-structure.md` for where things go: pipeline stages,
  code versus data versus outputs, entry points, dependencies and variants.

## Project Overview

tiny-autograd is project 1 of 18 in the Latent Space Project: small ML repos that test whether
neural networks encode concepts and strategy in their latent space, and whether we can read
and edit those representations. Everything trains on a single consumer GPU or a laptop.
`README.md` holds this project's goal, plan and experiments. The full plan, conventions and
open questions for all 18 repos live in the handoff doc: https://claude.ai/artifact/Qqw5xX4p5R62jhcNjhgKgz

- `src/tiny_autograd/` - the scalar engine, neural-network classes, the NumPy tensor engine
  (`tensor.py`), data and graph visualisation, plus `utils.py` for seeding, run folders and
  metrics.json.
- `train.py`, `predict.py`, `analyze.py` - separate CLIs (tyro), so analysis runs on saved
  weights without retraining.
- `notebooks/` - marimo notebooks, saved as plain `.py` files.
- `runs/<run-name>/` - outputs: reloadable scalar parameters, `metrics.json` and PNG plots.
  Gitignored.

Every routine command is a just recipe: `just setup`, `just quick` (CPU smoke test, under 5
minutes), `just train`, `just analyze <run>`, `just test`, and `just check` (Ruff and
basedpyright, as in CI). Use uv for everything; no pip, conda or requirements.txt.

## How to Work Here

- Learning is a primary goal. Use the implementation approach and library components
  named in Project-specific scope below. Readable, from-scratch PyTorch is the default;
  comments explain why, not what. Explain unfamiliar mechanisms in the README.
- No training wrappers that hide the loop (no Hugging Face Trainer, no Lightning). Keep
  forward, zero_grad, backward and step visible.
- The goal is evidence, not just working models: baselines, probes and interventions,
  not accuracy alone.
- Models used for activation analysis expose relevant named activations and patch sites
  through `return_cache` / `patch`, supported framework outputs or hooks, as stated in
  Project-specific scope. Do not impose an unused interface on every model.
- Seed the randomness the method uses. Record the resolved config, seeds, code and data
  identity, and environment with each run, following code-design.md.
- Plot titles state the finding, not just the axes.
- Small variations are run names (`runs/wd1.0-seed0`), never new repos.

## Project-specific scope

These requirements take precedence over shared design and structure defaults. They
preserve the shared rules for experimental integrity while specifying this project's
implementation, evaluation protocol and correctness checks.

- Implement the scalar `Value` engine, neural-network classes and SGD from scratch in Python.
  PyTorch is a test reference, not the implementation; NumPy is permitted for the optional
  tensor extension. The framework-reuse default does not apply to the mechanism being learned
  here.
- The scalar exercise is CPU-only. GPU detection, PyTorch model modes, tensor-shaped activation
  caches and the `return_cache` / `patch` interface are not requirements. Use `engine.py`,
  `nn.py`, `data.py` and `viz.py` where they fit instead of forcing a pipeline directory
  hierarchy.
- Test local derivatives and composed graphs, including a value reused on multiple paths,
  against finite differences or PyTorch away from non-differentiable points. Check documented
  gradient accumulation/reset behaviour and a tiny learning task.
- Save scalar parameters and architecture settings in a simple reloadable format; safetensors is
  not required. Analysis covers the computation graph, learning curve and decision boundary.
  Claims about held-out prediction still require a proper split.

## Definition of Done

- just quick runs end to end on CPU in under 5 minutes, as a smoke test.
- PyTorch runs fit on one consumer GPU (8–12 GB VRAM) and support cuda, mps or cpu
  where the method permits. Project-specific scope states CPU-only exceptions.
- Each run saves weights, metrics.json and its plots to runs/<run-name>/.
- This README shows the headline result with plots, compares against baselines, and
  states negative results plainly.
- Tests cover relevant data and shape contracts, a tiny learning check where suitable,
  and the method-specific correctness checks in Project-specific scope.
