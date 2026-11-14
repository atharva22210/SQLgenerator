# Team workflow for coding agents

These instructions apply throughout this repository. Keep contributions small,
reviewable, and safe for teammates working in parallel.

## Project objective and learning workflow

- Build a model from scratch that generates SQL from a plain-English request
  and the relevant table schema. Learning pre-training and distributed
  training is a primary goal, alongside producing working code.
- Act as an instructor and collaborator for both teammates. Implement code,
  but keep the team involved in decisions that develop intuition about model
  design, data, optimization, evaluation, and distributed execution.
- Because this is a learning project, add comments and docstrings at important
  decision points in new code. Explain the purpose, assumptions, data shapes,
  and tradeoffs behind non-obvious logic; do not merely narrate obvious syntax.
  When behavior could surprise a learner, include a small example or state the
  invariant the code is preserving.
- Work in small milestones. Explain the purpose and important tradeoffs before
  substantial changes; explain the resulting behavior and evidence afterward.
  Do not complete the entire training project in one unattended pass.
- Before experiments, describe the question, hypothesis, baseline, variables,
  metrics, and expected resource use. Give the team an opportunity to predict
  the result and reason about the choices before running the experiment.
- After experiments, compare observations with predictions, explain plausible
  causes and uncertainty, and discuss what the next experiment would test.
  Do not silently change several experimental variables at once.
- Explicitly requested setup and routine implementation may proceed directly,
  with explanations of what was done and why. Do not turn every command or
  line of code into a lesson or require repeated approval for authorized work.
- Start with single-GPU correctness and a small multi-GPU communication test
  before training at scale. Initial infrastructure targets are Modal L4 GPUs
  and Weights & Biases for experiment tracking; these are not evidence that
  training is correct or that multiple GPUs improve throughput.
- Make setup reproducible for both teammates: document Python and dependency
  versions and distinguish local tooling from the remote GPU environment.
  When introducing uv project management, commit the project configuration
  and lockfile, but never the virtual environment.
- Have users enter credentials through a local prompt or provider interface.
  Do not request API keys in chat or embed them in commands, logs, or code.
  Keep credentials separate from shareable setup instructions.

## Understand the task and preserve existing work

- Read relevant code, documentation, and any more specific `AGENTS.md` before
  editing. Inspect `git status`, the active branch, and staged changes.
- Ask the user when requirements or intended behavior are unclear in a way
  that affects correctness, scope, data, or a public interface. Routine choices
  consistent with the task and existing conventions may proceed.
- Preserve existing user changes and staged work. Never discard, overwrite,
  automatically stash, or include unrelated changes in a commit.
- Do not run destructive Git commands, rewrite shared history, or force-push
  without explicit user authorization.

## Stay current with main

Before starting edits for each task, immediately before each commit, and
immediately before each push:

1. Run `git fetch origin`. Use freshly fetched `origin/main`, not a possibly
   stale local `main`, as the shared baseline.
2. Run `git rev-list --count HEAD..origin/main`. Require a result of `0`.
3. If behind, merge `origin/main` into the working branch. Routine conflict-free
   synchronization and its merge commit are authorized by this workflow. Prefer
   merging over rebasing a shared branch.
4. Verify the count again after synchronization and check the resulting diff.
   Run appropriate checks if merging changed application code.

If fetching fails or local changes prevent a safe merge, explain the blocker
and ask for the necessary input or access. Do not bypass the check, discard
work, or claim synchronization succeeded.

Zero behind refers to the last successful fetch. Another teammate can update
GitHub's `main` afterward; repeat the check at each checkpoint above.

## Always ask about merge conflicts

- On any merge, rebase, cherry-pick, or stash conflict, stop dependent work and
  ask the user how to proceed before editing conflict resolutions, choosing
  either side, staging resolutions, or continuing the operation.
- List the conflicted files, explain the competing changes in plain language,
  and offer a recommended resolution for review. Do not assume one side wins,
  even for a seemingly simple conflict.
- After the user provides direction, resolve only the approved conflicts,
  verify no unresolved files or conflict markers remain in the affected files,
  and run relevant checks before completing the operation.

## Use task branches and focused changes

- Work on the user's designated task branch. For new work without a designated
  branch, create a descriptive branch from updated `origin/main`, such as
  `sanika/add-query-validation`. Branch names with slashes are naming conventions,
  not nested branches.
- If currently on `main`, create a task branch before implementing changes.
  Do not commit task changes directly to `main`.
- Keep changes focused on the requested task and follow existing conventions.
  Avoid unrelated refactors, dependency changes, or generated files.
- Never commit credentials, tokens, private configuration, local environments,
  or build artifacts. Use placeholders when documenting configuration.

## Validate and commit

- Review the complete diff and run checks appropriate to the change. For Python
  model changes, run `python -m unittest discover -s tests` in an environment
  with the required dependencies installed. Report skipped tests, missing
  dependencies, and failures honestly.
- For documentation-only changes, inspect the wording and commands and run
  `git diff --check`; application tests are normally unnecessary.
- Stage explicit paths, inspect `git diff --cached`, and ensure only intended
  changes are included. Run `git diff --cached --check` before committing.
- When a commit is requested, use an explicit message to avoid leaving Git
  waiting for an interactive editor. Use an imperative summary such as
  `docs: define team collaboration workflow`, plus a body explaining the reason
  and material behavior changes when helpful.
- Do not amend existing commits unless requested. Report the commit hash,
  validation results, sync status, and whether changes remain local or were pushed.

## Push, review, and merge

- Push only when authorized by the user's task, to the intended remote branch.
  A local commit does not publish changes, and pushing does not merge into main.
- For authorized pull requests, use the task branch as the source and `main`
  as the target. Inspect the entire PR diff, including changes from earlier
  commits on the branch. Explain the problem, resulting behavior, and validation.
- Request teammate review through the team's agreed process. Address feedback
  with additional commits on the same branch. Do not approve your own work on
  behalf of a teammate or bypass required reviews or checks.
- Merge a PR only with explicit authorization and after required reviews,
  checks, and conflicts are resolved. Do not delete shared branches without
  authorization. Begin the next task from the latest main history.
