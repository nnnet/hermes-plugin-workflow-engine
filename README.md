# workflow-engine

Generic state-machine orchestrator for multi-turn agent workflows.

Skills (or any caller) declare a workflow as **phases + transitions +
per-phase prompt builders + a decide function**. The engine drives it
deterministically across turns — handling persistence, phase routing,
and mandatory-phrase guards.

The graph shape isn't constrained: cycles (clarification loops), linear
pipelines (draft → review → deploy), branching trees (triage), or DAGs
all work.

## Why this exists

Multi-turn behaviours like vague-desire clarification, risk assessment,
negotiation, or discovery all share the same shape:

1. Decompose input into named slots
2. Loop / branch on missing or contested slots
3. Reach a terminal state with a locked decision

Previously each was re-implemented as a 200-3000 line markdown SKILL.md
body — and LLMs routinely dropped sections, fabricated lock signals, or
never reached EXIT. This engine extracts the deterministic parts into
Python so the LLM only sees a small, focused per-phase prompt.

## What it gives you

- **Generic state machine** (`core/machine.py`) using pytransitions
- **JSON-persisted state container** (`core/state.py`)
- **Common detectors** (`core/detectors.py`) — non-answer, means-as-goal,
  slot extraction from markdown, lock-phrase check, grounding score
- **One CLI dispatch point** (`cli.py`) that takes a `--workflow PATH`

## Each workflow supplies

- A `WorkflowConfig` (slot schema + phases + transitions + decide_fn)
- One prompt builder per phase
- Optional domain-specific detectors

See [`docs/adding-a-workflow.md`](docs/adding-a-workflow.md) for a
walkthrough.

## Quick start (standalone)

```bash
pip install transitions

python3 cli.py \
    --workflow /path/to/my-workflow-dir \
    --session sess-001 \
    --user "user's latest message" \
    --prev-bot "bot's previous reply"
```

`/path/to/my-workflow-dir/` is a Python package — a directory with an
`__init__.py` that exports `WORKFLOW` (a `WorkflowConfig` instance).

Output: JSON with `workflow`, `phase`, `iteration`, `mini_prompt`
(the markdown the bot copies verbatim into its reply), `state_summary`,
`state_file`.

## Quick start (as a Hermes plugin)

```bash
hermes plugins install nnnet/workflow-engine
hermes plugins enable workflow-engine
hermes gateway restart
```

The plugin lives at `~/.hermes/plugins/workflow-engine/`. Skills invoke
the engine via its `cli.py`:

```bash
python3 ~/.hermes/plugins/workflow-engine/cli.py \
    --workflow ~/.hermes/skills/<your-skill>/workflow \
    --session $CHAT_ID --user "$USER_MSG" --prev-bot "$PREV_BOT"
```

## State directory

Defaults to `~/.hermes/workflow_state/<workflow-name>/<session>.json`.
Override with `--state-dir` or `WORKFLOW_STATE_DIR` env var.

## Tests

```bash
cd workflow-engine
pip install pytest transitions
pytest tests/
```

## License

MIT. See [LICENSE](LICENSE).
