# Agent Rules

Use simple moves.

Before work:
- Read this file.
- Look in `docs/notes/`.
- Read recent notes if they matter.
- For external Claude/Fable review, run from repo root:
  `claude -p --model claude-fable-5 --effort high --no-session-persistence --tools Read,Grep,Glob,Bash --permission-mode dontAsk "Review only, do not edit. ..."`

Work log:
- Keep notes in `docs/notes/YYYY-MM-DD/*.md`.
- Use date like `2026-07-04`. This sorts good by time.
- Make one note per task, or update one note if task keeps going.
- Use short file names, like `fix-login.md` or `add-tests.md`.

When working on `/goal`:
- Use notes as grounding work log.
- At start, write the high level plan.
- Then write how the plan will be done.
- During work, update note when thinking changes.
- Write what was done.
- Write what remains.
- Write why choices were made.
- Write what failed, and why it failed.
- Write if goal picture changed during work.
- Write risks, questions, and next useful step.
- Goal can outlive many context compactions.
- Next agent should be able to start from notes.
- Notes should show thinking and reasoning, not only final result.

In each note, write:
- What user asked.
- What was done.
- Why it was done.
- Reasoning.
- What worked.
- What did not work.
- Why it did not work.
- Files touched.
- Tests or checks run.
- Next useful step, if any.

Why notes exist:
- Context can get compacted.
- Future agent needs memory.
- Notes are work log that survives chat memory loss.

Style:
- Plain words.
- Short lines.
- No fancy talk.
- Facts over guesses.
- If not sure, say not sure.
