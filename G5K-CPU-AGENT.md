# G5K-CPU-AGENT.md — tricks for the CPU-brain agent dispatching GPU jobs

Running notes for a Claude Code instance living on a **CPU priority node** (the "brain", G5K.md Plan v3)
that dispatches GPU work to **besteffort** nodes. Append tricks as we learn them — this is a living file.

See also `G5K.md` (topology, oarsub recipes, the three architecture plans).

---

## 1. Poll your own runs. Do not wait to be asked. ⭐

**The trick:** after launching anything long, check on it *periodically and unprompted*. A GPU job on
besteffort can die at any moment and the log will just… stop. If you only look when the user asks,
you burn 20 minutes of their time discovering a job died 18 minutes ago.

Minimum discipline:
- Launch → confirm it actually started (see §3, §4 — a "LAUNCHED" echo proves nothing).
- Then re-check every few minutes: is the process alive, has the log advanced, did the job vanish?
- Report status changes *proactively*, especially deaths.

Two things make this cheap:
- **`/home` is NFS-shared** between the brain and the compute nodes, so you can `tail` the run log
  **directly from the brain** — no `oarsh` needed just to poll.
- A log that has not advanced past the last progress print for longer than the print interval is the
  tell. Compute the expected seconds/step from the first print and use it.

**Detecting death:** the job is gone if `oarstat -u $USER` no longer lists it, or if `oarsh` says:
```
oarsh: Cannot find cpuset file /dev/cpuset//oar/<user>_<jobid>/tasks
```
That message means *preempted / job ended*, not a network problem.

---

## 2. Besteffort gets preempted constantly — design for it

There is **no priority GPU queue** for us at Grenoble (`p3` is CPU-only; that is what the brain runs on).
So **every GPU run is preemptible**. In one afternoon we lost three OLMo-1B runs mid-training.

Rules of thumb:
- Runs under ~2 min usually survive. Anything ≥10 min is a coin flip.
- **Checkpoint at the smallest useful unit.** Write each arm/phase result to disk the moment it
  finishes and skip completed units on restart. A preemption then costs one unit, not the whole run:
  ```python
  ck = Path(f"results/.../_arm_{tag}_{model}_S{seed}_T{steps}.json")
  if ck.exists(): return json.loads(ck.read_text())   # resume
  ...
  ck.write_text(json.dumps(res))                       # checkpoint
  ```
- **Run a supervisor** that relaunches automatically. See `scripts/_supervise_olmo.sh`: every 60s it
  checks whether the process is alive, finds whatever besteffort job is *currently* running (the job id
  and node change after each preemption), and relaunches. Combined with checkpointing it grinds to
  completion unattended.
- **Reuse idle GPU jobs you already hold** rather than reserving more — check `oarstat -u $USER` first;
  we had two jobs sitting on GPUs running nothing but `sleep`.

---

## 3. `oarsh` gotchas

- **`oarsh` needs the job id in the environment**, or it refuses to connect:
  ```bash
  ssh fgrenoble "OAR_JOB_ID=<jobid> oarsh <node> '<command>'"
  ```
- **`oarsh` starts in `$HOME`, not your repo.** A relative path like `scripts/foo.sh` will silently
  resolve against `~` and fail with `No such file or directory` — while your outer `echo LAUNCHED`
  still prints, so it *looks* like it worked. **Always use absolute paths** for both the script and the
  log redirect. (Cost me three relaunches in a row.)
- Plain `ssh` cannot reach a compute node; `gpu=1` jobs do not own the whole node.

---

## 4. Detaching a long run through the double SSH hop

Backgrounding through `ssh → oarsh` hangs the outer call (the harness kills it at timeout, exit 143)
**even though the job started fine**. Use `setsid` + redirect + `</dev/null`:

```bash
ssh fgrenoble "OAR_JOB_ID=$JOB oarsh $NODE \
  'ENV=... setsid bash /abs/path/launcher.sh > /abs/path/run.log 2>&1 < /dev/null & echo LAUNCHED'"
```
Expect the outer command to return 143 / "Terminated" — **that is not a failure**. Verify by reading the
log from the brain (§1). Never conclude "launched" from the echo alone.

---

## 5. Watch the `/home` disk quota — HF caches will eat it

Quota is **24 GB soft / 97 GB hard** (`quota -s` on the frontend). Blowing it fails mid-run with:
```
OSError: [Errno 122] Disk quota exceeded
```
surfacing confusingly as `datasets.exceptions.DatasetGenerationError`.

The trap: **`load_dataset(...)` without `streaming=True` materialises the whole dataset as Arrow.**
`reasoning-core/procedural-pile` = **41 GB**, and it gets **re-generated per builder-hash** — we ended up
with 4 copies across nodes and hit 85 GB.

Fixes:
- Use **`streaming=True`** and keep only what you need, then cache the small subset yourself:
  ```python
  ds = load_dataset(AUX_HF, split="train", streaming=True)   # writes nothing
  # ...filter/cap..., then json.dump a few MB to data_cache/
  ```
  Streaming the pile for 1000 rows × 48 roster tasks scanned only 115k rows in **26 s** — vastly
  cheaper than the 41 GB materialisation it replaced.
- Check `du -sh ~/.cache/huggingface/datasets/*` before blaming your code.
- Small `{prompt,answer}` json caches in `data_cache/` make reruns instant and cost megabytes.

---

## 6. What is *not* reachable from the brain

- **`/mnt/nfs_share_magnet2` is not mounted** on the brain, the Grenoble frontend, or the vercors/kinovis
  GPU nodes. dsileo's frozen `FULLROSTER_*` TaskRow caches and the real `data_cache/` (fwdolci main,
  mmlu-cloze / mbpp eval jsonls) live there, so **the exact Table-2 pipeline cannot be reproduced from
  here.** Plan around it rather than rebuilding those caches (a regen is byte-different and confounds
  any comparison you were trying to make).
- What *is* reproducible exactly, from public HF: **BBH dev/test** (`lukaemon/bbh`, fixed subtask lists,
  `[5:25]`) and **MMLU math/logic cloze** (`cais/mmlu`, prompt `"{q}\nAnswer:"` + gold choice text).
  So you can copy the **eval yardstick** exactly even when you cannot copy the training data — put the
  approximation in the training data, never in the metric, and keep both arms on identical data so the
  contrast stays clean.

---

## 7. Environment notes

- `venv_fastmix/` (in this repo) has torch + transformers + datasets and works on GPU nodes.
  It does **not** have `reasoning_core` — anything importing `task_diagnostics.cache` needs a different env.
- The brain itself has no GPU and lacks `wrapt`/`trl`, so `per_task_influence.py` cannot even be
  imported there. Syntax-check with `python3 -m py_compile` before dispatching; it catches the typos
  that would otherwise cost a full launch cycle.
- Set `HF_HUB_ENABLE_HF_TRANSFER` off / ignore its deprecation warning; `TOKENIZERS_PARALLELISM=false`
  keeps logs readable.
