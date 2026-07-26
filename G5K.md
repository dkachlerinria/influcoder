# Connecting to Grid'5000 (G5K)

How to get from this machine to a running shell on a G5K GPU node, and how to check
what's free before reserving. Written for an agent driving this over SSH non-interactively.

## Rules (read before reserving anything)

1. **⚠️⚠️⚠️ ALWAYS REQUEST EXACTLY 1 GPU. NEVER THE WHOLE NODE. ⚠️⚠️⚠️**
   Every `oarsub` **must** use `-l gpu=1,...` and nothing broader (no `-l
   nodes=1`, no whole-node `host=...` selection, no requesting all GPUs on a
   cluster's nodes). This holds even though most G5K GPU nodes have 2+ GPUs —
   grabbing the whole node to get "your" GPU wastes the others and blocks
   anyone else from using them. Never hold more than one active GPU-bearing job
   across all sites at once, either. If `oarsub`/`ssh` output ever *suggests*
   reserving the whole node as a fix for something (e.g. the "not all its CPU
   cores are assigned" error when plain-`ssh`ing to a node — see "Running Claude
   Code" below) — **that is not a valid option here.** The fix is always
   `oarsh`, never a bigger reservation.
2. **Check for existing runs before submitting a new job.** Loop `oarstat -u dkachler`
   across every site (see "Checking your own running jobs" below) *first*. If
   anything is already `Running` or `Waiting` and holds a GPU, **stop and tell the
   user** — don't submit on top of it. Do not assume a job you don't remember
   submitting is stale; it may be a manual session the user started themselves.
3. **When checking for free GPUs to reserve, prioritize >=40GB models** (A100-40GB,
   A100-80GB, L40S, A40, MI210, RTX 8000/6000 Ada, GH200, etc.) over smaller cards
   (RTX 2080 Ti, T4, GTX-series) unless the user says otherwise. See the scan
   snippet further down — sort/filter candidates by `gpu_mem` descending and take
   the first free one.

## Topology

```
local machine --ssh--> access.grid5000.fr (frontend)  --ssh--> <site> frontend (e.g. nancy, grenoble) --ssh--> <compute node>
                        alias: g5k                              e.g. "nancy", "grenoble"                 only reachable while you hold
                                                                                                            a running OAR job on it
```

There is no single-hop path from outside G5K to a compute node. Always go through
the access frontend, then the site frontend, then the node.

## Prerequisites

- G5K account + SSH key already registered (see `~/.ssh/id_ed25519` / `.pub`).
- `~/.ssh/config` has an alias:
  ```
  Host g5k
      User dkachler
      Hostname access.grid5000.fr
      ForwardAgent no

  Host *.g5k
      User dkachler
  ```
  So `ssh g5k` connects straight to the access frontend. From there, plain
  `ssh <site>` works for any of the sites you have access to (no per-site config
  needed on the frontend — it has internal routing set up already).

## Running remote commands non-interactively

`ssh -J g5k <site> 'cmd'` (ProxyJump) does **not** work out of the box here — the
site frontend's host key isn't in the local `known_hosts`, there's no `ssh-askpass`
binary to prompt for host-key TOFU, so it fails with `Host key verification failed`.

Two working patterns instead:

**1. Nested `ssh` invocation** — fine for short commands, but gets ugly fast with
quoting (SQL-like `-p` properties in `oarsub` use single quotes, which collide with
outer shell quoting):
```bash
ssh g5k 'ssh nancy "oarnodes -J"' > out.json 2>err.log
```

**2. Nested heredocs** — the reliable pattern, avoids all quoting problems because
nothing needs escaping between hops:
```bash
ssh g5k <<'OUTER'
ssh grenoble <<'INNER'
oarsub -q besteffort -p "cluster='kinovis'" -l gpu=1,walltime=4:00:00 "sleep 14400"
INNER
OUTER
```
Use this whenever a command contains quotes, e.g. `oarsub -p "..."`.

Every hop prints a MOTD (usage policy links, per-site cluster listing, active
incidents) to stdout before your command's output — expect to grep/filter it out
when parsing results (`grep -v "WARNING\|post-quantum\|..."` or just search for the
JSON `{` start).

## Sites

Known sites (from `curl -s https://api.grid5000.fr/3.0/sites`): `bordeaux`,
`grenoble`, `lille`, `louvain`, `luxembourg`, `lyon`, `nancy`, `nantes`, `rennes`,
`sophia`, `strasbourg`, `toulouse`. `bordeaux` was unreachable (connection timeout)
as of 2026-07-17 — may be down/under deployment, don't assume it's always available.

This account (`dkachler`) is a member of the **`magnet`** group, part of the
**Abaca** federation, with priority queue access on `grenoble`, `nancy`, `rennes`,
`sophia`. Other clusters/sites are generally reachable but may only be usable via
the `besteffort` queue (see below) — the MOTD on each site frontend lists exactly
which clusters are in your priority queue vs besteffort-only.

## Checking your own running jobs before reserving

Before submitting anything, check every site for jobs you already own:
```bash
for site in grenoble nancy rennes sophia lyon lille nantes toulouse strasbourg luxembourg louvain; do
  echo "=== $site ==="
  ssh g5k "ssh $site 'oarstat -u dkachler'" 2>/dev/null | grep -A20 "Job id"
done
```
(`bordeaux` is currently unreachable — skip it or let it fail silently.) This
prints one line per job with its state (`R`unning/`W`aiting) and queue. If
anything shows up, get details with `oarstat -j <id> -f` on that site (gives
`assigned_hostnames`, `wanted_resources`, `type` INTERACTIVE vs PASSIVE, who/when
submitted) before deciding whether it's safe to add a new reservation — see Rule 2.
A job you don't remember submitting may be a manual session the user started
themselves (e.g. via `oarsub -I`); don't kill it without asking.

## Checking GPU availability before reserving

`oarnodes -J` on a site frontend dumps one JSON entry **per core resource**, not
per node or per GPU — a node with an 8-GPU/64-core split will report the same
`gpu_mem`/`gpu_model`/`cluster` on 64 separate entries, 8 cores sharing each
`gpudevice` index (0-7). To know if a *GPU* is free, group by `(host, gpudevice)`,
not by resource id or by host alone — a node can have some GPUs busy and others
idle at the same time.

Relevant fields per resource: `host`, `cluster`, `gpu_count`, `gpu_mem` (MiB),
`gpu_model`, `gpudevice`, `state` (`Alive`/`Dead`/`Suspected`/...), `drain`,
`maintenance`, `resource_id`.

`oarstat -J` on the same frontend dumps all jobs (queued/running/finished-recently).
For jobs with `state` in `Running`/`Launching`/`toLaunch`/`Finishing`, the
`assigned_resources` list gives the core resource ids currently held — map those
back to `(host, gpudevice)` via the `oarnodes` dump to know which GPUs are occupied.
A GPU is free iff no running job's `assigned_resources` covers its resource ids,
its `state` is `Alive`, and `drain`/`maintenance` are `NO`.

Minimal scan (run per site, `nodes.json` = `oarnodes -J` output, `stat.json` =
`oarstat -J` output):
```python
import json
nodes = json.load(open('nodes.json'))
jobs  = json.load(open('stat.json'))

MIN_MEM = 40000  # MiB

host_gpu = {}       # (host) -> {gpudevice: info}
resid_to_gd = {}     # resource_id -> (host, gpudevice)
for rid, r in nodes.items():
    if not r.get('gpu_mem') or int(r.get('gpu_mem')) < MIN_MEM or r.get('gpudevice') is None:
        continue
    h, gd = r['host'], r['gpudevice']
    host_gpu.setdefault(h, {})[gd] = r
    resid_to_gd[str(r['resource_id'])] = (h, gd)

busy = set()
for jid, j in jobs.items():
    if j.get('state') in ('Running', 'Launching', 'toLaunch', 'Finishing'):
        for rid in j.get('assigned_resources') or []:
            key = resid_to_gd.get(str(rid))
            if key:
                busy.add(key)

free_gpus = []
for h, gds in host_gpu.items():
    for gd, r in gds.items():
        down = r['state'] != 'Alive' or r['drain'] == 'YES' or r['maintenance'] == 'YES'
        if (h, gd) not in busy and not down:
            free_gpus.append((int(r['gpu_mem']), h, gd, r['gpu_model']))

# Rule 3: prioritize >=40GB cards — sort by VRAM descending and take the top hit.
for mem, h, gd, model in sorted(free_gpus, reverse=True):
    print(mem, h, gd, model)
```
Fetching `oarnodes -J`/`oarstat -J` on a busy site can be tens of MB of JSON —
redirect to a file rather than parsing inline in the SSH command. Run this per
site and merge results before picking a candidate, since the best free GPU may be
on a different site than the one you first checked.

## Reserving a GPU (`oarsub`)

Basic interactive job on a specific cluster (always `gpu=1` — see Rule 1):
```
oarsub -p "cluster='kinovis'" -l gpu=1,walltime=2:00:00 -I
```

If the account doesn't have priority access to that cluster, `oarsub` refuses with:
```
# You can only access the required resources in besteffort. Reserve with "-q besteffort" if it is what you want.
```
Retry with `-q besteffort -t besteffort`. **Besteffort jobs can be killed at any
time** if a higher-priority job wants those resources — it's not a firm reservation
for the full walltime, just what's available when your account has no priority
queue access there.

For a non-interactive hold (submit, then attach/connect later, e.g. when driving
this over one-shot SSH calls rather than a persistent `-I` session), submit a
passive job that just sleeps for the walltime:
```
oarsub -q besteffort -t besteffort -p "cluster='kinovis'" -l gpu=1,walltime=4:00:00 "sleep 14400"
```
This prints `OAR_JOB_ID=<id>`. Poll status with:
```
oarstat -j <id> -f
```
Look for `state = Running` and `assigned_hostnames` for the granted node(s).

To cancel a job early: `oardel <id>`.

## Connecting to your GPU

Once the job is `Running`, connect from the site frontend to the node your GPU is
on — no extra auth needed, OAR grants access to the job owner automatically for
the job's duration. **Use `oarsh`, not plain `ssh`** (see "Running Claude Code"
below for why: a `gpu=1` job doesn't own every core on the node, so plain `ssh`
can be refused). Plain `ssh` has appeared to work in some earlier tests here, but
don't rely on that — `oarsh` with `OAR_JOB_ID` set is the reliable pattern:
```
ssh g5k <<'OUTER'
ssh grenoble <<'INNER'
OAR_JOB_ID=<job_id> oarsh kinovis-2.grenoble.grid5000.fr 'nvidia-smi'
INNER
OUTER
```

## Running Claude Code (the CLI) on G5K

**Never run `claude` on a site frontend.** Frontends are shared login nodes for
light tasks (submitting jobs, editing files, short scripts) — running the Claude
Code CLI there is banned/throttled to avoid exhausting shared CPU that other users'
interactive sessions depend on. If you try it anyway, expect it to hang rather than
fail with a clear error (this is what happened when it was tried directly on the
Grenoble frontend).

`claude` (and node/npm via nvm) is already installed in `$HOME` on Grenoble — it's
just not usable from the frontend. It only works once you're on a **reserved
compute node**. Since the node's `$HOME` is the same NFS mount as the frontend's,
`claude` is already there once you connect — nothing to install.

**Don't plain-`ssh` to the node.** Since jobs here always request exactly `gpu=1`
(see Rule 1 — never the whole node), you'll get:
```
Cannot connect to node using 'ssh' because not all its CPU cores are assigned to the job which reserves it.
Reserve the whole node, or use 'oarsh' instead.
```
**Ignore the "reserve the whole node" suggestion in that message** — that would
violate Rule 1. The correct fix is always `oarsh` (OAR's job-aware ssh wrapper),
run from the site frontend, with `OAR_JOB_ID` set to your job:
```bash
ssh g5k <<'OUTER'
ssh grenoble <<'INNER'
OAR_JOB_ID=<job_id> oarsh <node> 'bash -c "source ~/.bashrc && claude --version"'
INNER
OUTER
```
Note it's `bash -c "source ~/.bashrc && ..."`, **not** `bash -lc`: a login shell
(`-l`) reads `~/.bash_profile`/`~/.profile`, not `~/.bashrc` — and nvm's installer
only patches `~/.bashrc`. `bash -lc claude` will fail with `command not found`
even though nvm/node/claude are all present. Verified working — this returned
`2.1.212 (Claude Code)` from `kinovis-4.grenoble.grid5000.fr`.

**Two very different modes, verified separately:**

1. **Non-interactive (`--print`), driven by an agent over SSH — verified working
   end-to-end**, including auth/model connectivity:
   ```bash
   ssh g5k <<'OUTER'
   ssh grenoble <<'INNER'
   OAR_JOB_ID=<job_id> oarsh <node> 'bash -c "source ~/.bashrc && claude --print \"your prompt here\""'
   INNER
   OUTER
   ```
   This is what an agent (e.g. this Claude Code session, driving G5K on your
   behalf) should use to actually run Claude Code *on* a GPU node — it doesn't
   need a pty and works through the plain heredoc pattern.

2. **Interactive TUI over a direct SSH `-t` chain, run by a human — cannot be
   driven by an agent this way.** An agent's own shell tool typically has no
   real local tty, so `ssh -t` can't propagate a pseudo-terminal through 3
   hops; `claude` detects this and refuses to launch its TUI, erroring instead
   of hanging. To get the actual interactive session this way, run it yourself
   from a real terminal:
   ```bash
   ssh -t g5k "ssh -t grenoble \"OAR_JOB_ID=<job_id> oarsh -t <node> 'bash -c \\\"source ~/.bashrc && claude\\\"'\""
   ```
   (Untested whether this exact quoting works from a genuine tty.)

3. **Interactive TUI in a detached `tmux` session — CAN be driven by an
   agent, and reconnects to Claude Code web automatically. Verified working
   end-to-end**, including surviving job kill/relaunch. `tmux` itself acts as
   the terminal emulator and allocates a real pty for whatever runs inside it,
   so `claude`'s TUI starts fine — it doesn't matter that the agent launching
   it (via `oarsh`, no tty) has no terminal of its own; `tmux new-session -d`
   just tells the tmux *server* to start something, which doesn't need one
   either.

   Write a small launch script to the shared home dir once (avoids fighting
   quoting through 3 SSH hops):
   ```bash
   ssh g5k <<'OUTER'
   ssh grenoble <<'INNER'
   cat > ~/claude_resume_launch.sh <<'SCRIPT'
   #!/bin/bash
   source ~/.bashrc
   tmux new-session -d -s claudeweb "claude --resume <session_id>"
   SCRIPT
   INNER
   OUTER
   ```
   **Starting a brand-new session instead of resuming — verified working too.**
   Use `"claude"` alone (drop `--resume <session_id>`):
   ```bash
   ssh g5k <<'OUTER'
   ssh grenoble <<'INNER'
   cat > ~/claude_fresh_launch.sh <<'SCRIPT'
   #!/bin/bash
   source ~/.bashrc
   tmux new-session -d -s <tmux_session_name> "claude"
   SCRIPT
   INNER
   OUTER
   ```
   Launch it the same way as above (`oarsh <node> bash ~/claude_fresh_launch.sh`).
   Confirmed: **`/rc active` shows immediately, even on a session that has
   never been opened in Claude Code web before** — remote-control linking is
   not something that needs a prior one-time pairing step; it's on by default
   for sessions started under this account.

   The session gets a `sessionId` internally right away, but Claude Code only
   writes the transcript `.jsonl` file to disk **after the first exchange** —
   so to find a fresh session's ID (e.g. to `--resume` it later), trigger one
   turn first:
   ```bash
   OAR_JOB_ID=<job_id> oarsh <node> 'bash -c "source ~/.bashrc && tmux send-keys -t <tmux_session_name> \"reply with exactly: OK\" Enter"'
   ```
   then find the newest file and read its `sessionId` field:
   ```bash
   OAR_JOB_ID=<job_id> oarsh <node> 'bash -c "ls -t ~/.claude/projects/-home-dkachler/*.jsonl | head -1"'
   ```
   Note **each new `tmux` session re-shows the one-time workspace-trust
   prompt**, even in a directory already trusted by an earlier session — it's
   evidently tracked per-session, not just per-directory. Confirm it the same
   way as before: `tmux send-keys -t <tmux_session_name> Enter`.

   Then, once a job is running, launch it and interact via `oarsh` — no pty
   needed for any of this, plain heredocs work:
   ```bash
   ssh g5k <<'OUTER'
   ssh grenoble <<'INNER'
   OAR_JOB_ID=<job_id> oarsh <node> bash /home/dkachler/claude_resume_launch.sh
   INNER
   OUTER
   ```
   Each new `tmux` session shows the workspace-trust prompt again (see note
   above) — advance it blind with `tmux send-keys -t claudeweb Enter` (same
   `oarsh` pattern). Check what's on screen anytime with:
   ```bash
   OAR_JOB_ID=<job_id> oarsh <node> 'bash -c "source ~/.bashrc && tmux capture-pane -t claudeweb -p"'
   ```
   Look for **`/rc active`** in the bottom-right status bar — that's the
   confirmed indicator that Claude Code web (claude.ai/code) is linked to this
   session. Remote-control linking is on by default (no pairing step needed —
   see above), and it persists: resuming a session ID later (from this same
   site `$HOME`, whether via this `tmux` recipe or run by a human directly)
   reconnects the link automatically. Confirmed: prompts sent from Claude
   Code web reach this process and its replies show up in the web UI in real
   time; confirmed: deleting the OAR job (`oardel <job_id>`) kills the node
   allocation and the `tmux`/`claude` process with it, and the web session
   goes dead immediately.

   To have a *human* attach live to this same tmux session (e.g. to type into
   it directly instead of via Claude Code web), from a real terminal:
   ```bash
   ssh -t g5k "ssh -t grenoble \"OAR_JOB_ID=<job_id> oarsh -t <node> 'tmux attach -t claudeweb'\""
   ```

Note: this setup (nvm + claude) is per-site — done on Grenoble only so far. Other
sites (nancy, rennes, sophia, ...) have separate `$HOME`s and would need the same
install repeated there before `claude` would be available on their nodes.

## Plan: hold-a-node + dispatch (the "reserve once, run many" pattern)

Instead of one OAR job per task (requeueing every time), hold the GPU with a
placeholder job and dispatch tasks onto it at will:

1. **Reserve with a keep-alive command** (always `gpu=1` — Rule 1):
   ```
   oarsub -l gpu=1,walltime=4:00:00 "sleep 14400"
   ```
   The `sleep` just holds the reservation; the GPU is yours for the walltime.
2. **Dispatch tasks** via `oarsh` — no queue, no scheduler, immediate:
   ```
   OAR_JOB_ID=<id> oarsh <node> 'bash ~/experiments/task1/run.sh'
   ```
   Between tasks, edit/copy files into the site `$HOME` (NFS — visible to the
   node instantly), then dispatch again. Iterate freely.
3. **Detach long tasks** so they don't depend on the `oarsh` connection staying
   open: `oarsh <node> 'nohup bash run.sh > log.txt 2>&1 &'` (or run them inside
   `tmux` on the node).

Trade-offs: idle GPU time still counts against usage/karma (fine for an
interactive afternoon, impolite for unattended overnight — use one-shot batch
`oarsub` jobs for that instead); everything dies at walltime; if the holding job
is besteffort, preemption kills all running tasks at once.

## Plan v1: external monitor (controller here, eyes on the node)

While a reservation is held, the local agent (this session) runs a polling loop
that periodically checks on the node and reacts — no human attention needed:

```bash
OAR_JOB_ID=<id> oarsh <node> 'tail -20 ~/experiments/current/log.txt; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

- Poll every 1–2 min via the harness's background/scheduled-wakeup machinery
  (not a foreground `sleep` loop).
- Also poll `oarstat -j <id>` from the frontend: a job that vanished before its
  walltime with tasks unfinished = besteffort preemption → re-scan for a free
  GPU, resubmit, re-dispatch (tasks must be resumable from checkpoints for this
  to be lossless).
- On task crash: read the traceback from the log, fix the script in `$HOME`
  over SSH, re-dispatch via `oarsh`.
- Limitations: visibility is at polling granularity, and the loop lives in this
  local session — laptop off ⇒ monitor off. Robust against *connection* drops
  (tasks run detached on the node), not against the controller disappearing.

## Plan v2: Claude Code worker *on* the GPU node (the pipeline)

A second Claude instance runs on the node itself, doing the tight GPU-local
loop (run → watch → debug → rerun) with results as local files. The local
session acts as supervisor. Builds directly on the verified tmux recipe above.

1. **Prereq (once per site):** nvm + node + `claude` installed in the site
   `$HOME` (done on Grenoble; other sites need the same install first).
2. **Reserve** a node with the sleep-placeholder job (4h hold pattern above).
3. **Launch the worker** on the node, in detached tmux (verified recipe in
   "Running Claude Code" above) — either interactive-TUI-in-tmux (drivable via
   `tmux send-keys` and Claude Code web), or fully headless with a mission:
   ```
   OAR_JOB_ID=<id> oarsh <node> 'bash -c "source ~/.bashrc && nohup claude -p \"Run every experiment in ~/experiments/queue/. For each: execute run.sh, monitor the log, debug and retry on failure (max 2 retries), write outcome to ~/experiments/status/<name>.json\" > ~/worker.log 2>&1 &"'
   ```
4. **Supervisor loop (local session):** poll the status files + `oarstat`;
   on preemption/walltime death, reserve a fresh node and relaunch the worker —
   it picks up from the status/checkpoint files, since the NFS `$HOME` is its
   persistent brain.
5. **Worker design rules:** the worker must treat `$HOME` as its memory — log
   every decision/outcome to files, never hold important state only in its
   context, because it can be killed mid-thought at any moment (walltime or
   besteffort preemption).

Caveats:
- **Credentials:** the worker needs Anthropic auth in the G5K `$HOME` — your
  key sitting on shared research infra. Acceptable but know it's there.
- **Ephemeral:** the worker dies with the OAR job. The supervisor + NFS state
  files provide the continuity, not the worker itself.
- **Frontend ban still applies:** the worker runs only on the reserved node,
  never the site frontend.
- Nodes have outbound internet, so the worker can reach the Anthropic API —
  verified implicitly by the working `--print` and web-linked tmux sessions.

## Plan v3 (VERIFIED): Claude Code brain on a CPU node, disposable GPU jobs

The best architecture of the three: the persistent controller lives *on G5K
itself*, on a cheap **priority-queue CPU node** (guaranteed, no preemption
gamble), and spawns/monitors disposable besteffort GPU jobs. Decouples the
expensive preemptible resource (GPU) from the persistent brain (CPU + NFS).

**Verified 2026-07-19 end-to-end on Grenoble** (test jobs 2939489 / 2939491,
both cleaned up):

1. **CPU-only priority clusters at Grenoble: `chartreuse2/3/4/7`, queue `p3`.**
   The default queue refuses them (`dahu` isn't in the Abaca set — "not enough
   resources"); submit explicitly with `-q p3`. Small request = firm hold:
   ```
   oarsub -q p3 -p "cluster='chartreuse3'" -l core=2,walltime=4:00:00 "sleep 14400"
   ```
   This is NOT besteffort — the walltime is genuinely guaranteed. (GPU-bearing
   p3 clusters exist too — vercors16/17 — but the brain needs no GPU; Rule 1
   spirit: take only the cores you need.)
2. **Node → frontend ssh works passwordlessly, zero setup.** From inside the
   CPU node, `ssh fgrenoble '<cmd>'` just works (same NFS home, keys already
   in place). First contact needs the host-key accept:
   `ssh -o BatchMode=yes -o StrictHostKeyChecking=no fgrenoble ...`.
3. **Submitting jobs from inside a compute node works.** OAR client commands
   don't run on nodes, but the frontend hop makes it a non-issue — verified
   submitting a real besteffort child job from inside chartreuse3-1:
   ```
   ssh -o BatchMode=yes fgrenoble "oarsub -q besteffort -t besteffort -p \"cluster='kinovis'\" -l gpu=1,walltime=2:00:00 'bash ~/experiments/x/run.sh'"
   ```
   (Job submission over ssh is legitimate light frontend use — the frontend
   ban is on running `claude` there, not on issuing oarsub.)
4. **Monitoring the child from the node works.** Queue state via
   `ssh fgrenoble 'oarstat -j <child_id>'`; and since the CPU node and every
   GPU node at the site share the same NFS `$HOME`, the child's logs,
   checkpoints and OAR stdout files are **local files** to the brain —
   `tail -f`, watch checkpoints appear, parse results the moment they're
   written. This is the "live on the GPU" easy-mode, without living on the GPU.

**The full pipeline:**
- Reserve the p3 CPU node (sleep placeholder, e.g. 4h+ walltime).
- Launch `claude` on it in detached tmux (same verified recipe as "Running
  Claude Code" above — chartreuse shares the Grenoble `$HOME` where nvm+claude
  are already installed; `/rc active` gives phone/web access to the brain).
- **`--permission-mode auto` is allowed for this brain** — launch it with
  `claude --permission-mode auto` inside the tmux command instead of bare
  `claude`. Verified 2026-07-20: status bar shows `⏵⏵ auto mode on`, session
  linked to Claude Code web as normal (`session_01JBa9Bhj5K9WejQ5WcqZjY1`).
  Since the brain must grind unattended for hours and drive its own GPU
  submissions/debugging without a human approving each step, auto mode is the
  point of v3, not a shortcut around it — manual/default permission mode
  defeats the "grind while you're away" premise.
- The brain's mission: submit besteffort GPU jobs via the frontend hop, watch
  their logs on NFS, debug failures, detect preemption (`oarstat` shows the
  job gone before walltime with work unfinished) and resubmit.
- GPU tasks must checkpoint to NFS and resume from checkpoints, so preemption
  costs at most the last checkpoint interval.

**Caveats:**
- The brain still dies at ITS walltime — guaranteed ≠ eternal. For longer
  horizons: chain (resubmit a successor CPU job that `claude --resume`s the
  same session — web-relink on resume is verified) and mind the day/night
  usage policy for long CPU holds.
- Anthropic API key lives in the G5K `$HOME` (same caveat as v2).
- Running `claude` on a chartreuse node specifically is the one unexercised
  step — but it's the same NFS `$HOME`/install verified working on kinovis
  nodes, so no surprise expected.
- Rule 2 still applies to the brain itself: it must not stack GPU jobs — one
  GPU-bearing child at a time.

## Gotchas

- Home directories on the access frontend (`access-north`/`access-south`) are
  **not** synced with site home directories — don't stage data there, use the
  per-site NFS mount or `scp ... login@<site>.g5k:`.
- `nvidia-smi --query-gpu=...` is a quick sanity check that you actually got a
  live GPU with the expected free memory, once connected to the node.
- Each hop's MOTD includes live incident/bug warnings for that site (e.g. broken
  MPI, wattmeter outages) — worth reading before reserving there, since they hint
  at what's currently unreliable on that site.
