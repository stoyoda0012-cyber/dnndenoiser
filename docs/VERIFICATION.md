# Verifying a measurement before it is published

How the principles in `AGENTS.md` §8.1 are applied to a measurement record — a
benchmark, a failure boundary, a reproduction — from its preregistration to the
moment it may be quoted outside the repository.

This is procedure, not doctrine. It says what to do and in what order. It
deliberately does not fix how many reviews a record gets or which model or person
performs them; those depend on the record.

Everything here was learned the expensive way on P2-A
(`docs/preregistration/P2A-position-shift-boundary.md`). Its revision log is the
worked example: four checks that compared an expression against itself, a renderer
guard that verified a sixth of what it claimed, and prose that quoted numbers no
guard reached — each of which passed every run until someone built the wrong input
by hand.

## 1. Before the first run

**Publish the preregistration.** With a person's approval, push it to the public
repository before any code that measures anything runs. A preregistration that
exists only in local history is attested only by timestamps its author controls.

**Every voiding self-check gets an accept/reject pair**, committed as tests:

- the correct input is accepted;
- a named wrong input — the failure the check exists for — is rejected;
- the rejection is pinned to its intended reason (`pytest.raises(..., match=...)`),
  because a bare `pytest.raises` passes when some other check rejects the input on
  the way.

Rejecting one constructed failure does not show that a check catches every failure.
Each test names the one it covers. A check that cannot be tested this way — for
example because it is written inline in a run loop — is listed as untested, not
counted silently.

**Every check names its independent source of truth**: a literal, an analytic
result, the stored data, an independent transform. If the answer is "the same
function that produced the value", it is a tautology and does not count.

**The renderer's guard gets a tamper test**: a copy of a record, altered one field at
a time, must be refused for every field the guard claims to cover. State what the
guard does *not* cover in the rendered report itself.

**Provenance is recorded by the run, not typed**, as three separate facts: the code
commit, whether the working tree was clean, and the version of the registration
document the run was made against. A run from a dirty tree is refused.

**Smoke-test the whole pipeline** at a size that takes seconds, into a scratch
directory.

Only then run the measurement. A full run exists to produce numbers; it is the most
expensive possible way to discover that a check was inert.

## 2. After the run, before anything is written about it

**The record is the only place numbers live.** Generated reports come from it.

**Numbers quoted in prose are anchored per occurrence.** Each carries a source anchor
naming the metric, condition, unit and record field it comes from, and a test
recomputes it from the record at the precision quoted. Numbers deliberately not from
the record — registered thresholds, values from discarded runs quoted as history —
carry an anchor that says so and why. The anchoring test is itself tested against
planted errors: a transcription error, the right value from the wrong cell, a removed
anchor, a sign flip.

The test cannot tell whether a sentence reads its source correctly. It pins each
number to a named source so that a reviewer can check the reading.

**Keep the registered rules while writing.** In particular: no inferential claim
about a cell no registered prediction names; no mechanism for a failed prediction
that the record does not test; no empirical regularity stated as a law; no quantity
quoted to more precision than its uncertainty across replicates *and* across
reasonable estimators supports; and where a number has a counterpart in another
direction, arm or level that changes how it reads, quote both or neither.

## 3. Review of the claims

A defined check can be automated. Whether it was the right check, and whether the
prose claims only what the record supports, cannot. That is what review is for.

- **Materials:** the frozen commit — preregistration, record, report, citation
  registry — identified by hash.
- **Not given:** the author's expected conclusions, suspicions or preferred reading.
  A brief that lists what the author thinks is wrong steers the reviewer toward it and
  away from everything else.
- **Independence:** a reviewer that does not share the author's context. A different
  model family, or a person, where one is available; a fresh context at minimum.
- **Checklist,** fixed before the review, plus anything else the reviewer finds:

  1. Does every claim stay inside `AGENTS.md` §5?
  2. Does every claim carry §6's conditions — data and provenance, split rule,
     noise model, seed count, metric and its independence assumptions, comparison
     conditions?
  3. Is a claim made about a cell no registered prediction names? If a cell is cited
     under a stated exception, does the citation meet the exception's conditions?
  4. Is a failed or undecided prediction softened, reinterpreted, or given a
     mechanism the record does not test?
  5. Is an empirical regularity stated as a law?
  6. Is a quantity quoted with more precision than its uncertainty supports?
  7. Is a directional or comparative claim made without clearing the confounds the
     record itself records?
  8. Is a number quoted from one direction, arm or level while a counterpart that
     changes its reading is left out?
  9. Does outward-facing text go beyond the record's `claim_scope`?
  10. Are the provenance limits — what a reader can and cannot verify — stated?

- **Output:** findings with a severity each, and a verdict on what, if anything, may
  be quoted outside the repository.

## 4. Publication

A push to the public repository publishes the history, not only the final tree.
Discarded records and superseded claims in earlier commits become readable.

1. Finish the local fixes, tests and review.
2. Review every commit the push would expose: what each adds, what it later
   supersedes, what it would be read as saying on its own. Scan them for
   developer-specific paths — the tree being clean is not enough:

   ```bash
   python tests/test_tracked_paths.py origin/main..HEAD   # must report 0
   ```

   The script also warns about local refs outside branches, remotes and tags —
   another tool's checkpoint, for example — that reach such a path. A log-based scan
   cannot see them, because they point at trees rather than commits. `git push`,
   `--all` and `--tags` do not send them; **`git push --mirror` does, so it is never
   used here.**

   If anything is found in an **unpublished** commit, rewrite the unpublished range so
   the commit no longer carries it, remap any hash references the rewrite breaks, and
   disclose the edit where the record is described. Once published, the only remedy is
   a force-push, which is a different and much worse decision.
3. Put the diff, the history and the open items in front of a person. The person
   decides whether, and in what form, it is published.
4. After approval, push a working branch and open a draft pull request against
   `main`. **A draft pull request is already public**, which is why step 3 comes
   before it. CI runs only on pull requests to `main`, pushes to `main`, and manual
   dispatch — a branch push alone does not run it.
5. Merge after CI and review.

Nothing from a record goes into the package README, the package documentation or a
release note until the record's claims have been reviewed and a person has decided
what may be quoted. That decision is recorded beside the record, naming the place, the statements and
the commit reviewed, and the numbers quoted there are tested as AGENTS.md §6, exception
(b), requires.
