# Repository instructions

## Keep the working branch current with main

For every code-change task, and immediately before creating any commit or
pushing changes:

1. Inspect `git status` and the current branch. Preserve all existing user
   changes; never discard them or automatically stash them.
2. Run `git fetch origin`. Use the freshly fetched `origin/main` as the source
   of truth, rather than the local `main` branch.
3. Run `git rev-list --count HEAD..origin/main`. The result must be `0` before
   making requested changes, creating a commit, or pushing.
4. If the count is greater than zero, merge `origin/main` into the working
   branch with `git merge origin/main`. The user authorizes this routine sync
   and its merge commit. Prefer merging to rebasing shared branches; never
   force-push to satisfy this rule. Recheck the count after merging.
5. Resolve merge conflicts carefully and run checks appropriate to the merged
   changes. If existing uncommitted work prevents a safe merge, conflicts need
   user decisions, or fetching fails, explain the blocker and obtain the needed
   input or access before continuing dependent work. Never claim the branch is
   current when it could not be verified.

If the current branch is `main`, create a task branch from the updated main
history before implementing changes. Do not automatically push or merge a
pull request unless the user's task authorizes it.

This check guarantees zero commits behind the last fetched `origin/main`.
Another contributor may update GitHub's `main` afterward, so fetch and check
again at each checkpoint above. Report the verified sync status when finishing.
