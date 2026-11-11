# Contributing to SQLgenerator

Use a separate branch for each contribution and open a pull request so the
team can review changes before merging them into `main`.

## Work on the sanika branch

Clone the project once, then check out the existing branch:

```bash
git clone https://github.com/atharva22210/SQLgenerator.git
cd SQLgenerator
git switch sanika
```

Before starting new work, check that your working tree is clean with
`git status`, then download the latest changes:

```bash
git pull --ff-only origin sanika
```

After editing files, review and commit the intended changes. Replace
`path/to/changed-file` with the file you want to include:

```bash
git diff
git add path/to/changed-file
git diff --cached
git commit -m "Describe the change"
git push origin sanika
```

Open a pull request on GitHub from `sanika` into `main` when the contribution
is ready for review. Include what changed and how you checked it.

## Collaborating with a friend

The repository owner can invite collaborators through the repository's
**Settings → Collaborators** page. After accepting an invitation with write
access, a collaborator can push to branches allowed by the repository's
rules. Knowing a branch name or being able to clone the repository does not
grant write access.

For simultaneous work, use separate branches and pull requests to avoid
overwriting one another's changes. If a push is rejected because someone
else pushed first, fetch and integrate their changes before trying again.
Avoid force-pushing shared branches.

Contributors without write access can use a fork and submit a pull request,
where the repository's visibility and policies permit it.
