## Issues

Any task should be created through issue tracker and assigned to a `Task: complexity` label. Doing this will connect project board with code and repo directly and will help backtracking changes. 


## Branches

Create new branch to add new features, fix bugs or anything else:
```bash
git checkout -b branch_name
```

## Fix an error in commit message

If you made an error in a commit message it could be easily fixed using these commands from an appropriate branch:

```bash
git commit --amend
git push --force
```

## References
- [3.2 Git Branching - Basic Branching and Merging](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)
- [git commit --ammend](https://stackoverflow.com/questions/8981194/changing-git-commit-message-after-push-given-that-no-one-pulled-from-remote)