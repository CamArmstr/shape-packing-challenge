# Solution Milestones

Archived snapshots of significant results during the shape-packing-challenge run.
All solutions are also recoverable via `git log --grep="best:"` + `git show <hash>:best_solution.json`.

## Milestone Archive

| File | Score (R) | Date | Method | Notes |
|------|-----------|------|--------|-------|
| `R2.970026_mar28.json` | ~2.970 | Mar 28 | Numba SA + hill-climber | Pre-LNS overnight floor; contacts=30 basin |
| `mar29-lns3-2.961912.json` | 2.961912 | Mar 29 04:05 | lns3 (LNS + GJK) | First break below leaderboard #1 (2.96175); found at 91s |
| `R2.961486.json` | 2.961486 | Mar 29 04:10 | lns3 | lns3 second hit |
| `R2.961327.json` | 2.961327 | Mar 29 04:13 | lns3 pass-2 | Fingerprinted; 2-11-2 topology, contacts=30 |
| `R2.960955.json` | 2.960955 | Mar 29 ~04:25 | deep_polish | Sustained GJK polish pass |
| `R2.960736.json` | 2.960736 | Mar 29 ~04:26 | deep_polish | Current best as of archive creation |
| `mar29-current.json` | (latest) | Mar 29 | — | Copy of best_solution.json at archive creation; may be stale |

## How to recover any commit
```bash
git log --oneline --grep="best: R=2.960"
git show <hash>:best_solution.json > recovered.json
python3 run.py recovered.json
```

## Topology fingerprint (R=2.961327)
- **2-11-2** ring structure: 2 inner (r<1.5), 11 mid (1.5-2.2), 2 outer (r≥2.2)
- 30 GJK contacts
- Shapes 1, 11, 12 sit at r=1.961327 (equidistant from MEC center)
- Shapes 13, 14 are the boundary-defining pair (r≈2.786)

See `best_fingerprint.md` for full contact pair list and `best_solution_annotated.png` for visualization.
