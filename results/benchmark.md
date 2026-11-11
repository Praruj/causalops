| incident kind | n | PSI rank | KS rank | PSI x importance | drift + intervention |
|---|---|---|---|---|---|
| unit_shift | 60 | 100% | 100% | 100% | 100% |
| null_spike | 60 | 100% | 95% | 100% | 90% |
| covariate_shift | 60 | 95% | 92% | 95% | 100% |
| label_shift | 60 | 47% | 23% | 100% | 100% |
| trap | 60 | 40% | 32% | 100% | 97% |
| all | 300 | **76%** (71%-81%) | **68%** (63%-73%) | **99%** (97%-100%) | **97%** (95%-99%) |

Intervention misses: 8, of which 8 said 'no cause found' and 0 named a wrong cause.
Trap incidents where the method blamed the harmless decoy: psi_rank 60%, ks_rank 68%, psi_x_importance 0%, intervention 0%
Clean windows with no fault: 0 false alarms out of 60.
Cause type named correctly (when the cause was right, trap excluded): 76%.
