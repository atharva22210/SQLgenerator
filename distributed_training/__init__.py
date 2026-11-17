"""Learning map for the distributed-training package.

Read experiment_plan -> ddp_core -> ddp_worker, then the Modal launcher.

* experiment_plan names the controlled A/B cases and their hyperparameters.
* ddp_core defines reproducible data assignment and the gradient mathematics.
* ddp_worker runs one model/optimizer copy on one GPU under torchrun.
* sql_validation builds a filtered SQL evaluation reader and computes its loss.
* cuda_preflight checks the real CUDA/NCCL/BF16 path on a tiny synthetic model.

There are two kinds of process creation in this project: Modal .spawn()
submits the suite to the cloud; torchrun launches cooperating GPU processes
inside its container. They have different responsibilities. Importing this
package alone does not allocate GPUs or start training.
"""
