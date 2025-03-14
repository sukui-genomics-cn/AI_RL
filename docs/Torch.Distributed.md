# Why you should prefer DDP over DP

DataParallel is an older approach to data parallelism. DP is trivially simple (with just one extra line of code) but it is much less performant. DDP improves upon the architecture in a few ways.

| DP                                                           | DDP                                    |
| ------------------------------------------------------------ | -------------------------------------- |
| More overhead, model is replicated and destroyed at each forward pass | Model is replicated only once          |
| Only support single node parallelism                         | Supports scaling to multiple machnies  |
| Slower, uses multithreading on a single process and runs into global interpreter lock contention | Faster because it uses multiprocessing |
|                                                              |                                        |

**Notes**

> If your model contains any `BatchNorm` Layers, it needs to be convertd to `SyncBatchNorm` to sync the running tats of `BatchNorm` Layers across replicas.
>
> Use the helper function `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` to convert all `BatchNorm` layers in the model to `SyncBatchNorm`

## DDP series multigpu

Diff for [single_gpu.py](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/single_gpu.py) v/s [multigpu.py](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py)

These are the changes you typically make to a single-GPU training scripts to enable DDP.

**Import**

- `torch.multiprocessing`: is a Pytorch wrapper around Pythons native multiprocessing
- The distributed process group contains all the processes that can communicate and synchronize with each other.

```python
import torch
import torch.nn.functional as F
from utils import MyTrainDataset

import torch.multiprocessing as mp
import torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataPrallel as DDP
from torch.distributed import init_process_group, destropy_process_group
import os

```

### Constructing the process group

- First, before initializing the group process, call `set_device`, which sets the default GPU for each process. This is important to prevent hangs or excessive memory utilization on GPU:0
- The process group can be initialized by TCP (defalut) or from a shared file-system. Read more on [process group initialization](https://pytorch.org/docs/stable/distributed.html#tcp-initialization)
- `init_process_group` initializes the distributed process group.
- Read more about [choosing a DDP backend](https://pytorch.org/docs/stable/distributed.html#which-backend-to-use)

```python
def ddp_setup(rank:int, world_size:int):
    """
    args:
    	rank: Unique identifier of each process
    	world_size: Total number of process
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
```

**Constructing the DDP Model**

```python
self.model = DDP(model, device_ids=[gpu_id])
```

**Distributing input data**

- `DistributedSampler` chunks the input data across all distributed process.
- The `DataLoader` combines a dataset and sampler, and provides an iterable over the given dataset.
- Each process will receive an input batch of 32 samples; the effective batch size is `32*nprocs`, or 128 when using 4 GPUs.

```python
train_data = torch.utils.data.DataLoader(
	dataset=train_dataset,
    batch_size=32,
    shuffle=False,
    sampler=DistributedSampler(train_dataset),
)
```

- Calling the `set_epoch()` method on the `DistributedSampler` at the begining of each epoch is necessary to make suffling work properly across multiple epochs. Otherwise, the same ordering will be used in each epoch.

```python
def _run_epoch(self, epoch):
    b_sz = len(next(iter(self.train_data))[0])  # call this additional line at every epoch
    for source, targets in self.train_data:
        ...
        self._run_batch(source, targes)
    
```

**Saving model checkpoints**

We only need to save model checkpoints from one process. Without this condition, each process would save its copy of the identical model. Read more on saving and loading models with DDP [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#save-and-load-checkpoints)

```python
- ckp = self.model.state_dict()
+ ckp = self.model.module.state_dict()
...
- if epoch % self.save_every = 0:
+ if self.gpu_id == 0 and epoch % self.save_every = 0:
    self._save_checkpoint(epoch)
```

> Note: `Collective call` are functions that run on all the distributed process, and they are used to gather certain states or values to a specific process. Collective calls require all ranks to run the collective code. In this example, _save_checkpoint should not have any collective calls because it is only run on the `rank:0` process. If you need to make any collective calls, it should be before the `if self.gpu_ids == 0` check.

**Running the distributed trianing job**

- include new arguments `rank` (replacing `device` ) and `world size`
- `rank` is auto-allocated by DDP when calling `mp.spawn`
- `world size` is the number of process across the trianing job. For GPU training. this corresponds to the number of GPUs in use, and each process works on a dedicated GPU.

```python
- def mian(device, total_epochs, save_every):
+ def main(rank, world_size, total_epochs, save_every):
    ddp_setup(rank world_size)
    dataset, model, optimizer = load_trian_objs()
  - trainer = Trainer(model, train_data, optimizer, device, save_every)
  + trainer = Trainer(model, train_data, optimizer, rank, saver_every)
    train.train(total_epochs)
  + destropy_process_group()

if __name__ = "__main__":
    import sys
    total_epoch = int(sys.argv[1])
    save_every = int(sys.argv[2])
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, save_every), nprocs=world_size)
```

## DDP Series Fault Tolerance

In distributed training, a single process failure can disrupt the entire training job. Since the susceptibility for failure can be higher here, making your training script robust is particularly important here. You might also prefer your training job to be elastic, for example, compute resources can join and leave dynamically over the course of the job.

PyTorch offers a utility called `torchrun` that provides fault-tolaerance and elastic training. When a failure occurs, `torchrun` logs the errors and attempts to automatically restart all the process form the last save "snashot" of the training job.

The snapshot save more than just the model state; it can include details about the number of epoch run, optimizer sates or any other stateful attribute of the trianing job necessary for it continuity.

**Why use torchrun**

`torchrun` handles the minutiae of distributed training so that you don't need to . For instance.

- you don't need to set environment variables or explicitly pass the `rank` and `world_size`; `torchrun` assigns this along several other [environment variables](https://pytorch.org/docs/stable/elastic/run.html#environment-variables).
- No need to call `mp.spawn` in your script; you only need a generic `main()` entry point, and launch the scrip with `torchrun`. This way the same script can be run in non-distributed as well as single-node and multinode setups.
- Gracefully restarting training from the last saved training snapshot.

**Graceful restarts**

For graceful restart, you should structure your train script like:

```python
def main():
    load_snapshot(snapshot_path)
    initialize()
    train()
    
def train():
    for batch in iter(dataset):
        train_step(batch)
        
        if should_checkpoint:
            save_snapshot(snapshot_path)
```

if a failure occurs, `torchrun` will terminate all the processed and restart them. Each process entry point first loads and initialize the last sav4e snapshot, and continues training from there. So at any failure, you only lose the training process from the last saved snapshot.

In elastic training, whenever there are any membership changes (adding or removing nodes), `torchrun` will terminate and spawn processes on available devices. Having this structure ensures your training job can continue without manual intervention.

**Process group initialization**

- `torchrun` assings `RANK, WORLD_SIZE` automatically, among other envariables

```python
- def ddp_setup(rank, world_size):
+ def ddp_setup():
-     """
-     Args:
-         rank: Unique identifier of each process
-         world_size: Total number of processes
-     """
-     os.environ["MASTER_ADDR"] = "localhost"
-     os.environ["MASTER_PORT"] = "12355"
-     init_process_group(backend="nccl", rank=rank, world_size=world_size)
+     init_process_group(backend="nccl")
     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

**Use torchrun provided environment variables**

```python
- self.gpu_id = gpu_id
+ self.gpu_id = int(os.environ["LOCAL_RANK"])
```

**Saving and loading snapshots**

Regulary storing all the relevant information in snapshots allows our training job to seamlessly resume after an interruption

```python
+ def _save_snapshot(self, epoch):
+     snapshot = {}
+     snapshot["MODEL_STATE"] = self.model.module.state_dict()
+     snapshot["EPOCHS_RUN"] = epoch
+     torch.save(snapshot, "snapshot.pt")
+     print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")

+ def _load_snapshot(self, snapshot_path):
+     snapshot = torch.load(snapshot_path)
+     self.model.load_state_dict(snapshot["MODEL_STATE"])
+     self.epochs_run = snapshot["EPOCHS_RUN"]
+     print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
```

**Loading a snapshot in the Trainer constructor**

When restarting an interrupted training job, you script will first try to load a snapshot to resume trianing from.

```python
class Trainer:
   def __init__(self, snapshot_path, ...):
   ...
+  if os.path.exists(snapshot_path):
+     self._load_snapshot(snapshot_path)
   ...
```

**Resuming training**

Training can resume form the last epoch run, instead of starting all over from scratch.

```python
def train(self, max_epochs: int):
-  for epoch in range(max_epochs):
+  for epoch in range(self.epochs_run, max_epochs):
      self._run_epoch(epoch)
```

