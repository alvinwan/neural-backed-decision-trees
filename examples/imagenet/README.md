# Neural-Backed Decision Trees on ImageNet

Just a loss hook in the Classy Vision workflow. The `classy_train.py` is 100% boilerplate. To launch a run with 8 GPUs on one node, use:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_CUDA_DEVICES:-8} \
    --use_env \
    classy_train.py \
    --config=${CONFIG:-configs/resnet18-nbdt.json} \
    --distributed_backend ddp
```