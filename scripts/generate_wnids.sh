for dataset in CIFAR10 CIFAR100;
  do
    python generate_wnids.py --dataset=${dataset};
  done;
