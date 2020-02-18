for dataset in CIFAR10 CIFAR100 tiny-imagenet-200;
  do
    for method in build random;
      do
        python generate_tree.py --method=${method} --dataset=${dataset};
        python test_generated_tree.py --method=${method} --dataset=${dataset};
      done;
  done;
