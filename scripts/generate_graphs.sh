for dataset in CIFAR10 CIFAR100 TinyImagenet200;
  do
    do
      python generate_graph.py --dataset=${dataset};
      python test_generated_graph.py --dataset=${dataset};
    done;
