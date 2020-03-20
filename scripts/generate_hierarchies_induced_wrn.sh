for i in "CIFAR10 wrn28_10_cifar10" "CIFAR100 wrn28_10_cifar100" "TinyImagenet200 wrn28_10"; do
  read dataset model <<< "${i}";

  python main.py --eval --pretrained --model=${model} --dataset=${dataset}
  python generate_hierarchy.py --method=induced --induced-checkpoint=checkpoint/ckpt-${dataset}-${model}.pth --dataset=${dataset}
  python test_generated_hierarchy.py --method=induced --induced-checkpoint=checkpoint/ckpt-${dataset}-${model}.pth --dataset=${dataset}
done;
