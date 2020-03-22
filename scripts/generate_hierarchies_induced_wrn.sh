for i in "CIFAR10 wrn28_10_cifar10" "CIFAR100 wrn28_10_cifar100" "TinyImagenet200 wrn28_10"; do
  read dataset model <<< "${i}";

  python main.py --pretrained --model=${model} --dataset=${dataset} --epochs=0
  nbdt-hierarchy --method=induced --induced-checkpoint=checkpoint/ckpt-${dataset}-${model}.pth --dataset=${dataset}
done;
