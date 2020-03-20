python main.py --eval --pretrained --model=wrn28_10_cifar10 --dataset=CIFAR10
python main.py --eval --pretrained --model=wrn28_10_cifar100 --dataset=CIFAR100

python generate_hierarchy.py --method=induced --induced-checkpoint=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --dataset=CIFAR10
python generate_hierarchy.py --method=induced --induced-checkpoint=checkpoint/ckpt-CIFAR100-wrn28_10_cifar100.pth --dataset=CIFAR100

python test_generated_hierarchy.py --method=induced --induced-checkpoint=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --dataset=CIFAR10
python test_generated_hierarchy.py --method=induced --induced-checkpoint=checkpoint/ckpt-CIFAR10-wrn28_10_cifar100.pth --dataset=CIFAR100
