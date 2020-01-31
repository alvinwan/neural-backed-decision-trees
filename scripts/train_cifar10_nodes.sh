for wnid in fall11 n03575240 n03791235 n04524313 n03791235 n03125870 n00015388 n01471682 n01886756 n02370806;
  do CUDA_VISIBLE_DEVICES=${1:-0} python main.py --dataset=CIFAR10node --wnid=${wnid} --model=ResNet10 --epochs=200 --batch-size=512;
done
