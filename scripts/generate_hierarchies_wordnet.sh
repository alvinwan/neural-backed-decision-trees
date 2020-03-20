python -c "import nltk;nltk.download('wordnet')"

# Generate WNIDs
for dataset in CIFAR10 CIFAR100;
do
  python generate_wnids.py --dataset=${dataset}
done;

# Generate and test hierarchies
for dataset in CIFAR10 CIFAR100 TinyImagenet200;
do
  python generate_hierarchy.py --dataset=${dataset} --single-path;
  python test_generated_hierarchy.py --dataset=${dataset} --single-path;
done;
