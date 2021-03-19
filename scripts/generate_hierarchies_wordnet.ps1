python -c "import nltk;nltk.download('wordnet')"

# Generate WNIDs
$DATASETS = @("CIFAR10", "CIFAR100")
foreach ($dataset in $DATASETS) {
  nbdt-wnids --dataset=$dataset
}

# Generate and test hierarchies
$MORE_DATASETS = @("CIFAR10", "CIFAR100", "TinyImagenet200")
foreach ($dataset in $MORE_DATASETS) {
  nbdt-hierarchy --dataset=$dataset --method=wordnet;
}