"""
Neural-Backed Decision Trees training on CIFAR10, CIFAR100, TinyImagenet200

The original version of this `main.py` was taken from kuangliu/pytorch-cifar.
The script has since been heavily modified to support a number of different
configurations and options: alvinwan/neural-backed-decision-trees
"""
import os
import argparse
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from nbdt import data, analysis, loss, models, metrics, tree as T
from nbdt.utils import progress_bar, generate_checkpoint_fname, generate_kwargs, Colors
from nbdt.thirdparty.wn import maybe_install_wordnet
from nbdt.models.utils import load_state_dict, make_kwarg_optional
from nbdt.tree import Tree


def main():
    maybe_install_wordnet()
    datasets = data.cifar.names + data.imagenet.names + data.custom.names
    parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
    parser.add_argument(
        "--batch-size", default=512, type=int, help="Batch size used for training"
    )
    parser.add_argument(
        "--epochs",
        "-e",
        default=200,
        type=int,
        help="By default, lr schedule is scaled accordingly",
    )
    parser.add_argument("--dataset", default="CIFAR10", choices=datasets)
    parser.add_argument(
        "--arch", default="ResNet18", choices=list(models.get_model_choices())
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )

    # extra general options for main script
    parser.add_argument(
        "--path-resume", default="", help="Overrides checkpoint path generation"
    )
    parser.add_argument(
        "--name", default="", help="Name of experiment. Used for checkpoint filename"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Download pretrained model. Not all models support this.",
    )
    parser.add_argument("--eval", help="eval only", action="store_true")
    parser.add_argument(
        "--dataset-test",
        choices=datasets,
        help="If not set, automatically set to train dataset",
    )
    parser.add_argument(
        "--disable-test-eval",
        help="Allows you to run model inference on a test dataset "
        " different from train dataset. Use an anlayzer to define "
        "a metric.",
        action="store_true",
    )

    # options specific to this project and its dataloaders
    parser.add_argument(
        "--loss", choices=loss.names, default=["CrossEntropyLoss"], nargs="+"
    )
    parser.add_argument("--metric", choices=metrics.names, default="top1")
    parser.add_argument(
        "--analysis", choices=analysis.names, help="Run analysis after each epoch"
    )

    # other dataset, loss or analysis specific options
    data.custom.add_arguments(parser)
    T.add_arguments(parser)
    loss.add_arguments(parser)
    analysis.add_arguments(parser)

    args = parser.parse_args()
    loss.set_default_values(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print("==> Preparing data..")
    dataset_train = getattr(data, args.dataset)
    dataset_test = getattr(data, args.dataset_test or args.dataset)

    transform_train = dataset_train.transform_train()
    transform_test = dataset_test.transform_val()

    dataset_train_kwargs = generate_kwargs(
        args,
        dataset_train,
        name=f"Dataset {dataset_train.__class__.__name__}",
        globals=locals(),
    )
    dataset_test_kwargs = generate_kwargs(
        args,
        dataset_test,
        name=f"Dataset {dataset_test.__class__.__name__}",
        globals=locals(),
    )
    trainset = dataset_train(
        **dataset_train_kwargs,
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = dataset_test(
        **dataset_test_kwargs,
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    assert trainset.classes == testset.classes or args.disable_test_eval, (
        trainset.classes,
        testset.classes,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    Colors.cyan(f"Training with dataset {args.dataset} and {len(trainset.classes)} classes")
    Colors.cyan(
        f"Testing with dataset {args.dataset_test or args.dataset} and {len(testset.classes)} classes"
    )

    # Model
    print("==> Building model..")
    model = getattr(models, args.arch)

    if args.pretrained:
        print("==> Loading pretrained model..")
        model = make_kwarg_optional(model, dataset=args.dataset)
        net = model(pretrained=True, num_classes=len(trainset.classes))
    else:
        net = model(num_classes=len(trainset.classes))

    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    checkpoint_fname = generate_checkpoint_fname(**vars(args))
    checkpoint_path = "./checkpoint/{}.pth".format(checkpoint_fname)
    print(f"==> Checkpoints will be saved to: {checkpoint_path}")

    resume_path = args.path_resume or checkpoint_path
    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        if not os.path.exists(resume_path):
            print("==> No checkpoint found. Skipping...")
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device(device))

            if "net" in checkpoint:
                load_state_dict(net, checkpoint["net"])
                best_acc = checkpoint["acc"]
                start_epoch = checkpoint["epoch"]
                Colors.cyan(
                    f"==> Checkpoint found for epoch {start_epoch} with accuracy "
                    f"{best_acc} at {resume_path}"
                )
            else:
                load_state_dict(net, checkpoint)
                Colors.cyan(f"==> Checkpoint found at {resume_path}")

    # hierarchy
    tree = Tree.create_from_args(args, classes=trainset.classes)

    # loss
    criterion = None
    for _loss in args.loss:
        if criterion is None and not hasattr(nn, _loss):
            criterion = nn.CrossEntropyLoss()
        class_criterion = getattr(loss, _loss)
        loss_kwargs = generate_kwargs(
            args,
            class_criterion,
            name=f"Loss {args.loss}",
            globals=locals(),
        )
        criterion = class_criterion(**loss_kwargs)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(3 / 7.0 * args.epochs), int(5 / 7.0 * args.epochs)]
    )

    class_analysis = getattr(analysis, args.analysis or "Noop")
    analyzer_kwargs = generate_kwargs(
        args,
        class_analysis,
        name=f"Analyzer {args.analysis}",
        globals=locals(),
    )
    analyzer = class_analysis(**analyzer_kwargs)

    metric = getattr(metrics, args.metric)()

    # Training
    @analyzer.train_function
    def train(epoch):
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch, args.epochs)

        print("\nEpoch: %d / LR: %.04f" % (epoch, scheduler.get_last_lr()[0]))
        net.train()
        train_loss = 0
        metric.clear()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            metric.forward(outputs, targets)
            transform = trainset.transform_val_inverse().to(device)
            stat = analyzer.update_batch(outputs, targets, transform(inputs))

            progress_bar(
                batch_idx,
                len(trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d) %s"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * metric.report(),
                    metric.correct,
                    metric.total,
                    f"| {analyzer.name}: {stat}" if stat else "",
                ),
            )
        scheduler.step()


    @analyzer.test_function
    def test(epoch, checkpoint=True):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        metric.clear()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)

                if not args.disable_test_eval:
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    metric.forward(outputs, targets)
                transform = testset.transform_val_inverse().to(device)
                stat = analyzer.update_batch(outputs, targets, transform(inputs))

                progress_bar(
                    batch_idx,
                    len(testloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d) %s"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * metric.report(),
                        metric.correct,
                        metric.total,
                        f"| {analyzer.name}: {stat}" if stat else "",
                    ),
                )

        # Save checkpoint.
        acc = 100.0 * metric.report()
        print(
            "Accuracy: {}, {}/{} | Best Accurracy: {}".format(
                acc, metric.correct, metric.total, best_acc
            )
        )
        if acc > best_acc and checkpoint:
            Colors.green(f"Saving to {checkpoint_fname} ({acc})..")
            state = {
                "net": net.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(state, f"./checkpoint/{checkpoint_fname}.pth")
            best_acc = acc

    if args.disable_test_eval and (not args.analysis or args.analysis == "Noop"):
        Colors.red(
            " * Warning: `disable_test_eval` is used but no custom metric "
            "`--analysis` is supplied. I suggest supplying an analysis to perform "
            " custom loss and accuracy calculation."
        )

    if args.eval:
        if not args.resume and not args.pretrained:
            Colors.red(
                " * Warning: Model is not loaded from checkpoint. "
                "Use --resume or --pretrained (if supported)"
            )
        with analyzer.epoch_context(0):
            test(0, checkpoint=False)
    else:
        for epoch in range(start_epoch, args.epochs):
            with analyzer.epoch_context(epoch):
                train(epoch)
                test(epoch)

    print(f"Best accuracy: {best_acc} // Checkpoint name: {checkpoint_fname}")

if __name__ == '__main__':
    main()