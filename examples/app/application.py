from flask import Flask
from nbdt.model import SoftNBDT
from nbdt.models import ResNet18
from torchvision import transforms
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet


maybe_install_wordnet()
application = app = Flask(__name__)


@app.route('/')
def home():
    # load pretrained NBDT
    model = ResNet18()
    model = SoftNBDT(
      pretrained=True,
      dataset='CIFAR10',
      arch='ResNet18',
      hierarchy='wordnet',
      model=model)

    # load + transform image
    im = load_image_from_path("https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=32")
    transform = transforms.Compose([
      transforms.Resize(32),
      transforms.CenterCrop(32),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    x = transform(im)[None]

    # run inference
    outputs, decisions = model.forward_with_decisions(x)  # use `model(x)` to obtain just logits
    _, predicted = outputs.max(1)
    return {
        'predicted': [DATASET_TO_CLASSES['CIFAR10'][pred] for pred in predicted],
        'decisions': [[info['name'] for info in decision] for decision in decisions]
    }


if __name__ == '__main__':
    app.run()
