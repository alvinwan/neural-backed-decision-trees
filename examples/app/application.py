from flask import Flask, flash, request, redirect, url_for
from nbdt.model import SoftNBDT
from nbdt.models import ResNet18
from torchvision import transforms
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet
from werkzeug.utils import secure_filename
from PIL import Image
import os

maybe_install_wordnet()
application = app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def inference(im):
    # load pretrained NBDT
    model = ResNet18()
    model = SoftNBDT(
      # pretrained=True,
      dataset='CIFAR10',
      arch='ResNet18',
      hierarchy='wordnet',
      model=model)

    # load + transform image
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


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


image_urls = {
    'cat': 'https://images.pexels.com/photos/126407/pexels-photo-126407.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=300',
    'bear': 'https://images.pexels.com/photos/158109/kodiak-brown-bear-adult-portrait-wildlife-158109.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=300',
    'dog': 'https://images.pexels.com/photos/1490908/pexels-photo-1490908.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=300',
}


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' or request.args.get('url', None):
        url = request.form.get('url', request.args.get('url', None))
        if url:
            im = load_image_from_path(url)
            return inference(im)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            im = Image.open(file.stream)
            return inference(im)
        return f'nope. Allowed file? ({file.filename}) Got a file? ({bool(file)})'
    return f'''
    <!doctype html>
    <title>Upload new File OR specify URL</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=input name=url placeholder=url>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <a href="?url={image_urls['cat']}"><img src="{image_urls['cat']}"></a>
    <a href="?url={image_urls['bear']}"><img src="{image_urls['bear']}"></a>
    <a href="?url={image_urls['dog']}"><img src="{image_urls['dog']}"></a>
    '''


if __name__ == '__main__':
    app.run()
