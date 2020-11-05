import os  # NOQA: E402
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # NOQA: E402
import matplotlib.pyplot as plt
import torchvision.models as models

from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

### 1. Load a pre-trained Model
# model = models.alexnet(pretrained=True)
model = models.vgg16(pretrained=True)
### 2. Create an instance of Backprop with the model
backprop = Backprop(model)
path = './figure'
path_names = os.listdir(path)
for i in range(len(path_names)):
    peacock = apply_transforms(load_image(os.path.join(path,path_names[i])))
    backprop.visualize(peacock, None,guided=True, use_gpu=True)
    plt.show()