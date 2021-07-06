# vgg16-haiku
VGG-16 in JAX and Haiku, ported from the [torchvision implementation](https://github.com/pytorch/vision/blob/cd181889d72ebab77f6657055bd415c2db63fa31/torchvision/models/vgg.py) and weights.

Download pickle with pretrained weights [here](https://drive.google.com/file/d/13AB7ADVhZQen1j6kB99A8Pn_220EHr0A/view?usp=sharing)

Usage:

```python
import imageio
import numpy as np

from vgg.vgg import get_model

model, params = get_model()

image = imageio.imread('some_image.jpg') / 255
image = np.transpose(image, [2, 0, 1])
# imagenet normalization:
image -= np.array([0.485, 0.456, 0.406])[:, None, None]
image /= np.array([0.229, 0.224, 0.225])[:, None, None]

#prediction:
print(np.argmax(model.apply(params, image)))
```
