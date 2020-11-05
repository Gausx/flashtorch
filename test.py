import os  # NOQA: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # NOQA: E402

import torchvision.models as models

from flashtorch.activmax import GradientAscent


model = models.vgg16(pretrained=True)

# Print layers and corresponding indicies

print(list(model.features.named_children()))

conv1_2 = model.features[2]
conv1_2_filters = [17, 33, 34, 57]

conv2_1 = model.features[5]
conv2_1_filters = [27, 40, 68, 73]

conv3_1 = model.features[10]
conv3_1_filters = [31, 61, 147, 182]

conv4_1 = model.features[17]
conv4_1_filters = [238, 251, 338, 495]

conv5_1 = model.features[24]
conv5_1_filters = [45, 271, 363, 409]

g_ascent = GradientAscent(model.features)


g_ascent.visualize(conv1_2, conv1_2_filters, title='conv1_2');
g_ascent.visualize(conv2_1, conv2_1_filters, title='conv2_1');
g_ascent.visualize(conv3_1, conv3_1_filters, title='conv3_1');
g_ascent.visualize(conv4_1, conv4_1_filters, title='conv4_1');
g_ascent.visualize(conv5_1, conv5_1_filters, title='conv5_1');

g_ascent.visualize(conv5_1, title='Randomly selected filters from conv5_1');

g_ascent.visualize(conv5_1, 3, title='conv5_1 filter 3');

output = g_ascent.visualize(conv5_1, 3, title='conv5_1 filter 3', return_output=True);

print('num_iter:', len(output))
print('optimized image:', output[-1].shape)

g_ascent.deepdream('./figure/toucan.jpg', conv5_1, 33)


output = g_ascent.optimize(conv5_1, 3)

print('num_iter:', len(output))
print('optimized image:', output[-1].shape)