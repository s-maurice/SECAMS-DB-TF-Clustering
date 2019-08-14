import PIL
import numpy as np
from PIL import Image

plot_types = ["correct", "incorrect", "no_normal", "no_normal_correct", "irregular"]

im_list_sk = []
im_list_tf = []
for plot_type in plot_types:
    im_list_sk.append("SciKit Learn/" + "Scikit Learn (" + plot_type + ")" + ".png")
    im_list_tf.append("SciKit Learn/" + "TensorFlow (" + plot_type + ")" + ".png")

imgs_sk = [PIL.Image.open(i) for i in im_list_sk]
imgs_tf = [PIL.Image.open(i) for i in im_list_sk]

# Combine Image
img_pairs = zip(imgs_sk, imgs_tf)
img_comb_list = [np.hstack((np.asarray(i.resize(i.size)) for i in img_pair)) for img_pair in img_pairs]

# Save Image
for filter_type, img_pair in zip(plot_types, img_comb_list):
    cur_pair = PIL.Image.fromarray(img_pair)
    cur_pair.save('Combined Plots/' + filter_type + ".png")
