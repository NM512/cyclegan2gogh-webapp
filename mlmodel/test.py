from .options.test_options import TestOptions
from .data import create_dataset
from .models import create_model
from .util.visualizer import save_images
from .util import html

import os
import numpy as np
from PIL import Image

def cycle_gan(folder_path):
    # optクラスの中で既に入力値の変換処理があることに気を付けること
    opt = TestOptions().parse()  # get test options
    #
    print(opt)
    
    # hard-code some parameters for test
    opt.dataroot = folder_path
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)     # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    if opt.eval:
        print("test_mode")
        model.eval()
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        processed = np.array((visuals["fake"][0].detach().numpy().copy().transpose(1,2,0) + 1) *128, dtype="uint8")
        return processed

if __name__ == '__main__':
    ret = cycle_gan()
    result = Image.fromarray(np.uint8(ret))
    result.save("result.png")
    