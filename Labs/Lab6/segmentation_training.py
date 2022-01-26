# from PIL import Image, ImageEnhance
import os.path
from keras_segmentation.models.unet import vgg_unet
import matplotlib.pyplot as plt
from IPython.display import Image


def main():
    # TODO this is an example, not actual exercise

    # im = Image.open('../Labs/Lab6/example_dataset/annotations_prepped_test/0016E5_07959.png')

    model = vgg_unet(n_classes=50,  input_height=320, input_width=640)

    model.train(
        train_images="../Labs/Lab6/example_dataset/images_prepped_train/",
        train_annotations="../Labs/Lab6/example_dataset/annotations_prepped_train/",
        checkpoints_path="../Labs/Lab6/tmp/vgg_unet_1", epochs=5
    )

    out = model.predict_segmentation(
        inp="../Labs/Lab6/example_dataset/images_prepped_test/0016E5_07965.png",
        out_fname="../Labs/Lab6/tmp/out.png"
    )

    plt.imshow(out)
    Image('/tmp/out.png')



