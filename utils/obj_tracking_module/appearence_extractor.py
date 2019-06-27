# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import random
import string
import tensorflow as tf
from PIL import ImageColor, Image


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = np.squeeze(np.asarray(f(batch_data_dict))).tolist()
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = np.squeeze(np.asarray(f(batch_data_dict))).tolist()


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    cv2.imwrite("/content/sample_data/{}.jpg".format(
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))), image)
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "%s:0" % output_name)
        # self.input_var = tf.get_default_graph().get_tensor_by_name(
        #     "input:0")
        # self.output_var = tf.get_default_graph().get_tensor_by_name(
        #     "pool5:0")

        # assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes, mask=None):
        # print(image.shape)
        image_patches = []
        #adding dummy images to make batch_size 10 permanently because of googlenet
        # i = len(boxes)//10

        #This loop is not required since we are generating feature for only one bbox at
        #a time, but left for future if we intend to use for all boxes at once. 
        for box in boxes:
            patch = None
            if mask is not None:
                rgb = ImageColor.getrgb('gray')
                solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
                pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
                pil_mask = Image.fromarray(np.uint8(255.0*(np.ones_like(mask)-mask))).convert('L')
                image = Image.composite(pil_solid_color, pil_image, pil_mask)
                image = np.array(image.getdata()).reshape((image.size[0], image.size[1], 3))
            
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
            # else:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder

            # features = encoder(bgr_image, rows[:, 2:6].copy())
            # # print(features.shape)
            # feat = np.squeeze(np.asarray(features))



    # encoder = create_box_encoder(args.model, batch_size=10)
    # generate_detections(encoder, args.mot_dir, args.output_dir,
                        # args.detection_dir)


