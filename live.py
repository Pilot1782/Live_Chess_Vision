#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# TensorFlow Chessbot
# This contains ChessboardPredictor, the class responsible for loading and
# running a trained CNN on chessboard screenshots. Used by chessbot.py.
# A CLI interface is provided as well.
#
#   $ ./tensorflow_chessbot.py -h
#   usage: tensorflow_chessbot.py [-h] [--url URL] [--filepath FILEPATH]
#
#    Predict a chessboard FEN from supplied local image link or URL
#
#    optional arguments:
#      -h, --help           show this help message and exit
#      --url URL            URL of image (ex. http://imgur.com/u4zF5Hj.png)
#     --filepath FILEPATH  filepath to image (ex. u4zF5Hj.png)
#
# This file is used by chessbot.py, a Reddit bot that listens on /r/chess for
# posts with an image in it (perhaps checking also for a statement
# "white/black to play" and an image link)
#
# It then takes the image, uses some CV to find a chessboard on it, splits it up
# into a set of images of squares. These are the inputs to the tensorflow CNN
# which will return probability of which piece is on it (or empty)
#
# Dataset will include chessboard squares from chess.com, lichess
# Different styles of each, all the pieces
#
# Generate synthetic data via added noise:
#  * change in coloration
#  * highlighting
#  * occlusion from lines etc.
#
# Take most probable set from TF response, use that to generate a FEN of the
# board, and bot comments on thread with FEN and link to lichess analysis.
#
# A lot of tensorflow code here is heavily adopted from the
# [tensorflow tutorials](https://www.tensorflow.org/versions/0.6.0/tutorials/pdes/index.html)
import copy
import os
import sys
import threading
import time
import traceback
from itertools import repeat
from tkinter import PhotoImage
from typing import Union, Callable

import PIL.Image
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageTk
import chess
import chess.engine
import chess.svg
import cv2
import mss
import numpy as np
import pygetwindow
import tensorflow as tf

import tksvg
from customtkinter import (
    CTk,
    CTkLabel,
    CTkSlider,
    HORIZONTAL,
    VERTICAL,
    CTkImage, CTkButton, CTkEntry, CTkFrame,
)
from numba import jit

import chessboard_finder
import helper_image_loading
from helper_functions import shortenFEN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Ignore Tensorflow INFO debug messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Ignore floating point off by one error

# global vars
cache = {}


def load_graph(frozen_graph_filepath):
    # Load and parse the protobuf file to retrieve the unserialized graph_def.
    with tf.io.gfile.GFile(frozen_graph_filepath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import graph def and return.
    with tf.Graph().as_default() as graph:
        # Prefix every op/nodes in the graph.
        tf.import_graph_def(graph_def, name="tcb")
    return graph


def bind(num, min_val, max_val):
    """Bind num between min and max"""
    return max(min(num, max_val), min_val)


def verticalness(_img):
    """
    Returns the standard deviation of vertical and horizontal lines in the image.

    :param _img: the grayscale numpy array of the image
    :return:
    """

    gx, gy = np.gradient(_img)

    gx_pos = gx.copy()
    gx_pos[gx_pos < 0] = 0
    gx_neg = -gx.copy()
    gx_neg[gx_neg < 0] = 0

    gy_pos = gy.copy()
    gy_pos[gy_pos < 0] = 0
    gy_neg = -gy.copy()
    gy_neg[gy_neg < 0] = 0

    # 1-D amplitude of hough transform for gradients about X & Y axes
    hough_gx = gx_pos.sum(axis=1) * gx_neg.sum(axis=1)
    hough_gy = gy_pos.sum(axis=0) * gy_neg.sum(axis=0)

    return min(hough_gx.std() / hough_gx.size,
               hough_gy.std() / hough_gy.size)


def rotate_vertical(_img, max_rot=90):
    """
    Rotates the image to be vertical, up to max_rot degrees.

    :param _img: the grayscale numpy array of the image
    :param max_rot: the maximum number of degrees to rotate the image
    :return:
    """
    stds = np.array(list(zip(range(-max_rot, max_rot + 1), repeat(0))))
    for i in range(-max_rot, max_rot + 1):
        stds[i + max_rot][1] = verticalness(_img.rotate(i, expand=True))

    # get the rotations with stds over 8000
    rotations = stds[stds[:, 1] > 8000]

    # get the smallest rotation
    rotation = rotations[rotations[:, 1].argmin()][0]

    # rotate the image
    _img.rotate(rotation, expand=True).show()

    return rotation


@jit
def auto_rotate(_img, crop=False) -> float:
    """
    Automatically rotates the image to be vertical.
    :param _img: the grayscale numpy array of the image
    :param crop: whether to crop the image to the detected rectangle
    :return: the angle of rotation
    """
    _input_img = _img.copy()
    if len(_input_img.shape) == 3:
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    # find all the lines in the image
    edges = cv2.Canny(_img, 50, 150, apertureSize=3)

    # find the contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get the areas of the contours
    areas = [cv2.contourArea(c) for c in contours]
    areas = np.array(areas)
    areas = np.sort(areas)

    # exclude the contours that are too small
    areas = areas[areas > 100]

    # exclude areas larger than 0.9 * max_area
    max_area = np.max(areas)
    areas = areas[areas < 0.9 * max_area]

    # use a box plot to exclude the outliers
    q1 = np.percentile(areas, 25)
    q3 = np.percentile(areas, 75)
    iqr = q3 - q1
    min_area = q1 - 1.5 * iqr
    areas = areas[areas > min_area]

    # draw the contours shaded with a random color
    _output_img = _input_img.copy()
    for c in contours:
        if cv2.contourArea(c) not in areas:
            continue
        cv2.drawContours(_output_img, [c], -1, (255, 255, 255), -1)

    # find new contours from our mask
    if len(_output_img.shape) == 3:
        _gray = cv2.cvtColor(_output_img, cv2.COLOR_BGR2GRAY)
    else:
        _gray = _output_img
    _, mask = cv2.threshold(_gray, 240, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # only use the largest contour
    max_area = 0
    max_contour = None
    for c in contours:
        if cv2.contourArea(c) > max_area:
            max_area = cv2.contourArea(c)
            max_contour = c

    # draw the largest contour filled in
    rot_rect = cv2.minAreaRect(max_contour)

    new_angle = rot_rect[2]
    if abs(new_angle) > 30:
        print(f"Correcting angle from {new_angle}")
        sign = np.sign(new_angle)
        while (45 % new_angle) == 45:
            new_angle = new_angle - 45

        new_angle = 45 % new_angle * -sign

    if new_angle != 0:
        print(f"Rotating by {new_angle} degrees")

    rot_rect = (rot_rect[0], (rot_rect[1][0], rot_rect[1][1]), new_angle)

    if not crop:
        return new_angle, _output_img

    # crop tmp to the four points in the rotated rectangle
    box = cv2.boxPoints(rot_rect)
    box = np.intp(box)

    box2 = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    box2[0] = box[np.argmin(s)]
    box2[2] = box[np.argmax(s)]

    diff = np.diff(box, axis=1)
    box2[1] = box[np.argmin(diff)]
    box2[3] = box[np.argmax(diff)]

    mHeight = max(
        int(np.sqrt((box2[2][0] - box2[0][0]) ** 2 + (box2[2][1] - box2[0][1]) ** 2)),
        int(np.sqrt((box2[3][0] - box2[1][0]) ** 2 + (box2[3][1] - box2[1][1]) ** 2)),
    )
    mWidth = max(
        int(np.sqrt((box2[1][0] - box2[0][0]) ** 2 + (box2[1][1] - box2[0][1]) ** 2)),
        int(np.sqrt((box2[2][0] - box2[3][0]) ** 2 + (box2[2][1] - box2[3][1]) ** 2)),
    )

    dst = np.array([[0, 0], [mWidth - 1, 0], [mWidth - 1, mHeight - 1], [0, mHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(box2, dst)
    tmp = cv2.warpPerspective(_input_img, M, (mWidth, mHeight))

    # resize the image to a square
    tmp = cv2.resize(tmp, (max(mWidth, mHeight), max(mWidth, mHeight)))
    # cv2.imshow("tmp", tmp)
    # cv2.waitKey(0)

    return new_angle, tmp


class ChessboardPredictor(object):
    """ChessboardPredictor using saved model"""

    def __init__(self, frozen_graph_path='saved_models/frozen_graph.pb'):
        # Restore model using a frozen graph.
        print("\t Loading model '%s'" % frozen_graph_path)
        graph = load_graph(frozen_graph_path)
        self.sess = tf.compat.v1.Session(graph=graph)

        # Connect input/output pipes to model.
        self.x = graph.get_tensor_by_name('tcb/Input:0')
        self.keep_prob = graph.get_tensor_by_name('tcb/KeepProb:0')
        self.prediction = graph.get_tensor_by_name('tcb/prediction:0')
        self.probabilities = graph.get_tensor_by_name('tcb/probabilities:0')
        print("\t Model restored.")

    def get_prediction(self, tiles):
        """Run trained neural network on tiles generated from image"""
        if tiles is None or len(tiles) == 0:
            print("Couldn't parse chessboard")
            return None, 0.0

        # Reshape into Nx1024 rows of input data, format used by neural network
        validation_set = np.swapaxes(np.reshape(tiles, [32 * 32, 64]), 0, 1)

        # Run neural network on data
        guess_prob, guessed = self.sess.run(
            [self.probabilities, self.prediction],
            feed_dict={self.x: validation_set, self.keep_prob: 1.0})

        # Prediction bounds
        a = np.array(list(map(lambda x: x[0][x[1]], zip(guess_prob, guessed))))
        tile_certainties = a.reshape([8, 8])[::-1, :]

        # Convert guess into FEN string
        # guessed is tiles A1-H8 rank-order, so to make a FEN we just need to flip the files from 1-8 to 8-1
        def label_index2name(label_index):
            return ' ' if label_index == 0 else ' KQRBNPkqrbnp'[label_index]

        piece_names = list(map(lambda k: '1' if k == 0 else label_index2name(k), guessed))
        fen = '/'.join([''.join(piece_names[i * 8:(i + 1) * 8]) for i in reversed(range(8))])
        return fen, tile_certainties

    # Wrapper for chessbot
    def make_prediction(self, url):
        """Try and return a FEN prediction and certainty for URL, return Nones otherwise"""
        img, url = helper_image_loading.loadImageFromURL(url, max_size_bytes=2000000)
        result = [None, None, None]

        # Exit on failure to load image
        if img is None:
            print(f"Couldn't load URL: {url}")
            return result

        # Resize image if too large
        img = helper_image_loading.resizeAsNeeded(img)

        # Exit on failure if image was too large teo resize
        if img is None:
            print('Image too large to resize: "%s"' % url)
            return result

        # Look for chessboard in image, get corners and split chessboard into tiles
        tiles, corners = chessboard_finder.findGrayscaleTilesInImage(img)

        # Exit on failure to find chessboard in image
        if tiles is None:
            print("Couldn't find chessboard in image")
            return result

        # Make prediction on input tiles
        fen, tile_certainties = self.get_prediction(tiles)

        # Use the worst case certainty as our final uncertainty score
        certainty = tile_certainties.min()

        # Get visualize link
        visualize_link = helper_image_loading.getVisualizeLink(corners, url)

        # Update result and return
        result = [fen, certainty, visualize_link]
        return result

    def close(self):
        print("Closing session.")
        self.sess.close()


class GUI(threading.Thread):
    """GUI for chessboard prediction"""

    def __init__(self):
        super().__init__()
        self.brightness_label = None
        self.rotation_slider = None
        self.eval_slider = None
        self.status = None
        self.y_label = None
        self.size_label = None
        self.x_label = None
        self.crop_size = None
        self.preview_cropped = None
        self.crop_y_slider = None
        self.crop_x_slider = None
        self.crop_size_slider = None
        self.crop_y = None
        self.crop_x = None
        self.root = None
        self.preview_full = None

        self.active = True
        self.start()

    def callback(self):
        self.active = False
        self.root.quit()

        sys.exit(0)

    def is_active(self):
        return self.active

    def run(self):
        self.root = CTk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        self.preview_full = CTkLabel(
            self.root, width=200, height=200,
            image=CTkImage(
                PIL.Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)),
                size=(200, 200)
            ),
            text="",
        )
        self.preview_full.grid(row=0, column=1)
        self.preview_cropped = CTkLabel(
            self.root, width=200, height=200,
            image=CTkImage(
                PIL.Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)),
                size=(200, 200)
            ),
            text="",
        )
        self.preview_cropped.grid(row=0, column=3)

        self.status = CTkLabel(self.root, text="...")
        self.status.grid(row=1, column=3)

        # cropping controls
        # (as a percentage of the image)

        # cropping controls
        self.crop_size_slider = CTkSlider(
            self.root, from_=0, to=100,
            orientation=HORIZONTAL, number_of_steps=20)
        self.crop_size_slider.set(80)
        self.crop_size_slider.grid(row=2, column=1)
        self.rotation_slider = CTkSlider(
            self.root, from_=0, to=180,
            orientation=HORIZONTAL, number_of_steps=45)
        self.rotation_slider.set(30)
        self.rotation_slider.grid(row=2, column=3)

        self.crop_x_slider = CTkSlider(
            self.root, from_=-100, to=100,
            orientation=HORIZONTAL, number_of_steps=40)
        self.crop_x_slider.set(0)
        self.crop_x_slider.grid(row=3, column=1)
        self.crop_y_slider = CTkSlider(
            self.root, from_=-100, to=100,
            orientation=VERTICAL, number_of_steps=40)
        self.crop_y_slider.set(0)
        self.crop_y_slider.grid(row=0, column=0)

        # labels for cropping controls
        self.y_label = CTkLabel(self.root, text="Y%")
        self.y_label.grid(row=1, column=0)
        self.size_label = CTkLabel(self.root, text="Size")
        self.size_label.grid(row=2, column=2)
        self.x_label = CTkLabel(self.root, text="X%")
        self.x_label.grid(row=3, column=2)
        self.brightness_label = CTkLabel(self.root, text="Bgt%")
        self.brightness_label.grid(row=1, column=4)

        # eval (pos is better for white, neg is better for black)
        self.eval_slider = CTkSlider(self.root, from_=-1000, to=1000,
                                     fg_color="black", progress_color="white",
                                     orientation=VERTICAL)
        self.eval_slider.set(0)
        self.eval_slider.configure(state="disabled")
        self.eval_slider.grid(row=0, column=4)

        # set the default window size
        self.root.geometry("600x300")

        # load tksvg
        tksvg.load(self.root)

        # start the update loop
        self.root.after(100, self.update)

        self.root.mainloop()

    def crop(self, img: np.ndarray):
        if not self.is_active() or any((
                img is None,
                len(img.shape) != 3,
                img.shape[2] != 3,
                self.crop_size_slider is None,
                self.crop_x_slider is None,
                self.crop_y_slider is None,
        )):
            print(
                f"Not active or missing sliders: Active: {self.is_active()}, img: {img is None}, img.shape: {img.shape}")
            print(
                f"Sliders: crop_size: {self.crop_size_slider is None}, crop_x: {self.crop_x_slider is None}, crop_y: {self.crop_y_slider is None}")
            return img

        height, width, _ = img.shape

        # percentage of image to crop
        self.crop_size = bind(self.crop_size_slider.get(), 1, 100)  # 1<->100
        self.crop_x = self.crop_x_slider.get()  # -100<->100
        self.crop_y = self.crop_y_slider.get()  # -100<->100

        # get crop parameters
        crop_width = int(self.crop_size / 100 * min(width, height))
        crop_height = crop_width

        # the center of the image is (width // 2, height // 2)
        crop_x = int((width - crop_width) * (self.crop_x / 100 / 2 + 0.5) + crop_width / 2)
        crop_y = int((height - crop_height) * (-self.crop_y / 100 / 2 + 0.5) + crop_height / 2)

        if not any((
                self.x_label is None,
                self.y_label is None,
                self.size_label is None,
        )):
            self.x_label.configure(text=f"X%: {self.crop_x}")
            self.y_label.configure(text=f"Y%: {self.crop_y}")
            self.size_label.configure(text=f"Size: {self.crop_size}%")
        else:
            print(f"Labels: x: {type(self.x_label)}, y: {type(self.y_label)}, size: {type(self.size_label)}")

        # crop image
        left = crop_x - crop_width // 2
        right = left + crop_width
        top = crop_y - crop_height // 2
        bottom = top + crop_height
        img = img[top:bottom, left:right]

        return img

    def rotate(self, img: np.ndarray):
        if not self.is_active() or any((
                img is None,
                len(img.shape) != 3,
                img.shape[2] != 3,
                self.rotation_slider is None,
        )):
            return img

        # rotate image
        max_rotation = self.rotation_slider.get()  # 0<->180 (-90<->90) degrees

        try:
            angle, out_img = auto_rotate(img, crop=True)
            print(f"Rotating by {angle} degrees")
        except Exception as e:
            print(f"Exception in rotate: {e}")
            return img
        # angle = bind(angle, -max_rotation // 2, max_rotation // 2)
        # img = Image.fromarray(img).rotate(angle, expand=True)
        # img = np.array(img)

        return out_img

    def update(self):
        if not self.is_active():
            return

        try:
            self.update_preview()
        except Exception:
            print("Exception in update loop: %s" % traceback.format_exc())
        finally:
            self.root.after(100, self.update)

    def update_preview(self):
        global cache

        if not self.is_active() or cache is None:
            return

        best_move = cache["best_moves"]
        img = cache["img"]
        cropped = cache["cropped"]
        fps = cache["fps"]
        certainty = cache["certainty"]

        if best_move is not None and best_move != (None, None):
            w_move, b_move = best_move
            w_uci_move = w_move.move.uci() if w_move is not None and w_move is not None else "..."
            b_uci_move = b_move.move.uci() if b_move is not None and b_move.move is not None else "..."

            w_eval = w_move.info["score"].white().score(mate_score=1000000) \
                if w_move is not None else 0
            b_eval = (b_move.info["score"].white().score(mate_score=1000000)
                      if b_move is not None else 0) * -1

            a_eval = bind((w_eval - b_eval) / 2, min_val=-1000, max_val=1000)

            self.eval_slider.set(a_eval)

            # if there is a forced mate in n, then the eval is M1000000 - n
            if w_eval >= 999950:
                w_eval = f"M{1000000 - w_eval}"
            elif w_eval == 0:
                w_eval = "Resign"
            elif w_eval <= -999950:
                w_eval = f"L{1000000 + w_eval}"

            if b_eval >= 999950:
                b_eval = f"M{1000000 - b_eval}"
            elif b_eval == 0:
                b_eval = "Resign"
            elif b_eval <= -999950:
                b_eval = f"L{1000000 + b_eval}"

            if b_eval[0] in "ML" or w_eval[0] in "ML":
                self.status.configure(text_color="yellow")

            self.status.configure(
                text=f"{f'FPS: {round(fps, 0)}, ' if fps is not None else ''}"
                     "Best moves:\n"
                     f"W: {w_uci_move}, {w_eval}\n"
                     f"B: {b_uci_move}, {b_eval}"
            )

        if cropped is not None and isinstance(cropped, np.ndarray):
            cropped = Image.fromarray(cropped)
            # resize to 200 a tall but preserve aspect ratio
            cropped = helper_image_loading.resizeAsNeeded(cropped, max_size=(200, 200), max_fail_size=(3000, 3000))
            self.preview_cropped.configure(image=CTkImage(cropped, size=cropped.size))
        elif cropped is not None and isinstance(cropped, PhotoImage):
            self.preview_cropped.configure(image=cropped)
        elif cropped is not None and isinstance(cropped, str):
            self.render_svg(cropped)

        if img is not None and self.crop_size is not None:
            # resize to 200 a tall but preserve aspect ratio
            img = Image.fromarray(img)
            width, height = img.size

            # draw a box where the crop will be
            crop_w = int(self.crop_size / 100 * min(width, height))
            crop_h = int(self.crop_size / 100 * min(width, height))
            crop_x = int((width - crop_w) * (self.crop_x / 100 / 2 + 0.5) + crop_w / 2)
            crop_y = int((height - crop_h) * (self.crop_y * -1 / 100 / 2 + 0.5) + crop_h / 2)
            tl = ((crop_x - crop_w // 2), (crop_y - crop_h // 2))
            br = ((crop_x + crop_w // 2), (crop_y + crop_h // 2))
            draw = ImageDraw.Draw(img)
            draw.rectangle([tl, br], outline=(255, 0, 0), width=4)
            img: Image = helper_image_loading.resizeAsNeeded(img, max_size=(200, 200), max_fail_size=(3000, 3000))

            # convert to tkinter image
            img = CTkImage(img, size=img.size)

            # update preview
            self.preview_full.configure(image=img)

            # keep reference to prevent garbage collection
            self.preview_full.image = img

        # update window title
        if certainty is not None:
            self.root.title("Chessboard Predictor - Certainty: %.1f%%" % certainty)

    def render_svg(self, img: str):
        if not self.is_active():
            return

        try:
            print("Rendering SVG")

            img2 = tksvg.SvgImage(data=img, width=200, height=200, name="svg")

            # update preview
            self.preview_cropped.configure(image=img2)

            # keep reference to prevent garbage collection
            self.preview_cropped.image = img2
        except Exception:
            print(f"Exception in render_svg: {traceback.format_exc()}")


class FloatSpinbox(CTkFrame):
    def __init__(self, *args,
                 width: int = 100,
                 height: int = 32,
                 step_size: Union[int, float] = 1,
                 command: Callable = None,
                 bounds: tuple[Union[int, float], Union[int, float]] = (0, 100),
                 **kwargs):
        super().__init__(*args, width=width, height=height, **kwargs)

        self.step_size = step_size
        self.command = command
        self.bounds = bounds

        self.configure(fg_color=("gray78", "gray28"))  # set frame color

        self.grid_columnconfigure((0, 2), weight=0)  # buttons don't expand
        self.grid_columnconfigure(1, weight=1)  # entry expands

        self.subtract_button = CTkButton(self, text="-", width=height - 6, height=height - 6,
                                         command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=0, padx=(3, 0), pady=3)

        self.entry = CTkEntry(self, width=width - (2 * height), height=height - 6, border_width=0)
        self.entry.grid(row=0, column=1, columnspan=1, padx=3, pady=3, sticky="ew")

        self.add_button = CTkButton(self, text="+", width=height - 6, height=height - 6,
                                    command=self.add_button_callback)
        self.add_button.grid(row=0, column=2, padx=(0, 3), pady=3)

        # default value
        self.entry.insert(0, "0.0")

    def add_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            if self.bounds[1] is not None:
                value = min(float(self.entry.get()) + self.step_size, self.bounds[1])
            else:
                value = float(self.entry.get()) + self.step_size
            self.entry.delete(0, "end")
            self.entry.insert(0, value)
        except ValueError:
            return

    def subtract_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            if self.bounds[0] is not None:
                value = max(float(self.entry.get()) - self.step_size, self.bounds[0])
            else:
                value = float(self.entry.get()) - self.step_size
            self.entry.delete(0, "end")
            self.entry.insert(0, value)
        except ValueError:
            return

    def get(self) -> Union[float, None]:
        try:
            return float(self.entry.get())
        except ValueError:
            return None

    def set(self, value: float):
        self.entry.delete(0, "end")
        self.entry.insert(0, str(float(value)))


class Cache(dict):
    slots = ("changes", "__default", "__default_keys")
    changes = set()
    _default = dict()
    _default_keys = set()

    def __init__(self, default: dict[str, any]):
        self._default = copy.deepcopy(default)
        self._default_keys = set(default.keys())
        super().__init__(copy.deepcopy(default))
        self.changes = set()

    def __setitem__(self, key, value):
        if key in self._default_keys:
            super().__setitem__(key, value)
            self.changes.add(key)
        else:
            raise ValueError(f"Key {key} not in default dict: {self._default_keys}")

    def __delitem__(self, key):
        super().__delitem__(key)
        self.changes.add(key)

    def clear(self, keys: tuple = None):
        if keys is None:
            super().clear()
            self.changes.clear()
        else:
            for key in keys:
                self.__setitem__(key, copy.deepcopy(self._default[key]))

    def clear_changes(self):
        self.changes.clear()

    def reset(self):
        self.clear()
        self.update(self._default)
        self.changes.clear()


###########################################################
# MAIN FUNCTION


def stream():
    # Selecting the correct game window
    try:
        video_game_windows = pygetwindow.getAllWindows()
        titles = [window.title.lower() for window in video_game_windows]

        triggers = ["vrchat", "chess.com"]

        if any([trigger.lower() in title.lower() for trigger in triggers for title in titles]):
            # video_game_window = pygetwindow.getWindowsWithTitle("VRChat" if "VRChat" in titles else "Chess.com")[0]
            video_game_window = pygetwindow.getWindowsWithTitle("VRChat")[0] if "VRChat" in titles else \
                pygetwindow.getWindowsWithTitle("Chess.com")[0]
        else:
            print("=== All Windows ===")
            for index, window in enumerate(video_game_windows):
                # only output the window if it has a meaningful title
                if window.title:
                    print("[{}]: {}".format(index, window.title))
            # have the user select the window they want
            try:
                user_input = int(input(
                    "Please enter the number corresponding to the window you'd like to select: "))
            except ValueError:
                print("You didn't enter a valid number. Please try again.")
                return
            # "save" that window as the chosen window for the rest of the script
            video_game_window: pygetwindow.Window = video_game_windows[user_input]
    except Exception as err:
        print("Failed to select game window: {}".format(err))
        return

    # Activate that Window
    activation_retries = 30
    activation_success = False
    while activation_retries > 0:
        try:
            video_game_window.activate()
            activation_success = True
            break
        except pygetwindow.PyGetWindowException as we:
            print("Failed to activate game window: {}".format(str(we)))
            print("Trying again... (you should switch to the game now)")
        except Exception as err:
            print("Failed to activate game window: {}".format(str(err)))
            print(
                "Read the relevant restrictions here: "
                "https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow")
            activation_success = False
            break
        # wait a little bit before the next try
        time.sleep(1.0)
        activation_retries = activation_retries - 1
    # if we failed to activate the window, then we'll be unable to send input to it,
    # so exits the script now
    if not activation_success:
        return
    print("Successfully activated the game window...")

    # get the window region
    rect = video_game_window._getWindowRect()
    region = {
        "top": rect.top,
        "left": rect.left,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top,
    }

    with mss.mss() as camera:
        if camera is None:
            print("""DXCamera failed to initialize. Some common causes are:
                1. You are on a laptop with both an integrated GPU and discrete GPU.
                   Go into Windows Graphic Settings,select python.exe and set it to Power Saving Mode.
                   If that doesn't work, then read this:
                   https://github.com/SerpentAI/D3DShot/wiki/Installation-Note:-Laptops
                2. The game is an exclusive full screen game. Set it to windowed mode.""")
            return

        # Initialize predictor, takes a while, but only needed once
        predictor = ChessboardPredictor()
        engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")

        time.sleep(1.0)

        w_board = chess.Board()
        b_board = chess.Board()
        default_cache = {
            "svg": None,
            "img": None,
            "cropped": None,

            "fen": None,

            "best_w": set(),
            "best_b": set(),

            "exp_w": set(),
            "exp_b": set(),

            "best_moves": (None, None),

            "eval_w": None,
            "eval_b": None,

            "fills": set(),

            "ars_w": None,
            "ars_b": None,

            "fps": None,
            "certainty": None,
        }
        global cache
        cache = Cache(default_cache)

        gui = GUI()

        while gui.is_active():
            t_start = time.perf_counter()
            if not gui.is_active():
                break

            # ---
            # Get the frame and do some preprocessing
            src = np.array(camera.grab(region))

            # remove the alpha channel
            src = src[:, :, :3]
            # reverse the order of the channels
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

            if src is None:
                print("Failed to grab frame")
                continue

            # crop frame
            if gui is not None:
                frame = gui.crop(src)
            else:
                continue

            # check if the difference between the current frame and the last frame is significant
            # (around 4 pixels changing from black to white)
            if cache["img"] is None or src.shape != cache["img"].shape or np.sum(np.abs(cache["img"] - src)) > 1000:
                # rotate the frame
                frame = gui.rotate(frame)
                cache["img"] = src
            print(f"{round(time.perf_counter() - t_start, 3) * 1000}ms Frame processed")
            # ---

            # ---
            # Find the corners and tiles of the chessboard
            # Look for chessboard in image, get corners and split chessboard into tiles
            t_tmp = time.perf_counter()
            try:
                tiles, corners = chessboard_finder.findGrayscaleTilesInImage(frame)
            except ValueError:
                print("Couldn't parse chessboard")
                continue

            # skip on failure to find chessboard in image
            if tiles is None or len(tiles) == 0:
                print("Couldn't find chessboard in image, updating window rect")

                rect = video_game_window._getWindowRect()
                region = {
                    "top": rect.top,
                    "left": rect.left,
                    "width": rect.right - rect.left,
                    "height": rect.bottom - rect.top,
                }

                cache["cropped"] = frame
                continue

            tl = (corners[0], corners[1])
            br = (corners[2], corners[3])
            cropped = frame[tl[1]:br[1], tl[0]:br[0]]
            # ---

            # ---
            # Make prediction on input tiles to find pieces
            fen, tile_certainties = predictor.get_prediction(tiles)
            print(f"{round(time.perf_counter() - t_tmp, 3) * 1000}ms Fen made")
            short_fen: str = shortenFEN(fen)

            # Use the worst case certainty as our final uncertainty score
            certainty = tile_certainties.min()

            # Has the board changed?
            if short_fen != cache["fen"]:
                # the board has changed
                w_fen = short_fen + " w - - 0 1"
                b_fen = short_fen + " b - - 0 1"

                w_board = chess.Board(w_fen)
                b_board = chess.Board(b_fen)

                # change the cache
                cache["fen"] = short_fen
                cache.clear((
                    "best_w", "best_b",
                    "exp_w", "exp_b",
                    "ars_w", "ars_b",
                    "eval_w", "eval_b",
                    "svg", "fills",
                ))
            # ---

            # ---
            # Update the SVG and find the best move for black and white
            best_move_w = None
            best_move_b = None
            if any((w_board.is_game_over(), w_board.is_checkmate(), w_board.is_stalemate())):
                print("Game over")
            else:
                # The board has changed

                t_tmp = time.perf_counter()
                w_status = w_board.status()
                b_status = b_board.status()

                if not any((w_status == chess.STATUS_VALID, b_status == chess.STATUS_VALID)):
                    print("Boards are invalid")
                    print(w_board, w_status)
                    print(b_board, b_status)

                print(f"{round(time.perf_counter() - t_tmp, 3) * 1000}ms Initial board checks")

                t_tmp = time.perf_counter()
                try:
                    if len(cache["best_w"]) < 3 and w_status == chess.STATUS_VALID:
                        best_move_w = engine.play(
                            w_board,
                            chess.engine.Limit(time=(0.1 if len(cache["best_w"]) < 2 else 0.05)),
                            info=chess.engine.INFO_SCORE,
                            ponder=True,
                        )

                        if best_move_w.move is not None:
                            cache["eval_w"] = best_move_w.info["score"]
                            if best_move_w.move.uci() not in cache["best_w"]:
                                cache["best_w"].update((best_move_w.move.uci(),))
                                cache.changes.update(("best_w",))
                        if best_move_w.ponder is not None:
                            if best_move_w.ponder.uci() not in cache["exp_w"]:
                                cache["exp_w"].update((best_move_w.ponder.uci(),))
                                cache.changes.update(("exp_w",))

                    if len(cache["best_b"]) < 3 and b_status == chess.STATUS_VALID:
                        best_move_b = engine.play(
                            b_board,
                            chess.engine.Limit(time=(0.1 if len(cache["best_b"]) < 2 else 0.05)),
                            info=chess.engine.INFO_SCORE,
                            ponder=True,
                        )

                        if best_move_b.move is not None:
                            cache["eval_b"] = best_move_b.info["score"]
                            if best_move_b.move.uci() not in cache["best_b"]:
                                cache["best_b"].update((best_move_b.move.uci(),))
                                cache.changes.update(("best_b",))
                        if best_move_b.ponder is not None:
                            if best_move_b.ponder.uci() not in cache["exp_b"]:
                                cache["exp_b"].update((best_move_b.ponder.uci(),))
                                cache.changes.update(("exp_b",))
                except chess.engine.EngineTerminatedError:
                    print(f"Stockfish died, fen: {short_fen} board status: {w_status} {b_status}")
                    break
                print(f"{round(time.perf_counter() - t_tmp, 3) * 1000}ms Stockfish eval")

                if cache["svg"] is not None and "best_w" not in cache.changes and "best_b" not in cache.changes:
                    svg = cache["svg"]
                else:
                    # update the SVG

                    t_tmp = time.perf_counter()
                    ars = ()
                    uci_moves = set()
                    exp_ars = ()
                    # check for more good moves
                    if any((_ in cache.changes for _ in ("best_w", "exp_w", "ars_w"))):
                        _ars = ()
                        for move in cache["best_w"]:
                            move = chess.Move.from_uci(move)
                            _ars += (chess.svg.Arrow(move.from_square, move.to_square, color='green'),)
                            uci_moves.add(move.uci())
                        for move in cache["exp_w"]:
                            move = chess.Move.from_uci(move)
                            if move.uci() not in uci_moves:
                                exp_ars += (chess.svg.Arrow(move.from_square, move.to_square, color='#FF5A5A19'),)

                        cache["ars_w"] = _ars
                        ars += _ars
                    elif cache["ars_w"] is not None:
                        ars += cache["ars_w"]

                    if any((_ in cache.changes for _ in ("best_b", "exp_b", "ars_b"))):
                        _ars = ()
                        for move in cache["best_b"]:
                            move = chess.Move.from_uci(move)
                            _ars += (chess.svg.Arrow(move.from_square, move.to_square, color='red'),)
                            uci_moves.add(move.uci())
                        for move in cache["exp_b"]:
                            move = chess.Move.from_uci(move)
                            if move.uci() not in uci_moves:
                                exp_ars += (chess.svg.Arrow(move.from_square, move.to_square, color='#5AFF5A19'),)

                        cache["ars_b"] = _ars
                        ars += _ars
                    elif cache["ars_b"] is not None:
                        ars += cache["ars_b"]

                    for exp in exp_ars:
                        uci = chess.Move(exp.tail, exp.head).uci()
                        if uci not in uci_moves:
                            ars += (exp,)
                            uci_moves.add(uci)

                    print(f"{round(time.perf_counter() - t_tmp, 3) * 1000}ms Arrows made")

                    # only change fills if there is a board change
                    t_tmp = time.perf_counter()
                    if "fills" in cache.changes:
                        fills = {}
                        if w_board.is_check():
                            # White King is in check

                            fills[w_board.king(chess.WHITE)] = 'pink'
                            # get the attacker
                            attackers = w_board.attackers(chess.BLACK, w_board.king(chess.WHITE))
                            for att in attackers:
                                fills[att] = 'red'

                                attacks = w_board.attacks(att)  # att -> attacked square
                                attackers = w_board.attackers(chess.WHITE, att)  # square -> att
                                for attack in attacks:
                                    if not w_board.piece_at(attack) is None \
                                            and w_board.piece_at(attack).color == chess.WHITE \
                                            and w_board.piece_at(attack).piece_type != chess.KING:
                                        fills[attack] = 'yellow'

                                for attacker in attackers:
                                    if not w_board.piece_at(attacker) is None:
                                        fills[attacker] = 'purple'

                        if b_board.is_check():
                            # Black King is in check

                            fills[b_board.king(chess.BLACK)] = 'pink'
                            # get the attacker
                            attackers = b_board.attackers(chess.WHITE, b_board.king(chess.BLACK))
                            for att in attackers:
                                fills[att] = 'green'

                                attacks = b_board.attacks(att)  # att -> attacked square
                                attackers = b_board.attackers(chess.BLACK, att)  # square -> att
                                for attack in attacks:
                                    if not b_board.piece_at(attack) is None \
                                            and b_board.piece_at(attack).color == chess.BLACK \
                                            and b_board.piece_at(attack).piece_type != chess.KING:
                                        fills[attack] = 'yellow'

                                for attacker in attackers:
                                    if not b_board.piece_at(attacker) is None:
                                        fills[attacker] = 'purple'

                        cache["fills"] = fills
                    else:
                        fills = cache["fills"]
                    print(f"{round(time.perf_counter() - t_tmp, 3) * 1000}ms Fills made")

                    t_tmp = time.perf_counter()
                    # create a board with just the arrows and fills
                    svg = chess.svg.board(
                        w_board,
                        arrows=ars,
                        fill=fills,
                        size=200,
                        coordinates=False,
                    )
                    print(f"{round(time.perf_counter() - t_tmp, 3) * 1000}ms SVG generation")
                cropped = svg

            cache["cropped"] = cropped
            # ---

            # ---
            # Update the GUI
            if cache["best_w"] and best_move_w is None:
                best_move_w = chess.engine.PlayResult(
                    move=chess.Move.from_uci(list(cache["best_w"])[0]),
                    ponder=None,
                    info={"score": cache["eval_w"]},
                )

            if cache["best_b"] and best_move_b is None:
                best_move_b = chess.engine.PlayResult(
                    move=chess.Move.from_uci(list(cache["best_b"])[0]),
                    ponder=None,
                    info={"score": cache["eval_b"]},
                )

            t_tmp = time.perf_counter()
            cache["fps"] = 1 / (time.perf_counter() - t_start)
            cache["certainty"] = certainty * 100
            cache["best_moves"] = best_move_w, best_move_b
            print(f"{round(time.perf_counter() - t_tmp, 3) * 1000}ms Final view update")
            cache.clear_changes()
            # ---

            t_end = time.perf_counter()
            print(f"{round(t_end - t_start, 3) * 1000}ms Prediction total ({round(1 / (t_end - t_start), 0)}fps)\n")

    gui.callback()
    predictor.close()


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)

    stream()
