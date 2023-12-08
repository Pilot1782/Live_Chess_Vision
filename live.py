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
import os
import sys
import threading
import time
from io import StringIO
from typing import Union, Callable

import PIL.Image
import PIL.ImageTk
import chess
import chess.engine
import chess.svg
import dxcam
import pygetwindow
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Ignore Tensorflow INFO debug messages
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from customtkinter import (
    CTk,
    CTkLabel,
    CTkSlider,
    HORIZONTAL,
    VERTICAL,
    CTkImage, CTkButton, CTkEntry, CTkFrame,
)

from helper_functions import shortenFEN
import helper_image_loading
import chessboard_finder


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

    def getPrediction(self, tiles):
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
        def labelIndex2Name(label_index):
            return ' ' if label_index == 0 else ' KQRBNPkqrbnp'[label_index]

        pieceNames = list(map(lambda k: '1' if k == 0 else labelIndex2Name(k), guessed))  # exchange ' ' for '1' for FEN
        fen = '/'.join([''.join(pieceNames[i * 8:(i + 1) * 8]) for i in reversed(range(8))])
        return fen, tile_certainties

    # Wrapper for chessbot
    def makePrediction(self, url):
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
        fen, tile_certainties = self.getPrediction(tiles)

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

        self.is_active = True
        self.start()

    def callback(self):
        self.is_active = False
        self.root.quit()

        sys.exit(0)

    def isActive(self):
        return self.is_active

    def run(self):
        self.root = CTk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        # layout:
        """
        | Y% Slider | Full preview | | | Cropped preview | Eval slider |
        |           |              | | | Status          |             |
        """

        self.preview_full = CTkLabel(self.root, width=200, height=200,
                                     image=CTkImage(
                                         PIL.Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)),
                                         size=(200, 200)
                                     ),
                                     text="",
                                     )
        self.preview_full.grid(row=0, column=1)
        self.preview_cropped = CTkLabel(self.root, width=200, height=200,
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
        self.crop_size_slider = CTkSlider(self.root, from_=1, to=100,
                                          orientation=HORIZONTAL, number_of_steps=99)
        self.crop_size_slider.set(100)
        self.crop_size_slider.grid(row=2, column=1)

        self.crop_x_slider = CTkSlider(self.root, from_=-100, to=100,
                                       orientation=HORIZONTAL, number_of_steps=100)
        self.crop_x_slider.set(0)
        self.crop_x_slider.grid(row=3, column=1)
        self.crop_y_slider = CTkSlider(self.root, from_=-100, to=100,
                                       orientation=VERTICAL, number_of_steps=100)
        self.crop_y_slider.set(0)
        self.crop_y_slider.grid(row=0, column=0)

        # labels for cropping controls
        self.y_label = CTkLabel(self.root, text="Y%")
        self.y_label.grid(row=1, column=0)
        self.size_label = CTkLabel(self.root, text="Size")
        self.size_label.grid(row=2, column=2)
        self.x_label = CTkLabel(self.root, text="X%")
        self.x_label.grid(row=3, column=2)

        # eval (pos is better for white, neg is better for black)
        self.eval_slider = CTkSlider(self.root, from_=-1000, to=1000, fg_color="black", progress_color="white",
                                     orientation=VERTICAL)
        self.eval_slider.set(0)
        self.eval_slider.configure(state="disabled")
        self.eval_slider.grid(row=0, column=4)

        # set the default window size
        self.root.geometry("600x300")

        self.root.mainloop()

    def crop(self, img: np.ndarray):
        if not self.isActive():
            return img

        height, width, _ = img.shape

        # percentage of image to crop
        self.crop_size = self.crop_size_slider.get()  # 1<->100
        self.crop_x = self.crop_x_slider.get()  # -100<->100
        self.crop_y = self.crop_y_slider.get()  # -100<->100

        # get crop parameters
        crop_width = int(self.crop_size / 100 * min(width, height))
        crop_height = int(self.crop_size / 100 * min(width, height))

        # the center of the image is (width // 2, height // 2)
        crop_x = int((width - crop_width) * (self.crop_x / 100 / 2 + 0.5) + crop_width / 2)
        crop_y = int((height - crop_height) * (self.crop_y * -1 / 100 / 2 + 0.5) + crop_height / 2)

        self.x_label.configure(text=f"X%: {self.crop_x}")
        self.y_label.configure(text=f"Y%: {self.crop_y}")
        self.size_label.configure(text=f"Size: {self.crop_size}%")

        # crop image
        left = crop_x - crop_width // 2
        right = left + crop_width
        top = crop_y - crop_height // 2
        bottom = top + crop_height
        img = img[top:bottom, left:right]

        return img

    def updatePreview(
            self,
            img: np.ndarray,
            cropped: np.ndarray = None,
            certainty: float = None,
            best_move: tuple[chess.engine.PlayResult] = None,
            fps: float = None,
    ):
        if not self.isActive():
            return

        if best_move is not None:
            w_uci_move = best_move[0].move.uci() if best_move[0] is not None and best_move[0] is not None else "..."
            b_uci_move = best_move[1].move.uci() if best_move[1] is not None and best_move[
                1].move is not None else "..."

            w_eval = best_move[0].info["score"].white().score(mate_score=1000000) \
                if best_move[0] is not None else 0
            b_eval = (best_move[1].info["score"].white().score(mate_score=1000000)
                      if best_move[1] is not None else 0) * -1

            a_eval = bind((w_eval - b_eval) / 2, min_val=-1000, max_val=1000)

            self.eval_slider.set(a_eval)

            # if there is a forced mate in n, then the eval is M1000000 - n
            if w_eval >= 999950:
                w_eval = f"M{1000000 - w_eval}"
            if b_eval >= 999950:
                b_eval = f"M{1000000 - b_eval}"

            if w_uci_move == 0 or b_uci_move == 0:
                self.status.configure(
                    text="Resign (" + (
                        f'B: {b_uci_move}' if w_uci_move == 0 else f'W: {w_uci_move}'
                    ))
            else:
                self.status.configure(
                    text=f"{f'FPS: {round(fps, 0)}, ' if fps is not None else ''}"
                         "Best moves:\n"
                         f"W: {w_uci_move}, {w_eval}\n"
                         f"B: {b_uci_move}, {b_eval}")

        if cropped is not None:
            cropped = Image.fromarray(cropped)
            # resize to 200 a tall but preserve aspect ratio
            cropped = helper_image_loading.resizeAsNeeded(cropped, max_size=(200, 200), max_fail_size=(3000, 3000))
            self.preview_cropped.configure(image=CTkImage(cropped, size=cropped.size))

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


###########################################################
# MAIN FUNCTION


def stream():
    # Selecting the correct game window
    try:
        videoGameWindows = pygetwindow.getAllWindows()
        titles = [window.title for window in videoGameWindows]

        if "VRChat" in titles:
            videoGameWindow = pygetwindow.getWindowsWithTitle("VRChat")[0]
        else:
            print("=== All Windows ===")
            for index, window in enumerate(videoGameWindows):
                # only output the window if it has a meaningful title
                if window.title:
                    print("[{}]: {}".format(index, window.title))
            # have the user select the window they want
            try:
                userInput = int(input(
                    "Please enter the number corresponding to the window you'd like to select: "))
            except ValueError:
                print("You didn't enter a valid number. Please try again.")
                return
            # "save" that window as the chosen window for the rest of the script
            videoGameWindow: pygetwindow.Window = videoGameWindows[userInput]
    except Exception as err:
        print("Failed to select game window: {}".format(err))
        return

    # Activate that Window
    activationRetries = 30
    activationSuccess = False
    while activationRetries > 0:
        try:
            videoGameWindow.activate()
            activationSuccess = True
            break
        except pygetwindow.PyGetWindowException as we:
            print("Failed to activate game window: {}".format(str(we)))
            print("Trying again... (you should switch to the game now)")
        except Exception as err:
            print("Failed to activate game window: {}".format(str(err)))
            print(
                "Read the relevant restrictions here: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow")
            activationSuccess = False
            break
        # wait a little bit before the next try
        time.sleep(1.0)
        activationRetries = activationRetries - 1
    # if we failed to activate the window, then we'll be unable to send input to it,
    # so exits the script now
    if not activationSuccess:
        return
    print("Successfully activated the game window...")

    # get the window region
    rect = videoGameWindow._getWindowRect()
    region = (rect.left, rect.top, rect.right, rect.bottom)

    camera = dxcam.create(device_idx=0, region=region)
    if camera is None:
        print("""DXCamera failed to initialize. Some common causes are:
            1. You are on a laptop with both an integrated GPU and discrete GPU. Go into Windows Graphic Settings, select python.exe and set it to Power Saving Mode.
             If that doesn't work, then read this: https://github.com/SerpentAI/D3DShot/wiki/Installation-Note:-Laptops
            2. The game is an exclusive full screen game. Set it to windowed mode.""")
        return

    # Initialize predictor, takes a while, but only needed once
    predictor = ChessboardPredictor()
    engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
    camera.start(target_fps=10, video_mode=True)
    gui = GUI()

    time.sleep(1.0)

    w_board = chess.Board()
    b_board = chess.Board()
    board_detected = False
    """
    cache = {
        "svg": PIL object of the svg
        "changes": set of changes to the svg
        
        "best_w": set of best moves for white as uci strings
        "best_b": set of best moves for black as uci strings
        "fills": set of fills for the svg
        "ars_w": set of arrows for white
        "ars_b": set of arrows for black
    """
    __cache = {
        "svg": None,
        "changes": set(),

        "best_w": set(),
        "best_b": set(),
        "eval_w": None,
        "eval_b": None,
        "fills": set(),
        "ars_w": set(),
        "ars_b": set(),
    }

    while gui.isActive():
        t_start = time.perf_counter()
        t_tmp = t_start
        if not gui.isActive():
            break

        src = camera.get_latest_frame()

        # crop frame
        frame = gui.crop(src)
        if board_detected:
            gui.updatePreview(img=src)
        else:
            gui.updatePreview(img=src, cropped=frame)
        print(f"{round(time.perf_counter() - t_tmp, 3)}s Initial view update")

        t_tmp = time.perf_counter()
        # Look for chessboard in image, get corners and split chessboard into tiles
        try:
            tiles, corners = chessboard_finder.findGrayscaleTilesInImage(frame)
        except ValueError:
            print("Couldn't parse chessboard")
            board_detected = False
            continue

        # Exit on failure to find chessboard in image
        if tiles is None or len(tiles) == 0:
            print("Couldn't find chessboard in image")
            board_detected = False
            continue

        board_detected = True
        tl = (corners[0], corners[1])
        br = (corners[2], corners[3])
        cropped = frame[tl[1]:br[1], tl[0]:br[0]]

        fen, tile_certainties = predictor.getPrediction(tiles)
        print(f"{round(time.perf_counter() - t_tmp, 3)}s Fen made")
        short_fen = shortenFEN(fen)

        # Use the worst case certainty as our final uncertainty score
        certainty = tile_certainties.min()

        # Has the board changed?
        if w_board.fen() != short_fen + " w - - 0 1":
            # the board has changed
            w_fen = short_fen + " w - - 0 1"
            b_fen = short_fen + " b - - 0 1"

            w_board = chess.Board(w_fen)
            b_board = chess.Board(b_fen)

            __cache = {
                "svg": None,
                "changes": {"best_w", "best_b", "fills", "ars_w", "ars_b"},

                "best_w": set(),
                "best_b": set(),
                "fills": set(),
                "ars_w": set(),
                "ars_b": set(),
            }

        best_move_w = None
        best_move_b = None
        if any((w_board.is_game_over(), w_board.is_checkmate(), w_board.is_stalemate())):
            print("Game over")
        else:
            t_tmp = time.perf_counter()
            w_status = w_board.status()
            b_status = b_board.status()

            if not any((w_status == chess.STATUS_VALID, b_status == chess.STATUS_VALID)):
                print("Boards are invalid")
                print(w_board, w_status)
                print(b_board, b_status)

            print(f"{round(time.perf_counter() - t_tmp, 3)}s Initial board checks")

            t_tmp = time.perf_counter()
            try:
                if len(__cache["best_w"]) < 3 and w_status == chess.STATUS_VALID:
                    best_move_w = engine.play(w_board, chess.engine.Limit(time=0.1),
                                              info=chess.engine.INFO_SCORE)
                    if best_move_w.move is not None:
                        __cache["eval_w"] = best_move_w.info["score"]
                        if best_move_w.move.uci() not in __cache["best_w"]:
                            __cache["best_w"].update((best_move_w.move.uci(),))
                            __cache["changes"].update(("best_w",))

                if len(__cache["best_b"]) < 3 and b_status == chess.STATUS_VALID:
                    best_move_b = engine.play(b_board, chess.engine.Limit(time=0.1),
                                              info=chess.engine.INFO_SCORE)
                    if best_move_b.move is not None:
                        __cache["eval_b"] = best_move_b.info["score"]
                        if best_move_b.move.uci() not in __cache["best_b"]:
                            __cache["best_b"].update((best_move_b.move.uci(),))
                            __cache["changes"].update(("best_b",))
            except chess.engine.EngineTerminatedError:
                print(f"Stockfish died, fen: {short_fen} board status: {w_status} {b_status}")
                break
            print(f"{round(time.perf_counter() - t_tmp, 3)}s Stockfish eval")

            if __cache["svg"] is not None and "best_w" not in __cache["changes"] and "best_b" not in __cache["changes"]:
                svg = __cache["svg"]
            else:
                t_tmp = time.perf_counter()
                ars = []
                # check for more good moves
                if __cache["best_w"] and "best_w" in __cache["changes"]:
                    _ars = []
                    for move in __cache["best_w"]:
                        move = chess.Move.from_uci(move)
                        _ars.append(chess.svg.Arrow(move.from_square, move.to_square, color='green'))

                    __cache["ars_w"] = _ars
                    ars.extend(_ars)
                else:
                    ars.extend(__cache["ars_w"])

                if __cache["best_b"] and "best_b" in __cache["changes"]:
                    _ars = []
                    for move in __cache["best_b"]:
                        move = chess.Move.from_uci(move)
                        _ars.append(chess.svg.Arrow(move.from_square, move.to_square, color='red'))

                    __cache["ars_b"] = _ars
                    ars.extend(_ars)
                else:
                    ars.extend(__cache["ars_b"])
                print(f"{round(time.perf_counter() - t_tmp, 3)}s Arrows made")

                # only change fills if there is a board change
                t_tmp = time.perf_counter()
                if "fills" in __cache["changes"]:
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
                                if not w_board.piece_at(attack) is None and w_board.piece_at(
                                        attack).color == chess.WHITE and w_board.piece_at(attack).piece_type != chess.KING:
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
                                if not b_board.piece_at(attack) is None and b_board.piece_at(
                                        attack).color == chess.BLACK and b_board.piece_at(attack).piece_type != chess.KING:
                                    fills[attack] = 'yellow'

                            for attacker in attackers:
                                if not b_board.piece_at(attacker) is None:
                                    fills[attacker] = 'purple'

                    __cache["fills"] = fills
                else:
                    fills = __cache["fills"]
                print(f"{round(time.perf_counter() - t_tmp, 3)}s Fills made")

                t_tmp = time.perf_counter()
                svg = chess.svg.board(
                    w_board,
                    arrows=ars,
                    fill=fills,
                    size=200,
                    coordinates=False,
                )
                print(f"{round(time.perf_counter() - t_tmp, 3)}s SVG generation")

                t_tmp = time.perf_counter()
                svg = svg2rlg(StringIO(svg))
                print(f"{round(time.perf_counter() - t_tmp, 3)}s SVG to ReportLab")

                t_tmp = time.perf_counter()
                pil_svg = renderPM.drawToPIL(svg)
                print(f"{round(time.perf_counter() - t_tmp, 3)}s SVG rendered to PIL")

                svg = np.array(pil_svg)

                # update cache
                __cache["svg"] = svg
                __cache["changes"].clear()
            cropped = svg

            if __cache["best_w"] and best_move_w is None:
                best_move_w = chess.engine.PlayResult(
                    move=chess.Move.from_uci(list(__cache["best_w"])[0]),
                    ponder=None,
                    info={"score": __cache["eval_w"]},
                )

            if __cache["best_b"] and best_move_b is None:
                best_move_b = chess.engine.PlayResult(
                    move=chess.Move.from_uci(list(__cache["best_b"])[0]),
                    ponder=None,
                    info={"score": __cache["eval_b"]},
                )

        t_tmp = time.perf_counter()
        gui.updatePreview(
            img=src, cropped=cropped,
            certainty=certainty * 100,
            best_move=(best_move_w, best_move_b),
            fps=1 / (time.perf_counter() - t_start),
        )
        print(f"{round(time.perf_counter() - t_tmp, 3)}s Final view update")

        t_end = time.perf_counter()

        print(f"{round(t_end - t_start, 3)}s Prediction total\n")

    predictor.close()
    gui.callback()


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)

    stream()
