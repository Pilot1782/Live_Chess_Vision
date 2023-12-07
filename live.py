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
import io
import os
import sys
import threading
import time
from io import StringIO

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
    CTkImage,
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
        self.fen = None
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
        self.fen = CTkLabel(self.root)
        self.fen.grid(row=0, column=4)

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
            fen: str = None,
            best_move: tuple[chess.engine.PlayResult] = None,
    ):
        if not self.isActive():
            return

        if best_move is not None:
            w_uci_move = best_move[0].move.uci() if best_move[0] is not None else "..."
            b_uci_move = best_move[1].move.uci() if best_move[1] is not None else "..."
            if w_uci_move == "0000" or b_uci_move == "0000":
                self.status.configure(text="Resign")
            else:
                self.status.configure(text=f"Best moves:\nW: {w_uci_move}\nB:{b_uci_move}")

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

        # update fen
        if fen is not None:
            text = ""
            for row in fen.split(' ')[0].split("/"):
                for char in row:
                    if char.isdigit():
                        for i in range(int(char)):
                            text += "_ "
                    else:
                        text += char + " "
                text += "\n"

            self.fen.configure(text=text)

        # update window title
        if certainty is not None:
            self.root.title("Chessboard Predictor - Certainty: %.1f%%" % certainty)


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
    camera.start(target_fps=15, video_mode=True)
    gui = GUI()

    time.sleep(1.0)

    w_board = chess.Board()
    b_board = chess.Board()
    best_move_w = None
    best_move_b = None
    board_detected = False

    while gui.isActive():
        t_start = time.perf_counter()
        if not gui.isActive():
            break

        src = camera.get_latest_frame()

        # crop frame
        frame = gui.crop(src)
        if board_detected:
            gui.updatePreview(img=src)
        else:
            gui.updatePreview(img=src, cropped=frame)

        # Look for chessboard in image, get corners and split chessboard into tiles
        tiles, corners = chessboard_finder.findGrayscaleTilesInImage(frame)

        # Exit on failure to find chessboard in image
        if tiles is None or len(tiles) == 0:
            print("Couldn't find chessboard in image")
            board_detected = False
            continue
        else:
            board_detected = True
            tl = (corners[0], corners[1])
            br = (corners[2], corners[3])
            cropped = frame[tl[1]:br[1], tl[0]:br[0]]

        fen, tile_certainties = predictor.getPrediction(tiles)
        short_fen = shortenFEN(fen)

        # Use the worst case certainty as our final uncertainty score
        certainty = tile_certainties.min()

        if w_board.fen() != short_fen + " w - - 0 1":
            w_fen = short_fen + " w - - 0 1"
            b_fen = short_fen + " b - - 0 1"

            w_board = chess.Board(w_fen)
            b_board = chess.Board(b_fen)

        if any((w_board.is_game_over(), w_board.is_checkmate(), w_board.is_stalemate())):
            print("Game over")
        else:
            w_status = w_board.status()
            b_status = b_board.status()

            if not any((w_status == chess.STATUS_VALID, b_status == chess.STATUS_VALID)):
                print("Boards are invalid")
                print(w_board)
                print(b_board)

            fills = {}
            if w_board.is_check():
                fills[w_board.king(chess.WHITE)] = 'pink'
                # get the attacker
                attackers = w_board.attackers(chess.BLACK, w_board.king(chess.WHITE))
                atts = []
                for att in attackers:
                    fills[att] = 'red'
                    atts.append(att)

                for att in atts:
                    attacks = w_board.attacks(att)
                    for attack in attacks:
                        if not w_board.piece_at(attack) is None and w_board.piece_at(
                                attack).color == chess.WHITE and w_board.piece_at(attack).piece_type != chess.KING:
                            fills[attack] = 'yellow'

            if b_board.is_check():
                fills[b_board.king(chess.BLACK)] = 'pink'
                # get the attacker
                attackers = b_board.attackers(chess.WHITE, b_board.king(chess.BLACK))
                atts = []
                for att in attackers:
                    fills[att] = 'green'
                    atts.append(att)

                for att in atts:
                    attacks = b_board.attacks(att)
                    for attack in attacks:
                        if not b_board.piece_at(attack) is None and b_board.piece_at(
                                attack).color == chess.BLACK and b_board.piece_at(attack).piece_type != chess.KING:
                            fills[attack] = 'yellow'

            try:
                best_move_w = engine.play(w_board, chess.engine.Limit(time=0.1),
                                          ponder=True) if w_status == chess.STATUS_VALID else None
                best_move_b = engine.play(b_board, chess.engine.Limit(time=0.1),
                                          ponder=True) if b_status == chess.STATUS_VALID else None
            except chess.engine.EngineTerminatedError:
                print(f"Stockfish died, fen: {short_fen}")
                break

            ars = []
            if best_move_w is not None:
                ars.append(chess.svg.Arrow(best_move_w.move.from_square, best_move_w.move.to_square, color='green'))
            if best_move_b is not None:
                ars.append(chess.svg.Arrow(best_move_b.move.from_square, best_move_b.move.to_square, color='red'))

            svg = chess.svg.board(
                w_board,
                arrows=ars,
                fill=fills,
                size=200,
            )
            svg = svg2rlg(StringIO(svg))
            png = io.BytesIO()
            renderPM.drawToFile(svg, png, fmt="PNG")
            png.seek(0)
            svg = np.array(Image.open(png))
            cropped = svg

        gui.updatePreview(src, cropped, certainty * 100, short_fen, (best_move_w, best_move_b))

        t_end = time.perf_counter()

        print("Prediction took %g seconds" % (t_end - t_start))

    predictor.close()
    gui.callback()


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)

    stream()
