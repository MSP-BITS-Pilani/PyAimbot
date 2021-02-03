import numpy as np
from termcolor import colored
import timeit
import _thread
import imutils
import time
import mss
import cv2
import os
import signal
import sys
import pynput
import ctypes
from lib.grab import grab_screen
import torch
import torchvision

sct = mss.mss()
Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def set_pos(x, y):
    x = 1 + int(x * 65536./Wd)
    y = 1 + int(y * 65536./Hd)
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

def mouse_lclick():
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(0, 0, 0, (0x0002 | 0x0004), 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

if __name__ == "__main__":
    print("Do not run this file directly.")

def start(ENABLE_AIMBOT):
    """
    Image Classifier PyTorch model

    The output is a single floating number indicating the probability of a
    person in the field-of-view window
    """

    # Specify a path
    MODEL_PATH = "models/model.pt"
    THRESHOLD = 0.5

    # Load model for inference
    model = torch.load(MODEL_PATH)
    model.eval()

    # Wait for buffering
    time.sleep(0.4)

     # Define screen capture area
    print("[INFO] loading screencapture device...")
    W, H = None, None
    origbox = (int(Wd/2 - ACTIVATION_RANGE/2),
               int(Hd/2 - ACTIVATION_RANGE/2),
               int(Wd/2 + ACTIVATION_RANGE/2),
               int(Hd/2 + ACTIVATION_RANGE/2))

    # Log whether aimbot is enabled
    if not ENABLE_AIMBOT:
        print("[INFO] aimbot disabled, using visualizer only...")
    else:
        print(colored("[OKAY] Aimbot enabled!", "green"))

    # Handle Ctrl+C in terminal, release pointers
    def signal_handler(sig, frame):
        # release the file pointers
        print("\n[INFO] cleaning up...")
        sct.close()
        cv2.destroyAllWindows()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # Test for GPU support
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print(colored("[OKAY] CUDA is working!", "green"))
    else:
        print(
            colored("[WARNING] CUDA acceleration is disabled!", "yellow"))

    model = model.to(device)

    print()

    # loop over frames from the video file stream
    while True:
        start_time = timeit.default_timer()
        frame = np.array(grab_screen(region=origbox))
        
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[: 2]

        # Convert to PyTorch tensor in device with batch dimension
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).to(device)
        # Convert from numpy (H, W, C) to pytorch (C, H, W)
        frame_tensor = frame_tensor.permute(0, 3, 1, 2)

        # Obtain the probability value
        confidence = model(frame_tensor).squeeze().item()

        # Convert frame to cv2 compatible format for visualization        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        frame = cv2.UMat(frame)

        # Label on the frame indicating confidence score
        text = "CONFIDENCE: {}%".format(int(confidence * 100))
        cv2.putText(frame, text, (0, 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2))

        if ENABLE_AIMBOT and confidence > THRESHOLD:
            mouse_lclick()
        
        cv2.imshow("Neural Net Vision (Pine)", frame)
        elapsed = timeit.default_timer() - start_time
        sys.stdout.write(
            "\r{1} FPS with {0} MS interpolation delay \t".format(int(elapsed*1000), int(1/elapsed)))
        sys.stdout.flush()
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    # Clean up on exit
    signal_handler(0, 0)
