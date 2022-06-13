import os
IMG_SHAPE = 32
Band = 11
BATCH_SIZE = 10
EPOCHS = 150
BASE_OUTPUT = "D:/py_codes/Unet_tf/result"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
test_ratio = 0.3