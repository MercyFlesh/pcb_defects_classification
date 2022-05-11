import os

CLASSES = [
    "open", "short", "mousebit", 
    "spur", "copper", "pin-hole"
]

DATASET_PATH = "PCBData"
OUTPUT_PATH = "output"
DEFECTS_PATH = os.path.sep.join([DATASET_PATH, "defects"])
TEST_PATH = os.path.sep.join([DATASET_PATH, "test_defects"])
