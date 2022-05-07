import cv2
import os
from tools import config

def main():
    os.makedirs(config.DEFECTS_PATH, exist_ok=True)
    for catigorie in config.CLASSES:
        os.makedirs(os.path.sep.join([config.DEFECTS_PATH, catigorie]), exist_ok=True)
    
    for group in os.listdir(config.DATASET_PATH):
        if group.startswith('group'):
            print(f"[INFO] Extracted defects from {group}")
            group_images_path = os.path.sep.join([config.DATASET_PATH, group, group[5:]])
            group_annotations_path = f"{group_images_path}_not"
            for annotation in os.listdir(group_annotations_path):
                image = cv2.imread(os.path.sep.join([group_images_path, f"{annotation[:-4]}_test.jpg"]))
                with open(os.path.sep.join([group_annotations_path, annotation]), 'r') as f:
                    defects_data = f.read()
                    for i, line in enumerate(defects_data.splitlines()):
                        (x1, y1, x2, y2, c) = map(lambda t: int(t), line.split(' '))
                        label = config.CLASSES[c-1]
                        region = image[y1:y2, x1:x2]
                        cv2.imwrite(os.path.sep.join([config.DEFECTS_PATH, label, f"defect_{annotation[:-4]}_{i}.jpg"]), region)

if __name__ == "__main__":
    main()