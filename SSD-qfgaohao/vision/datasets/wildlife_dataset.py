import numpy as np
import pathlib
import xml.etree.ElementTree as ET
import cv2
import pandas as pd

class WildlifeDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False):
        """Dataset for Wildlife data.
        Args:
            root: The root directory of Wildlife dataset
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "validation_files.txt"         # MODIFY
        else:
            image_sets_file = self.root / "train_files.txt"
        self.ids = WildlifeDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # Load annotations
        self.annotations = pd.read_csv(self.root / "Annotations.csv")
        self.parsed_annotations = {}
        self.annotations_list = self.annotations.values

        label_map = {}

        self.class_names = list(set(self.annotations["Label"]))
        self.class_names.sort()
        self.class_names.insert(0,"BACKGROUND")

        indices = {}
        for i in range(len(self.class_names)):
            indices[self.class_names[i]] = i

        self.class_dict = {}
        self.class_dict["BACKGROUND"] = 0
        for row in self.annotations_list:
            self.class_dict[row[2]] = indices[row[2]]
            if(row[0] not in self.parsed_annotations):
                self.parsed_annotations[row[0]] = [[],[],[]]
            self.parsed_annotations[row[0]][0].append([float(row[3]),float(row[4]),float(row[5]),float(row[6])])
            self.parsed_annotations[row[0]][1].append(indices[row[2]])
            self.parsed_annotations[row[0]][2].append(0)

        print(self.class_dict)

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):

        return (np.array(self.parsed_annotations[image_id][0], dtype=np.float32),
                np.array(self.parsed_annotations[image_id][1], dtype=np.int64),
                np.array(self.parsed_annotations[image_id][2], dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"train/{image_id}"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



