import os
import matplotlib.pyplot as plt

from pose_utils import SatellitePoseEstimationDataset

## ~~ Settings ~~ ##
#  Change these to match your setup

dataset_dir = "..\\speed"      # Root directory of dataset (contains /images, LICENSE.MD, *.json)
default_margins = (.2, .2, .1) # (x, y, z) offset between satellite body and rectangular cage used as cropping target



def create_bbox_json(margins=default_margins):
    print ("Generating dataset")
    my_dataset = SatellitePoseEstimationDataset(root_dir=dataset_dir)
    print (" Dataset generation complete")

    my_dataset.generate_bbox_json(margins)

# A json file with bounding box training labels is generated if it does not already exist
bbox_json_filepath_expected = os.path.join(dataset_dir, "train_bbox.json")
if not os.path.exists(bbox_json_filepath_expected):
    create_bbox_json()

