import os
import io
import json
from PIL import Image
import cv2
import numpy as np
from utils import create_panoptic_label
from panoptic_quality import pq_compute
from panopticapi.utils import rgb2id, id2rgb


class PQEvaluator(object):
    def __init__(self, ann_file, ann_folder, output_dir='panoptic_eval', foreground=False):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self.foreground = foreground
        self.predictions = []

    def update(self, result, file_name, image_id):
        """
        Get panoptic prediction (RGB) and save it.
        Get panoptic result dict and update self.predictions.
        args:
            result: panoptic postprocessor result (from DETR).
            file_name: image file name in annotation file.
            image_id: image id in annotation file.
        """
        segments_info = result['segments_info']
        panoptic_seg = np.array(Image.open(io.BytesIO(result['png_string'])))
        # create panoptic label (id)
        panoptic_pred, segments_info = create_panoptic_label(rgb2id(panoptic_seg), segments_info, return_segments_info=True)
        # get panoptic pred RGB
        panoptic_pred_color = id2rgb(panoptic_pred)
        # save the panoptic pred RGB image in '.png' format to preserve information
        cv2.imwrite(os.path.join(self.output_dir, file_name.replace(".jpg", ".png")), panoptic_pred_color[...,::-1])

        res_pan = {}
        res_pan['image_id'] = image_id
        res_pan['file_name'] = file_name.replace(".jpg", ".png")
        res_pan['segments_info'] = segments_info
        self.predictions.append(res_pan)
        

    def summarize(self):
        json_data = {'annotations': self.predictions}
        predictions_json = os.path.join(self.output_dir, "predictions.json")
        with open(predictions_json, 'w') as f:
            json.dump(json_data, f)
        return pq_compute(self.gt_json, predictions_json, gt_folder=self.gt_folder, pred_folder=self.output_dir, foreground_eval=self.foreground)

