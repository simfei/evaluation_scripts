import numpy as np
import collections


def create_panoptic_label(panoptic_seg, segments_info, 
                          thing_list, 
                          label_divisor=256, ignore_label=0, 
                          return_segments_info=False):
    """
    Process the panoptic segmentation output by the DETR model into panoptic prediction in id-format.
    args:
        panoptic_seg: np.array, shape (height, width, 3). The panoptic segmentation output by the model.
        segments_info: list of segmentation info, each elements is a dict, dict keys: 'id', 'isthing', 'category_id', 'area'. 
    returns:
        panoptic_label: panoptic label in id-format (category_id*label_divisor + instance_id).
        segments_info: the same format as the arg 'segments_info', only change 'id' to the new id.
    """
    if panoptic_seg.ndim != 2:
        raise ValueError("input arg 'panoptic_seg' should be in 2D shape!")
    height, width = panoptic_seg.shape
    semantic_label = np.ones_like(panoptic_seg) * ignore_label
    instance_label = np.zeros_like(panoptic_seg)
    instance_count = collections.defaultdict(int)
    for i, segment in enumerate(segments_info):
        idx = segment['id']
        isthing = segment['isthing']
        category_id = segment['category_id']
        selected_pixels = panoptic_seg == idx
        pixel_area = np.sum(selected_pixels)
        # if pixel_area != segment['area']:
        #     raise ValueError('Expect %d pixels for segment %s, gets %d.' %
        #                     (segment['area'], segment, pixel_area))
        semantic_label[selected_pixels] = category_id
        new_id = category_id * label_divisor
        if category_id in thing_list:
            instance_count[category_id] += 1
            if instance_count[category_id] >= label_divisor:
                raise ValueError('Too many instances for category %d in this image.' %
                                category_id)
            instance_label[selected_pixels] = instance_count[category_id]
            new_id = category_id * label_divisor + instance_count[category_id]
        segments_info[i]['id'] = new_id
    panoptic_label = semantic_label * label_divisor + instance_label
    if return_segments_info:
        return panoptic_label.astype(np.int32), segments_info
    else:
        return panoptic_label.astype(np.int32)