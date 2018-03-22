from processing_code.bbox_postprocessing import *
from timeit import default_timer as timer

class Postprocess(object):
    """
    Postprocessing bounding boxes. Merge along the splitting lines.
    """

    def __init__(self, settings, history):
        self.settings = settings
        self.history = history

    # a bit messy
    def postprocess_bboxes_along_splitlines(self, active_coordinates, bboxes, DEBUG_POSTPROCESS_COLOR, DEBUG_SHOW_LINES=False):
        start = timer()

        splitlines = get_splitlines(active_coordinates)

        backward_compatible_bbox = []

        for bbox in bboxes:
            #intobbox = [bbox["topleft"]["y"], bbox["topleft"]["x"], bbox["bottomright"]["y"], bbox["bottomright"]["x"]]
            intobbox = [bbox["label"], [bbox["topleft"]["y"], bbox["topleft"]["x"], bbox["bottomright"]["y"], bbox["bottomright"]["x"]], bbox["confidence"], 0]
            backward_compatible_bbox.append(intobbox)

        #print("bboxes", backward_compatible_bbox)

        # structure of bounding box = top, left, bottom, right, (top < bottom), (left < right)
        # structure of crop = left, bottom, right, top, (bottom < top),(left < bottom)
        threshold_for_ratio = 2.0
        overlap_px_h = self.settings.overlap_px

        new_bounding_boxes, keep_bboxes = process_bboxes_near_splitlines(splitlines, backward_compatible_bbox, overlap_px_h, threshold_for_ratio, DEBUG_POSTPROCESS_COLOR)

        postprocessed_bboxes = []

        all_boxes = new_bounding_boxes + keep_bboxes
        for bbox in all_boxes:
            # from [bbox["label"], [bbox["topleft"]["y"], bbox["topleft"]["x"], bbox["bottomright"]["y"], bbox["bottomright"]["x"]], bbox["confidence"], 0]

            dictionary = {}
            dictionary["label"] = bbox[0]
            dictionary["confidence"] = bbox[2]
            dictionary["topleft"] = {}
            dictionary["topleft"]["y"] = bbox[1][0]
            dictionary["topleft"]["x"] = bbox[1][1]
            dictionary["bottomright"] = {}
            dictionary["bottomright"]["y"] = bbox[1][2]
            dictionary["bottomright"]["x"] = bbox[1][3]

            postprocessed_bboxes.append(dictionary)

        time = timer() - start
        self.history.report_postprocessing(time,self.settings.frame_number)

        return postprocessed_bboxes
