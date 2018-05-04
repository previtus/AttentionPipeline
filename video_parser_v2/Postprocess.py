from processing_code.bbox_postprocessing import *
from timeit import default_timer as timer

class Postprocess(object):
    """
    Postprocessing bounding boxes. Merge along the splitting lines.
    """

    def __init__(self, settings, history):
        self.settings = settings
        self.history = history

    def get_iou(self, bboxA, bboxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """

        bb1 = {}
        bb1['x1'] = bboxA["topleft"]["x"]#top left corner
        bb1['y1'] = bboxA["topleft"]["y"]
        bb1['x2'] = bboxA["bottomright"]["x"]
        bb1['y2'] = bboxA["bottomright"]["y"]
        bb2 = {}
        bb2['x1'] = bboxB["topleft"]["x"]#top left corner
        bb2['y1'] = bboxB["topleft"]["y"]
        bb2['x2'] = bboxB["bottomright"]["x"]
        bb2['y2'] = bboxB["bottomright"]["y"]

        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        #bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        perc = intersection_area / float(bb2_area)

        #print("intersection_area",intersection_area, "/ bb2_area",bb2_area, " = ", perc)
        #print("bb1",bb1)
        #print("bb2",bb2)
        return perc

    def postprocess_boxes_inside_each_other(self, bboxes, threshold = 0.95):
        # Reject a bounding box which is inside another bounding box by threshold %
        if len(bboxes)==0:
            return bboxes

        postprocessed_bboxes = []

        for i in range(0,len(bboxes)):
            bbox = bboxes[i]

            reject = False
            #print("len(bboxes)",len(bboxes))
            #print("len(others)",len(others))
            for j in range(0, len(bboxes)):
                if j==i:
                    continue
                other = bboxes[j]

                perc = self.get_iou(other, bbox)
                if self.settings.verbosity >= 3: print("bbox",i," is in ",j," by ",perc)

                if perc > threshold:
                    reject = True
                    if self.settings.verbosity >= 3:
                        print("Rejecting bounding box",i,", because its inside another bbox specifically ",j," by:",perc,"(>",threshold,")")
                        #bbox["label"] = "car"
                    break

            if not reject:
                postprocessed_bboxes.append(bbox)
                if self.settings.verbosity >= 3:
                    print("Keeping bounding box", i, ", because its not inside another bbox till", j, " (<",threshold, ")")

        return postprocessed_bboxes


    def non_max_suppression_tf(self, session, boxes, scores, classes, max_boxes, iou_threshold):
        print("<Tensorflow for NMS>")
        import tensorflow as tf
        from keras import backend as K

        max_boxes_tensor = K.variable(max_boxes, dtype='int32')
        session.run(tf.variables_initializer([max_boxes_tensor]))
        nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)

        boxes = K.gather(boxes, nms_index)
        scores = K.gather(scores, nms_index)
        classes = K.gather(classes, nms_index)

        with session.as_default():
            boxes_out = boxes.eval()
            scores_out = scores.eval()
            classes_out = classes.eval()

        return boxes_out, scores_out, classes_out

    def postprocess_nms(self, bboxes, iou_threshold = 0.5):
        # towards 0.01 its more drastic and deletes more bboxes which are overlapped

        if len(bboxes)==0:
            return bboxes

        import tensorflow as tf
        sess = tf.Session()

        arrays = []
        scores = []
        classes = []
        for j in range(0, len(bboxes)):
            bbox = bboxes[j]
            oldbbox = [bbox["label"], [bbox["topleft"]["y"], bbox["topleft"]["x"], bbox["bottomright"]["y"], bbox["bottomright"]["x"]], bbox["confidence"], 0]

            classes.append(oldbbox[0])
            score = oldbbox[2]
            arrays.append(list(oldbbox[1]))
            scores.append(score)

        arrays = np.array(arrays)

        allowed_number_of_boxes = 500
        nms_arrays, nms_scores, nms_classes = self.non_max_suppression_tf(sess, arrays, scores, classes, allowed_number_of_boxes, iou_threshold)
        reduced_bboxes = []
        for j in range(0, len(nms_arrays)):
            a = [nms_classes[j], nms_arrays[j], nms_scores[j], 0]
            reduced_bboxes.append(a)

        postprocessed_bboxes = []
        for bbox in reduced_bboxes:
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

        return postprocessed_bboxes


    # a bit messy
    def postprocess_bboxes_along_splitlines(self, active_coordinates, bboxes, distance_threshold, threshold_for_ratio, H_multiple,W_multiple, DEBUG_POSTPROCESS_COLOR, DEBUG_SHOW_LINES=False):

        splitlines = get_splitlines(active_coordinates)

        backward_compatible_bbox = []

        for bbox in bboxes:
            #intobbox = [bbox["topleft"]["y"], bbox["topleft"]["x"], bbox["bottomright"]["y"], bbox["bottomright"]["x"]]
            intobbox = [bbox["label"], [bbox["topleft"]["y"], bbox["topleft"]["x"], bbox["bottomright"]["y"], bbox["bottomright"]["x"]], bbox["confidence"], 0]
            backward_compatible_bbox.append(intobbox)

        #print("bboxes", backward_compatible_bbox)

        # structure of bounding box = top, left, bottom, right, (top < bottom), (left < right)
        # structure of crop = left, bottom, right, top, (bottom < top),(left < bottom)

        new_bounding_boxes, keep_bboxes = process_bboxes_near_splitlines(splitlines, backward_compatible_bbox, distance_threshold, threshold_for_ratio,H_multiple,W_multiple, DEBUG_POSTPROCESS_COLOR)

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


        return postprocessed_bboxes

    def postprocess(self, projected_active_coordinates, projected_final_evaluation, DEBUG_POSTPROCESS_COLOR=False, DEBUG_SHOW_LINES=False):
        # careful!
        # both postprocessing methods are not adapted to multiple classes
        # really it should be separated by class, run independently and then merged back
        # for one class (just 'person' for example) it runs nicely

        start = timer()

        processed_evaluations0 = projected_final_evaluation


        # 1] merge along splitlines
        #processed_evaluations = postprocess.postprocess_bboxes_along_splitlines(projected_active_coordinates,projected_final_evaluation, True)
        self.settings.postprocess_merge_splitline_threshold_for_ratio = 2.0
        # Checking similarity with square [] ~ ((h / w) > threshold_for_ratio)
        self.settings.postprocess_merge_splitline_distance_threshold =2 * 20 # was in self.settings.overlap_px
        # Two nearby bboxes will be considered, if their closest distance is less than this

        H_multiple = 2.0
        W_multiple = 4.0
        ### THRESHOLDs
        # h < L*overlap_px_h
        # l + r < K*overlap_px_h
        # H_multiple = 2.0 = L
        # W_multiple = 4.0 = K
        distance_threshold = self.settings.postprocess_merge_splitline_distance_threshold
        threshold_for_ratio = self.settings.postprocess_merge_splitline_threshold_for_ratio

        processed_evaluations1 = self.postprocess_bboxes_along_splitlines(projected_active_coordinates, processed_evaluations0, distance_threshold, threshold_for_ratio,H_multiple,W_multiple, DEBUG_POSTPROCESS_COLOR, DEBUG_SHOW_LINES)

        # 2] NMS
        # NMS setting - iou threshodld
        self.settings.postprocess_iou_threshold = 0.5
        processed_evaluations2 = self.postprocess_nms(processed_evaluations1, self.settings.postprocess_iou_threshold)


        # 3] Delete boxes inside others (by significant %)
        perc_threshold = 0.95
        processed_evaluations3 = self.postprocess_boxes_inside_each_other(processed_evaluations2, perc_threshold)


        time = timer() - start
        self.history.report_postprocessing(time,self.settings.frame_number)

        return processed_evaluations3
        #return processed_evaluations1
