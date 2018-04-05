from PIL import Image, ImageDraw
import cv2
from operator import add

import numpy as np
from timeit import default_timer as timer
class AttentionModel(object):
    """
    Calculation of which crop should be active. Stands in the middle of two evaluations - attention evaluation
    should produce a rough estimation where we should look, this class determines which crops are active from it.
    """

    def __init__(self, settings, cropscoordinates, evaluation, history):
        self.settings = settings
        self.cropscoordinates = cropscoordinates
        self.evaluation = evaluation
        self.history = history

    def get_active_crops_intersections(self, projected_evaluation, evaluation_coordinates, frame):
        """
            Even faster, just looking at intersections
        """
        start = timer()
        #projected_evaluation = self.cropscoordinates.project_evaluation_to(projected_evaluation, 'original_image_to_evaluation_space')
        scaled_extend = self.settings.extend_mask_by * self.cropscoordinates.scale_ratio_of_evaluation_crop

        evaluations_bboxes = self.cropscoordinates.evaluations_to_bboxes(projected_evaluation)

        active_coordinates = []
        already_loaded_ids = []

        scale = self.cropscoordinates.scale_ratio_of_evaluation_crop
        nw = int(self.settings.w * scale)
        nh = int(self.settings.h * scale)
        image_size = (nw, nh)

        # look at % sized corner
        crop_size = 608 # self.crop_size_in_evaluation
        mask_over = 0.1

        for a in evaluations_bboxes:

            ## edit the a into a_extended
            top = a[1]#bbox["topleft"]["y"]
            left = a[0]#bbox["topleft"]["x"]
            bottom = a[3]#bbox["bottomright"]["y"]
            right = a[2]#bbox["bottomright"]["x"]
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image_size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image_size[0], np.floor(right + 0.5).astype('int32'))

            a_extended = [left - scaled_extend, top - scaled_extend, right + scaled_extend, bottom + scaled_extend]

            for b_coordinate in evaluation_coordinates:
                box = list(b_coordinate[1])
                id = b_coordinate[0]


                # intersections in corners

                a_l = [box[0]+0, box[1]+0, box[0]+crop_size * (1 - mask_over), box[1]+crop_size * (1 - mask_over)]
                b_l = [box[0]+0, box[1]+crop_size * (mask_over), box[0]+crop_size * (1 - mask_over), box[1]+crop_size * (1 - mask_over) + crop_size * (mask_over)]
                c_l = [box[0]+crop_size * mask_over, box[1]+0, box[0]+crop_size * (1 - mask_over) + crop_size * mask_over, box[1]+crop_size * (1 - mask_over)]
                d_l = [box[0]+crop_size * mask_over, box[1]+crop_size * mask_over, box[0]+crop_size * (1 - mask_over) + crop_size * mask_over, box[1]+crop_size * (1 - mask_over) + crop_size * mask_over]

                corner_empty = False
                for p in [a_l, b_l, c_l, d_l]:
                    intersects = self.bounding_boxes_intersect(a_extended, p)

                    if not intersects:
                        corner_empty = True
                        break
                if corner_empty:
                    continue

                # intersections in the main square

                intersects = self.bounding_boxes_intersect(a_extended, box)

                # print(intersects)
                if intersects:
                    if id not in already_loaded_ids:
                        #print(id, "<=>", a_extended, box)

                        active_coordinates.append([id, box])
                        already_loaded_ids.append(id)

        end = timer()
        time_active_coords = end - start
        if self.settings.verbosity > 2:
            print("Attention model found",len(active_coordinates),"active crops in the image (out of",len(evaluation_coordinates),") in ",time_active_coords)

        self.history.report_time_getting_active_crops(time_active_coords, self.settings.frame_number)
        self.history.report_attention(len(active_coordinates), len(evaluation_coordinates), self.settings.frame_number)

        return active_coordinates

    def get_active_crops_older(self, projected_evaluation, evaluation_coordinates, frame):
        """
            Faster, because we don't scale the temp mask image
            possibly can be further rewritten...
        """
        start = timer()
        projected_evaluation = self.cropscoordinates.project_evaluation_to(projected_evaluation, 'original_image_to_evaluation_space')
        scaled_extend = self.settings.extend_mask_by * self.cropscoordinates.scale_ratio_of_evaluation_crop

        mask_image = self.mask_image_eval_space_size(projected_evaluation, scaled_extend)
        # building both the mask and in the evaluation space

        #self.settings.debugger.debug_attention_mask(mask_image, custom_name="__"+str(self.settings.frame_number)+"checkIfItsTheSame")

        # evaluation_coordinates are also in the evaluation space
        active_coordinates = self.active_coordinates_in_mask(mask_image, evaluation_coordinates)

        end = timer()
        time_active_coords = end - start
        if self.settings.verbosity > 2:
            print("Attention model found",len(active_coordinates),"active crops in the image (out of",len(evaluation_coordinates),") in ",time_active_coords)

        self.history.report_time_getting_active_crops(time_active_coords, self.settings.frame_number)
        self.history.report_attention(len(active_coordinates), len(evaluation_coordinates), self.settings.frame_number)

        return active_coordinates

    def in_between(self,p,a,b):
        # p in between of a and b
        if p >= a and p <= b:
            return True
        if p >= b and p <= a:
            return True
        return False

    def intervals_intersect(self, start1, end1, start2, end2):
        if start1 > end1:
            t = end1
            end1 = start1
            start1 = t
        if start2 > end2:
            t = end2
            end2 = start2
            start2 = t
        """Does the range (start1, end1) overlap with (start2, end2)?"""
        return end1 >= start2 and end2 >= start1

    def bounding_boxes_intersect(self, bbox1, bbox2):
        left1, top1, right1, bottom1 = bbox1
        left2, top2, right2, bottom2 = bbox2

        if (self.intervals_intersect(left1, right1, left2, right2) and
            self.intervals_intersect(bottom1, top1, bottom2, top2)):
            return True

        return False

    def active_coordinates_in_mask(self, mask_image, coordinates):
        #print("coordinates", coordinates)

        crop_size = self.cropscoordinates.crop_size_in_evaluation
        mask_over = 0.1
        #print("crop_size", crop_size)

        active_coordinates = []
        for crop_coordinates in coordinates:
                id = crop_coordinates[0]
                area = crop_coordinates[1]

                cropped_mask = mask_image.crop(box=area)
                cropped_mask = cropped_mask.resize((crop_size, crop_size), resample=Image.BILINEAR)
                cropped_mask.load()


                # four corners
                a = cropped_mask.crop(box=(0, 0, crop_size * (1 - mask_over), crop_size * (1 - mask_over)))
                b = cropped_mask.crop(
                    box=(0, crop_size * (mask_over), crop_size * (1 - mask_over), crop_size * (1 - mask_over) + crop_size * (mask_over)))
                c = cropped_mask.crop(
                    box=(crop_size * mask_over, 0, crop_size * (1 - mask_over) + crop_size * mask_over, crop_size * (1 - mask_over)))
                d = cropped_mask.crop(box=(
                    crop_size * mask_over, crop_size * mask_over, crop_size * (1 - mask_over) + crop_size * mask_over,
                    crop_size * (1 - mask_over) + crop_size * mask_over))

                a_l = [0, 0, crop_size * (1 - mask_over), crop_size * (1 - mask_over)]
                b_l = [0, crop_size * (mask_over), crop_size * (1 - mask_over), crop_size * (1 - mask_over) + crop_size * (mask_over)]
                c_l = [crop_size * mask_over, 0, crop_size * (1 - mask_over) + crop_size * mask_over, crop_size * (1 - mask_over)]
                d_l = [crop_size * mask_over, crop_size * mask_over, crop_size * (1 - mask_over) + crop_size * mask_over,crop_size * (1 - mask_over) + crop_size * mask_over]

                corner_empty = False
                for p in [a, b, c, d]:
                    p.load()
                    lum = np.sum(np.sum(p.getextrema(), 0))
                    # print(p.size, lum)
                    if lum == 0:
                        corner_empty = True
                        break

                if corner_empty:
                    continue

                extrema = cropped_mask.getextrema()
                extrema_sum = np.sum(extrema, 0)
                #print("summed extrema", extrema_sum)

                if extrema_sum == 0:  # and extrema_sum[1] == 0:
                    continue

                active_coordinates.append([id, area])

        return active_coordinates

    def mask_image(self, bboxes, EXTEND_BY):

        image_size = (self.settings.w, self.settings.h)
        image = Image.new("L", image_size, "black")

        draw = ImageDraw.Draw(image)

        for bbox in bboxes:
            predicted_class = bbox["label"]
            if predicted_class is 'crop':
                continue

            top = bbox["topleft"]["y"]
            left = bbox["topleft"]["x"]
            bottom = bbox["bottomright"]["y"]
            right = bbox["bottomright"]["x"]
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image_size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image_size[0], np.floor(right + 0.5).astype('int32'))

            draw.rectangle([left - EXTEND_BY, top - EXTEND_BY, right + EXTEND_BY, bottom + EXTEND_BY], outline="white", fill="white")

        del draw
        return image

    def mask_image_eval_space_size(self, bboxes_in_eval_space, EXTEND_BY):

        scale = self.cropscoordinates.scale_ratio_of_evaluation_crop
        nw = int(self.settings.w * scale)
        nh = int(self.settings.h * scale)

        image_size = (nw, nh)
        image = Image.new("L", image_size, "black")

        draw = ImageDraw.Draw(image)

        for bbox in bboxes_in_eval_space:
            predicted_class = bbox["label"]
            if predicted_class is 'crop':
                continue

            top = bbox["topleft"]["y"]
            left = bbox["topleft"]["x"]
            bottom = bbox["bottomright"]["y"]
            right = bbox["bottomright"]["x"]
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image_size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image_size[0], np.floor(right + 0.5).astype('int32'))

            draw.rectangle([left - EXTEND_BY, top - EXTEND_BY, right + EXTEND_BY, bottom + EXTEND_BY], outline="white", fill="white")

        del draw
        return image