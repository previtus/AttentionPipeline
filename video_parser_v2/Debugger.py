from PIL import Image, ImageDraw
import numpy as np

class Debugger(object):
    """
    Will have functions useful for debugging.
    """

    def __init__(self, settings, cropscoordinates, evaluation):
        self.settings = settings
        self.cropscoordinates = cropscoordinates
        self.evaluation = evaluation

    def debug_coordinates_in_frame(self, coordinates, frame_image, type, custom_name=''):
        print("Debugging",type+custom_name,"coordinates", len(coordinates), coordinates)
        print("   in image of resolution", frame_image.size)

        scale = 1.0
        if type == 'attention':
            scale = self.cropscoordinates.scale_ratio_of_attention_crop
        else:
            scale = self.cropscoordinates.scale_ratio_of_evaluation_crop

        scaled_image = self.evaluation.imageprocessing.scale_image(frame_image,scale)
        print("   scaled by", scale, " to ", scaled_image.size)

        bboxes = self.coordinates_to_fake_bboxes(coordinates)

        self.debug_draw_bboxes(bboxes, scaled_image, "debug_coordinates_in_frame_"+type+custom_name+".png")

    """ don't call ruins the inner data
    def debug_evaluation_to_bboxes_before_reprojection(self, evaluation, image, type, custom_name=''):
        print("Debugging",type,"evaluation ", len(evaluation), evaluation)

        all_bboxes = []
        for id, bboxes in evaluation:
            bboxes_in_img = self.cropscoordinates.project_back_to_original_image(id, bboxes, type)
            all_bboxes += bboxes_in_img

        print("all boxes BEFORE were ", all_bboxes)
        self.debug_draw_bboxes(all_bboxes, image, "debug_evaluation_to_bboxes_"+type+custom_name+".png")
    """

    def debug_evaluation_to_bboxes_after_reprojection(self, evaluation, image, type, custom_name=''):
        print("Debugging",type,"evaluation ", len(evaluation), evaluation)

        all_bboxes = evaluation
        self.debug_draw_bboxes(all_bboxes, image, "debug_evaluation_to_bboxes_"+type+custom_name+".png")

    def debug_attention_mask(self, mask_image, image=None, optional_bboxes=None, custom_name='', thickness=4):
        print("Debugging mask", mask_image.size)

        if image is None:
            mask_image.save("mask"+custom_name+".png")
        else:
            image = image.convert("RGBA")
            mask_image = mask_image.convert("RGBA")

            if optional_bboxes is not None:
                draw = ImageDraw.Draw(mask_image)
                for bbox in optional_bboxes:
                    predicted_class = bbox["label"]
                    if predicted_class is 'crop':
                        continue

                    top = bbox["topleft"]["y"]
                    left = bbox["topleft"]["x"]
                    bottom = bbox["bottomright"]["y"]
                    right = bbox["bottomright"]["x"]

                    for i in range(thickness):
                        draw.rectangle([left+i, top+i, right-i, bottom-i],outline="orange")
                del draw
            blended_image = Image.blend(image, mask_image, 0.5)
            blended_image.save("mask"+custom_name+".png")


    def coordinates_to_fake_bboxes(self, coordinates):
        fake_bboxes = []
        for crop_coordinate in coordinates:
            id = crop_coordinate[0]
            area = crop_coordinate[1]
            dictionary = {}
            dictionary["label"] = str(id)
            dictionary["confidence"] = 0.42
            dictionary["topleft"] = {}
            dictionary["topleft"]["x"] = area[0]
            dictionary["topleft"]["y"] = area[1]
            dictionary["bottomright"] = {}
            dictionary["bottomright"]["x"] = area[2]
            dictionary["bottomright"]["y"] = area[3]
            fake_bboxes.append(dictionary)
        return fake_bboxes

    def debug_draw_bboxes(self, bboxes, image=None, name=None, thickness=4):
        EXTEND_BY = 0

        if image is None:
            image_size = (self.settings.w, self.settings.h)
            image = Image.new("L", image_size, "black")

        image_size = image.size
        image = image.convert("RGBA")
        tmp = Image.new('RGBA', image_size, (0, 0, 0, 0))

        draw = ImageDraw.Draw(tmp)
        draw_orig = ImageDraw.Draw(image)

        for bbox in bboxes:
            predicted_class = bbox["label"]
            if predicted_class is 'crop':
                continue

            top = bbox["topleft"]["y"]
            left = bbox["topleft"]["x"]
            bottom = bbox["bottomright"]["y"]
            right = bbox["bottomright"]["x"]
            #top = max(0, np.floor(top + 0.5).astype('int32'))
            #left = max(0, np.floor(left + 0.5).astype('int32'))
            #bottom = min(image_size[1], np.floor(bottom + 0.5).astype('int32'))
            #right = min(image_size[0], np.floor(right + 0.5).astype('int32'))

            for i in range(thickness):
                draw.rectangle([left - EXTEND_BY + i, top - EXTEND_BY + i, right + EXTEND_BY - i, bottom + EXTEND_BY - i],
                               outline="red", fill=(255, 255, 255, 30))
                draw_orig.rectangle([left - EXTEND_BY + i, top - EXTEND_BY + i, right + EXTEND_BY - i, bottom + EXTEND_BY - i], outline="red")

            #print("draw rect", [left - EXTEND_BY, top - EXTEND_BY, right + EXTEND_BY, bottom + EXTEND_BY])

        image = Image.alpha_composite(image, tmp)
        del draw
        del draw_orig
        #image.show()
        if name is None:
            image.save("temp.png")
        else:
            image.save(name)

    def label_to_color(self,label):
        if label == 'person':
            return "white"
        elif label == 'car':
            return "blue"
        else:
            return "yellow"