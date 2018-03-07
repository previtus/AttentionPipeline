from PIL import Image, ImageDraw
from processing_code.file_handling import make_folder
class Renderer(object):
    """
    Draw final image with bboxes to a screen or file.
    """

    def __init__(self, settings):
        self.settings = settings

        self.render_files_into_folder = self.settings.render_files_into_folder
        self.render_folder_name = self.settings.render_folder_name
        if self.render_files_into_folder:
            make_folder(self.render_folder_name)

    def render(self, final_evaluation, frame):
        if self.render_files_into_folder:
            self.render_into_folder(final_evaluation, frame)



    def render_into_folder(self, final_evaluation, frame, thickness=5):
        """
        final_evaluation: bounding boxes, array of dictionaries for each bbox {} keys label, confidence, topleft, bottomright
        frame: loaded image

        final_evaluation [{'label': 'person', 'confidence': 0.22, 'topleft': {'y': 1417.7257525083612, 'x': 892.1739130434783}, 'bottomright': {'y': 1479.1304347826087, 'x': 924.6822742474917}}, {'label': 'person', 'confidence': 0.34, 'topleft': {'y': 1405.0836120401339, 'x': 944.5484949832776}, 'bottomright': {'y': 1470.1003344481605, 'x': 975.2508361204013}}, {'label': 'person', 'confidence': 0.29, 'topleft': {'y': 1399.6655518394648, 'x': 928.294314381271}, 'bottomright': {'y': 1509.8327759197325, 'x': 980.6688963210703}}, {'label': 'person', 'confidence': 0.32, 'topleft': {'y': 1419.531772575251, 'x': 881.3377926421405}, 'bottomright': {'y': 1520.6688963210702, 'x': 935.5183946488295}}, {'label': 'person', 'confidence': 0.23, 'topleft': {'y': 1405.0836120401339, 'x': 931.9063545150502}, 'bottomright': {'y': 1547.7591973244148, 'x': 973.4448160535118}}, {'label': 'person', 'confidence': 0.44, 'topleft': {'y': 1417.7257525083612, 'x': 2443.545150501672}, 'bottomright': {'y': 1526.0869565217392, 'x': 2474.247491638796}}, {'label': 'person', 'confidence': 0.4, 'topleft': {'y': 1415.9197324414715, 'x': 2541.0702341137126}, 'bottomright': {'y': 1542.3411371237457, 'x': 2584.4147157190637}}, {'label': 'person', 'confidence': 0.43, 'topleft': {'y': 1424.9498327759197, 'x': 2588.026755852843}, 'bottomright': {'y': 1542.3411371237457, 'x': 2631.371237458194}}, {'label': 'person', 'confidence': 0.43, 'topleft': {'y': 1417.7257525083612, 'x': 2719.866220735786}, 'bottomright': {'y': 1547.7591973244148, 'x': 2766.8227424749166}}, {'label': 'person', 'confidence': 0.53, 'topleft': {'y': 1410.5016722408027, 'x': 2902.274247491639}, 'bottomright': {'y': 1596.5217391304348, 'x': 2923.9464882943143}}, {'label': 'person', 'confidence': 0.22, 'topleft': {'y': 1410.5016722408027, 'x': 3218.3277591973247}, 'bottomright': {'y': 1497.190635451505, 'x': 3274.314381270903}}, {'label': 'person', 'confidence': 0.67, 'topleft': {'y': 1417.7257525083612, 'x': 2896.85618729097}, 'bottomright': {'y': 1609.1638795986623, 'x': 2961.872909698997}}, {'label': 'person', 'confidence': 0.42, 'topleft': {'y': 1419.531772575251, 'x': 2969.096989966555}, 'bottomright': {'y': 1583.8795986622074, 'x': 3021.4715719063547}}, {'label': 'person', 'confidence': 0.37, 'topleft': {'y': 1401.4715719063545, 'x': 3223.7458193979933}, 'bottomright': {'y': 1582.0735785953177, 'x': 3281.5384615384614}}, {'label': 'person', 'confidence': 0.5, 'topleft': {'y': 1439.3979933110368, 'x': 3053.9799331103677}, 'bottomright': {'y': 1630.836120401338, 'x': 3135.2508361204013}}]
        frame ('/home/ekmek/intership_project/video_parser_v1/_videos_to_test/DrivingNY/input/frames_0.2fps_236/0001.jpg', <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=3840x2160 at 0x7F6308569390>)

        """

        image = frame[1]
        name = frame[2] # contains just the file name, not to have a messy string splitting here

        draw = ImageDraw.Draw(image)

        for bbox in final_evaluation:
            predicted_class = bbox["label"]
            color = self.label_to_color(predicted_class)

            top = bbox["topleft"]["y"]
            left = bbox["topleft"]["x"]
            bottom = bbox["bottomright"]["y"]
            right = bbox["bottomright"]["x"]
            #top = max(0, np.floor(top + 0.5).astype('int32'))
            #left = max(0, np.floor(left + 0.5).astype('int32'))
            #bottom = min(image_size[1], np.floor(bottom + 0.5).astype('int32'))
            #right = min(image_size[0], np.floor(right + 0.5).astype('int32'))

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=color)
        del draw

        if self.settings.verbosity >= 2:
            print("Saved to", self.render_folder_name + name)
            image.save(self.render_folder_name + name)

    def label_to_color(self,label):
        if label == 'person':
            return "red"
        elif label == 'car':
            return "blue"
        else:
            return "yellow"