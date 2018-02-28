
class Settings(object):
    """
    Project wide management of settings
    """

    def __init__(self, args):
        self.attention_horizontal_splits = int(args.atthorizontal_splits)
        self.overlap_px = int(args.overlap_px)
        self.horizontal_splits = int(args.horizontal_splits)
        #self.anchorfile = args.anchorf
        self.startframe = int(args.startframe)
        self.endframe = int(args.endframe)
        self.attention = (args.attention == 'True')
        self.annotate_frames_with_gt = (args.annotategt == 'True')
        self.extend_mask_by = int(args.extendmask)
        self.att_frame_spread = int(args.attframespread)
        #thickness = str(args.thickness).split(",")
        #self.thickness = [float(thickness[0]), float(thickness[1])]
        self.allowed_number_of_boxes = 500
        self.reuse_last_experiment = (args.reuse_last_experiment == 'True')
        self.postprocess_merge_splitline_bboxes = (args.postprocess_merge_splitline_bboxes == 'True')

        self.debug_save_masks = args.debug_save_masks
        self.debug_save_crops = (args.debug_save_crops == 'True')
        self.debug_color_postprocessed_bboxes = (args.debug_color_postprocessed_bboxes == 'True')
        self.debug_just_count_hist = (args.debug_just_count_hist == 'True')

        self.INPUT_FRAMES = args.input
        self.RUN_NAME = args.name



