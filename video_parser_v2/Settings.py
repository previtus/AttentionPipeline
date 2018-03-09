
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

        self.render_history_every_k_frames = 20

        self.INPUT_FRAMES = args.input
        self.RUN_NAME = args.name

        ### w and h
        self.w = 0
        self.h = 0

        self.set_verbosity(2)

        # Renderer
        self.render_files_into_folder = True
        self.render_folder_name = "__Renders/"+self.RUN_NAME+"/"


        # Connection handling
        self.client_server = False
        self.server_ports_list = ["5000", "9999", "9998", "9997"]

    def set_w_h(self,w,h):
        self.w = w
        self.h = h

    def set_debugger(self,debugger):
        self.debugger = debugger

    def set_verbosity(self, verbosity):
        messages= ["0 = Muted",
        "1 = Minimal verbosity, frame name and speed only",
        "2 = Talkative state, each module will report what it is doing and what are the results",
        "3+ = Printing specially targeted messages, close to debugging"]

        str = "Setting project verbosity to: " + messages[min(verbosity, len(messages)-1)]

        print(str)

        self.verbosity = verbosity