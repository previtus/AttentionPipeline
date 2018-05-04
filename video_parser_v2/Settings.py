

class Settings(object):
    """
    Project wide management of settings
    """

    def __init__(self, args):
        self.attention_horizontal_splits = int(args.atthorizontal_splits)
        self.overlap_px = int(args.overlap_px)
        self.horizontal_splits = int(args.horizontal_splits)
        self.startframe = int(args.startframe)
        self.endframe = int(args.endframe)
        self.extend_mask_by = int(args.extendmask)
        self.att_frame_spread = int(args.attframespread)
        self.postprocess_merge_splitline_bboxes = (args.postprocess_merge_splitline_bboxes == 'True')

        self.debug_save_masks = args.debug_save_masks
        self.debug_save_crops = (args.debug_save_crops == 'True')
        self.debug_color_postprocessed_bboxes = (args.debug_color_postprocessed_bboxes == 'True')
        self.debug_just_count_hist = (args.debug_just_count_hist == 'True')
        self.debug_just_handshake = (args.debug_just_handshake == 'True')

        self.turn_off_attention_baseline = (args.turn_off_attention_baseline == 'True')
        #self.turn_off_attention_baseline = False

        self.render_history_every_k_frames = int(args.render_history_every)

        self.INPUT_FRAMES = args.input
        self.RUN_NAME = args.name

        verbosity = int(args.verbosity)

        self.opencv_or_pil = 'OpenCV' # 'PIL' or 'OpenCV'

        ### w and h
        self.w = 0
        self.h = 0

        self.set_verbosity(verbosity)

        # Renderer
        self.render_files_into_folder = (args.render_files_into_folder == 'True')
        #self.render_folder_name = "__Renders/"+self.RUN_NAME+"/"
        #on_server_path = "/home/vruzicka/storage_pylon5/__BigRun_25Apr/"
        on_server_path = ""
        self.render_folder_name = on_server_path+"__Renders/"+self.RUN_NAME+"/"


        # Connection handling
        self.client_server = True
        self.server_ports_list = []

        # local servers
        for i in range(5000,5040+1): self.server_ports_list.append(str(i))
        # first gpus
        for i in range(9000,9040+1): self.server_ports_list.append(str(i))
        # second gpus
        for i in range(9100,9140+1): self.server_ports_list.append(str(i))

        # Precomputing Attention
        self.precompute_attention_evaluation = (args.precompute_attention == 'True')
        self.precompute_number = int(args.precompute_number) # number of precomputed frames
        self.reserve_machines_for_attention = int(args.SetAttMach) # number of machines for attention (if its turned on)

        # limit number of servers (individual connections) available, if set to >0
        # remember +reserve_machines_for_attention is used for attention precomp.
        self.final_evaluation_limit_servers = int(args.LimitEvalMach)

        # is set during the run
        self.frame_number = -1

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

    def save_settings(self):
        string = "SETTINGS: \n"

        import copy,json

        save_dict = copy.deepcopy(vars(self))
        del save_dict["server_ports_list"]
        string+=(json.dumps(save_dict, indent=2))

        path = self.render_folder_name+"settings.txt"
        with open(path, "w") as text_file:
            text_file.write(string)