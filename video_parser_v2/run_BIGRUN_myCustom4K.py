# Server tricks with matplotlib plotting
import matplotlib, os, glob, fnmatch
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")
from timeit import default_timer as timer


from datetime import *

months = ["unk","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

import argparse

parser = argparse.ArgumentParser(description='Project: Find BBoxes in video.')
parser.add_argument('-horizontal_splits', help='number or horizontal splits in image', default='2')
parser.add_argument('-overlap_px', help='overlap in pixels', default='20')
parser.add_argument('-atthorizontal_splits', help='number or horizontal splits in image for attention model', default='1')
parser.add_argument('-input', help='path to folder full of frame images',
                    default="/home/ekmek/intership_project/video_parser_v1/_videos_to_test/PL_Pizza sample/input/frames/")
parser.add_argument('-name', help='run name - will output in this dir', default='_Test-'+day+month)
parser.add_argument('-thickness', help='thickness', default='10,2')
parser.add_argument('-extendmask', help='extend mask by', default='300')
parser.add_argument('-startframe', help='start from frame index', default='0')
parser.add_argument('-endframe', help='end with frame index', default='-1')
parser.add_argument('-attframespread', help='look at attention map of this many nearby frames - minus and plus', default='0')
parser.add_argument('-postprocess_merge_splitline_bboxes', help='PostProcessing merging closeby bounding boxes found on the edges of crops.', default='True')

parser.add_argument('-debug_save_masks', help='DEBUG save masks? BW outlines of attention model. accepts "one" or "all"', default='one')
parser.add_argument('-debug_save_crops', help='DEBUG save crops? Attention models crops. accepts "one" or "all"', default='False')
parser.add_argument('-debug_color_postprocessed_bboxes', help='DEBUG color postprocessed bounding boxes?', default='False')
parser.add_argument('-debug_just_count_hist', help='DEBUG just count histograms of numbers of used crops from each video do not evaluate the outside of attention model.', default='False')
parser.add_argument('-debug_just_handshake', help='DEBUG just handshake with servers', default='False')

parser.add_argument('-render_history_every', help='Every k frames we save all the plots.', default='50')
parser.add_argument('-verbosity', help='0 Muted, 1 = Minimal, 2 = Talkative, 3+ = Specific debug', default='1')

parser.add_argument('-precompute_attention', help='Hard switch for killing precomputation (by default on if there are enough servers)', default='True')
parser.add_argument('-precompute_number', help='How many frames we precompute. Seems like 1 is enough.', default='1')

parser.add_argument('-LimitEvalMach', help='# of machines for Final Evaluation, hard limit, otherwise always max', default='0')
parser.add_argument('-SetAttMach', help='# of machines for Attention Evaluation, needs to be specified if we want > 1', default='1')


from main_loop import main_loop

if __name__ == '__main__':
    args = parser.parse_args()

    start = timer()

    args.input = "/home/ekmek/intership_project/video_parser_v1/_videos_to_test/RuzickaDataset/input/S1000044_1fps/"

    """
    Settings:

	videos:
		S1000010_5fps	/home/vruzicka/storage_pylon5/move_all_from_pylon2/_videos_files/RuzickaDataset/input/S1000010_5fps/
		S1000051_5fps	/home/vruzicka/storage_pylon5/move_all_from_pylon2/_videos_files/RuzickaDataset/input/S1000051_5fps/

		S1000041_5fps	/home/vruzicka/storage_pylon5/move_all_from_pylon2/_videos_files/RuzickaDataset/input/S1000041_5fps/
		S1000021_5fps	/home/vruzicka/storage_pylon5/move_all_from_pylon2/_videos_files/RuzickaDataset/input/S1000021_5fps/

	setups:
		MY CUSTOM 4K:

	        args.atthorizontal_splits = 1
	        args.horizontal_splits = 2

	        args.atthorizontal_splits = 2
	        args.horizontal_splits = 4

	        args.atthorizontal_splits = 1
	        args.horizontal_splits = 3

	        args.atthorizontal_splits = 2
	        args.horizontal_splits = 6

	servers:
		
        args.SetAttMach = 2
        args.LimitEvalMach = 1 -> N
        for 2to4 and 2to6

        args.SetAttMach = 1
        args.LimitEvalMach = 1 -> N
        for 1to2
    """

    #base = [
    #    "S1000010_5fps",
    #    "S1000051_5fps",
    #    "S1000041_5fps",
    #    "S1000021_5fps"
    #]
    base = [
        "S1000041_5fps",
        "S1000021_5fps"
    ]

    root = "/home/vruzicka/storage_pylon5/move_all_from_pylon2/_videos_files/RuzickaDataset/input/"

    # TODO
    # 1 attention + 1 eval
    # 0 attention + 1 eval

    #finalEval_server_settings = list(range(18,0,-1))
    #finalEval_server_settings = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    finalEval_server_settings = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    AttEval_server_settings = [
        2
    ] #add 1 maybe
    splits_settings = [
        [1, 3], [2, 6]
    ]

    #duals = ['A', 'B']
    duals = ['A']

    list_links = [root+b+"/" for b in base]
    print(list_links)

    for index,input in enumerate(list_links):

        for splits_setting in splits_settings:
            print("Now we are doing", splits_settings, "splits")

            for AttEval_server_setting in AttEval_server_settings:
                print("Now we are doing", AttEval_server_setting, "servers allowed Attention")

                for finalEval_server_setting in finalEval_server_settings:
                    print("Now we are doing", finalEval_server_settings, "servers allowed Final Evaluation")

                    for dual in duals:
                        input_name = base[index]

                        if input_name == "S1000010_5fps":
                            if splits_setting[0]==1 and splits_setting[1]==2:
                                # skip 13-18 finalEval_server_setting
                                if finalEval_server_setting >= 13:
                                    continue
                            if splits_setting[0]==2 and splits_setting[1]==4:
                                # skip 10-18 finalEval_server_setting
                                if finalEval_server_setting >= 10:
                                    continue

                        print("Now we are doing", dual, "dual")

                        args.input = input

                        args.verbosity = 1
                        args.render_history_every = 200

                        args.endframe = 100

                        args.atthorizontal_splits = splits_setting[0]
                        args.horizontal_splits = splits_setting[1]
                        #args.overlap_px = 50


                        # Final Evaluation Machines
                        args.LimitEvalMach = finalEval_server_setting
                        # Attention Machines
                        args.SetAttMach = AttEval_server_setting

                        tmp_name = input_name+"_"+str(args.atthorizontal_splits)+"to"+str(args.horizontal_splits)
                        servers_name = str(args.SetAttMach) + "att_" + str(args.LimitEvalMach).zfill(2) + "eval"

                        args.name = "MyCustom4K_" + tmp_name + "_" + servers_name + "_" + dual

                        #args.debug_just_handshake = "True"

                        print("RUN", args.name)

                        main_loop(args)

                        end = timer()
                        time = (end - start)
                        print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")

