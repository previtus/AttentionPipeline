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

    #args.input = "/pylon2/ci4s8dp/vruzicka/_videos_files/liverpool_station_8k/input/frames_4fps/"
    #args.input = "/pylon2/ci4s8dp/vruzicka/_videos_files/RuzickaDataset/input/S1000040_5fps/"
    #args.endframe = '50'
    #args.atthorizontal_splits = 1
    #args.horizontal_splits = 2

    print('dont forget that you can -debug_just_handshake="True"')

    #args.name = "TestOnServer"

    main_loop(args)

    end = timer()
    time = (end - start)
    print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")



