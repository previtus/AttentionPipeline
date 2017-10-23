#Output one image every second, named out1.png, out2.png, out3.png, etc.

ffmpeg -i Exchanging_bags_day_indoor_1_original.mp4 -vf fps=1 test_1fps/frame_%d.png

