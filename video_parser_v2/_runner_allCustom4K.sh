#!/bin/sh
# This file is called ~/script.sh

splits_settings=[[1,2],[1,3],[2,4],[2,6]]


root_p="/home/vruzicka/storage_pylon5/move_all_from_pylon2/_videos_files/RuzickaDataset/input/"
# 
for base in "S1000010_5fps" "S1000051_5fps" "S1000041_5fps" "S1000021_5fps"
do
   for split_sets in 1 3
   do
      for fin_servers in 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1
      do
         input=$root_p$base"/"
         
         if [ "$split_sets" -eq 1 ]
            then
               att_splits=1
               fin_splits=2
         fi
         if [ "$split_sets" -eq 2 ]
            then
               att_splits=1
               fin_splits=3
         fi
         if [ "$split_sets" -eq 3 ]
            then
               att_splits=2
               fin_splits=4
         fi
         if [ "$split_sets" -eq 4 ]
            then
               att_splits=2
               fin_splits=6
         fi
                  
         att_servers=2

         tmp_name=$base"_"$att_splits"to"$fin_splits
         servers_name=$att_servers"att_"$fin_servers"eval"

         name="all_S3mod_MyCustom4K_"$tmp_name"_"$servers_name

         echo "python run_serverside.py -verbosity 1 -render_history_every 200 -endframe 100 -input "$input" -atthorizontal_splits "$att_splits" -horizontal_splits "$fin_splits" -LimitEvalMach "$fin_servers" -SetAttMach "$att_servers" -name "$name
         python run_serverside.py -verbosity 1 -render_history_every 200 -endframe 100 -input $input -atthorizontal_splits $att_splits -horizontal_splits $fin_splits -LimitEvalMach $fin_servers -SetAttMach $att_servers -name $name

      done
   done
done