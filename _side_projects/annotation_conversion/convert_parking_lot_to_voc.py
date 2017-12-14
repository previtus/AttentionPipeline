""" Conversion between annotation formats """

"""
# FROM FORMAT

 The ground truth and detection output is provided in comma separation format (CSV).
frameNumber, personNumber, bodyLeft, bodyTop, bodyRight, bodyBottom
personNumber - A unique identifier for the individual person
frameNumber - The frame number
bodyLeft,bodyTop,bodyRight,bodyBottom - The body bounding box in pixels

"""
input_file = "/home/ekmek/Downloads/datasets/crowds/PNNL ParkingLot/PL_Pizza_GT.txt"

"""
# TO FORMAT
where each image x.jpg is accompanied by annotation in x.xml

<annotation>
	<folder>frames_20171126</folder>
	<filename>0001.jpg</filename>
	<path>/home/ekmek/intership_project/video_parser/_videos_to_test/PittsMine/input/frames_20171126/0001.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>3840</width>
		<height>2160</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	
	<object>
		<name>person</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>52</xmin>
			<ymin>883</ymin>
			<xmax>172</xmax>
			<ymax>1213</ymax>
		</bndbox>
	</object>
	... # multiple objects

</annotation>

"""
output_folder = "/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames/"