from darkflow.net.build import TFNet
from evaluation_code.darkflow_extension import predict_extend

import numpy

def load_model():
    options = {#"model": "cfg/yolo.cfg",
               #"load": "bin/yolo.weights",
               "pbLoad": "built_graph/yolo.pb",
               "metaLoad": "built_graph/yolo.meta",
               "threshold": 0.3, #yad2k default "score_threshold" 0.3, "iou_threshold" 0.5, ps 0.0 is broken, set 0.01 min
               "gpu": 1.0} # turned on
               #"gpuName": gpuname} #'/gpu:0'
    #self.define('pbLoad', '', 'path to .pb protobuf file (metaLoad must also be specified)')
    #self.define('metaLoad', '', 'path to .meta file generated during --savepb that corresponds to .pb file')

    tfnet = TFNet(options)
    return tfnet

def convert_numpy_floats(result):
    # model result contains list of dictionaries, which have problematic data structure of numpy.float32
    # (in confidence). Lets convert these to me JSON-able
    for item in result:
        for key in item.keys():
            if isinstance(item[key], numpy.float32):
                item[key] = float(item[key])
    return result

def run_on_image(image_object, model):

    result = model.return_predict(image_object)
    result = convert_numpy_floats(result)


    return result

def run_on_images(image_objects, model):

    results = predict_extend(model, image_objects)

    #print("len(results)",len(results))
    for i in range(0,len(results)):
        results[i] = convert_numpy_floats(results[i])

    return results
