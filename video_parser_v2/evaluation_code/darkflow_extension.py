import math, numpy as np, time
from multiprocessing.pool import ThreadPool
pool = ThreadPool()

# Extension to single image evaluation via:
# tfnet.return_predict(imgcv)
# This method evaluates batch of leaded images.
# predict_extend(tfnet, input_images, batch_size=-1)
# - tfnet is the initialized model
# - input_images is array of loaded images (via cv2.imread(file) )
# - batch_size is optional setting of batch size

def predict_extend(tfnet, input_images, batch_size=-1, verbal=0):
    h, w, _ = input_images[0].shape
    if batch_size == -1:
        batch_size = tfnet.FLAGS.batch
    batch = min(batch_size, len(input_images))
    all_predictions = []

    # predict in batches
    n_batch = int(math.ceil(len(input_images) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(input_images))

        # collect images input in the batch
        this_batch = range(from_idx,to_idx)
        inp_feed = pool.map(lambda i: (
            np.expand_dims(
                tfnet.framework.preprocess(input_images[i]),
                0)
        ), this_batch)

        # Feed to the net
        feed_dict = {tfnet.inp : np.concatenate(inp_feed, 0)}
        if verbal>0:
            tfnet.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = tfnet.sess.run(tfnet.out, feed_dict)
        stop = time.time(); last = stop - start
        if verbal > 0:
            tfnet.say('Eval time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), last / len(inp_feed)))

        # Post processing
        if verbal > 0:
            tfnet.say('Post processing {} inputs ...'.format(len(inp_feed)))

        for i, net_out in enumerate(out):
            #image index [from_idx + i]
            bboxes = postprocess_bboxes(tfnet, net_out, h, w)
            all_predictions.append(bboxes)

    return all_predictions


def postprocess_bboxes(tfnet, net_out, h, w):
    """
    Takes net output, extracts bboxes
    """
    boxes = tfnet.framework.findboxes(net_out)

    # meta
    meta = tfnet.meta
    threshold = meta['thresh']

    resultsForJSON = []
    for b in boxes:
        boxResults = tfnet.framework.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults

        #if tfnet.FLAGS.json:
        if True:
            resultsForJSON.append(
                {"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top},
                 "bottomright": {"x": right, "y": bot}})
            continue
    return resultsForJSON

""" 
Example usage:

# initialize model (...set options)
tfnet = TFNet(options)


from darkflow_extension import predict_extend
for file in in_files:
    input_images.append(cv2.imread(file))

result = predict_extend(tfnet,input_images)

# Results will be list of json values, or empty arrays
#result[0] => [{'label': 'keyboard', 'confidence': 0.44, 'topleft': {'x': 71, 'y': 63}, 'bottomright': {'x': 597, 'y': 598}}]
#result[1] => []
#             ...

"""