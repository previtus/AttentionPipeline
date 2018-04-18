def get_scope_variable(self, scope_name, name, shape, dtype, initializer):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(name=name, dtype=dtype)
    return v


def non_max_suppression_tf(self, session, boxes, scores, classes, max_boxes, iou_threshold):
    print("<Tensorflow for NMS>")

    if True:
        print("All vars:", [n.name for n in tf.get_default_graph().as_graph_def().node])

        # make empty variables to be reused for:
        # All vars:
        # ['foo/Variable/initial_value', 'foo/Variable', 'foo/Variable/Assign', 'foo/Variable/read',
        # 'foo/init', 'foo/NonMaxSuppression/boxes', 'foo/NonMaxSuppression/scores',
        # 'foo/NonMaxSuppression', 'foo/Gather/params', 'foo/Gather',
        #
        # 'foo/Gather_1/params', 'foo/Gather_1', 'foo/Gather_2/params', 'foo/Gather_2']


        # grows rapidly:
        # All vars: ['foo/Variable/initial_value', 'foo/Variable', 'foo/Variable/Assign', 'foo/Variable/read',
        # 'foo/init', 'foo/NonMaxSuppression/boxes', 'foo/NonMaxSuppression/scores', 'foo/NonMaxSuppression',
        # 'foo/Gather/params', 'foo/Gather',
        # 'foo/Gather_1/params', 'foo/Gather_1',
        # 'foo/Gather_2/params', 'foo/Gather_2', 'foo_1/Variable/initial_value', 'foo_1/Variable',
        # 'foo_1/Variable/Assign', 'foo_1/Variable/read', 'foo_1/init', 'foo_1/NonMaxSuppression/boxes',
        # 'foo_1/NonMaxSuppression/scores', 'foo_1/NonMaxSuppression', 'foo_1/Gather/params', 'foo_1/Gather',
        # 'foo_1/Gather_1/params', 'foo_1/Gather_1',
        #
        # 'foo_1/Gather_2/params', 'foo_1/Gather_2', 'foo_2/Variable/initial_value', 'foo_2/Variable',
        # 'foo_2/Variable/Assign', 'foo_2/Variable/read', 'foo_2/init', 'foo_2/NonMaxSuppression/boxes',
        # 'foo_2/NonMaxSuppression/scores', 'foo_2/NonMaxSuppression', 'foo_2/Gather/params', 'foo_2/Gather',
        # 'foo_2/Gather_1/params', 'foo_2/Gather_1', 'foo_2/Gather_2/params', 'foo_2/Gather_2',
        #
        # 'foo_3/Variable/initial_value', 'foo_3/Variable', 'foo_3/Variable/Assign', 'foo_3/Variable/read', 'foo_3/init', 'foo_3/NonMaxSuppression/boxes', 'foo_3/NonMaxSuppression/scores', 'foo_3/NonMaxSuppression', 'foo_3/Gather/params', 'foo_3/Gather', 'foo_3/Gather_1/params', 'foo_3/Gather_1', 'foo_3/Gather_2/params', 'foo_3/Gather_2', 'foo_4/Variable/initial_value', 'foo_4/Variable', 'foo_4/Variable/Assign', 'foo_4/Variable/read', 'foo_4/init', 'foo_4/NonMaxSuppression/boxes', 'foo_4/NonMaxSuppression/scores', 'foo_4/NonMaxSuppression', 'foo_4/Gather/params', 'foo_4/Gather', 'foo_4/Gather_1/params', 'foo_4/Gather_1', 'foo_4/Gather_2/params', 'foo_4/Gather_2', 'foo_5/Variable/initial_value', 'foo_5/Variable', 'foo_5/Variable/Assign', 'foo_5/Variable/read', 'foo_5/init', 'foo_5/NonMaxSuppression/boxes', 'foo_5/NonMaxSuppression/scores', 'foo_5/NonMaxSuppression', 'foo_5/Gather/params', 'foo_5/Gather', 'foo_5/Gather_1/params', 'foo_5/Gather_1', 'foo_5/Gather_2/params', 'foo_5/Gather_2', 'foo_6/Variable/initial_value', 'foo_6/Variable', 'foo_6/Variable/Assign', 'foo_6/Variable/read', 'foo_6/init', 'foo_6/NonMaxSuppression/boxes', 'foo_6/NonMaxSuppression/scores', 'foo_6/NonMaxSuppression', 'foo_6/Gather/params', 'foo_6/Gather', 'foo_6/Gather_1/params', 'foo_6/Gather_1', 'foo_6/Gather_2/params', 'foo_6/Gather_2', 'foo_7/Variable/initial_value', 'foo_7/Variable', 'foo_7/Variable/Assign', 'foo_7/Variable/read', 'foo_7/init', 'foo_7/NonMaxSuppression/boxes', 'foo_7/NonMaxSuppression/scores', 'foo_7/NonMaxSuppression', 'foo_7/Gather/params', 'foo_7/Gather', 'foo_7/Gather_1/params', 'foo_7/Gather_1', 'foo_7/Gather_2/params', 'foo_7/Gather_2', 'foo_8/Variable/initial_value', 'foo_8/Variable', 'foo_8/Variable/Assign', 'foo_8/Variable/read', 'foo_8/init', 'foo_8/NonMaxSuppression/boxes', 'foo_8/NonMaxSuppression/scores', 'foo_8/NonMaxSuppression', 'foo_8/Gather/params', 'foo_8/Gather', 'foo_8/Gather_1/params', 'foo_8/Gather_1', 'foo_8/Gather_2/params', 'foo_8/Gather_2', 'foo_9/Variable/initial_value', 'foo_9/Variable', 'foo_9/Variable/Assign', 'foo_9/Variable/read', 'foo_9/init', 'foo_9/NonMaxSuppression/boxes', 'foo_9/NonMaxSuppression/scores', 'foo_9/NonMaxSuppression', 'foo_9/Gather/params', 'foo_9/Gather', 'foo_9/Gather_1/params', 'foo_9/Gather_1', 'foo_9/Gather_2/params', 'foo_9/Gather_2']


        # Now:
        # All vars: ['Const', 'foo/max_boxes_tensor/initial_value', 'foo/max_boxes_tensor',
        # 'foo/max_boxes_tensor/Assign', 'foo/max_boxes_tensor/read', 'init', 'NonMaxSuppression/boxes',
        # 'NonMaxSuppression/scores', 'NonMaxSuppression', 'Gather/params', 'Gather',
        # 'Gather_1/params', 'Gather_1', 'Gather_2/params', 'Gather_2', 'Const_1', 'init_1',
        # 'NonMaxSuppression_1/boxes', 'NonMaxSuppression_1/scores', 'NonMaxSuppression_1',
        #
        # 'Gather_3/params', 'Gather_3',
        # 'Gather_4/params', 'Gather_4',
        # 'Gather_5/params', 'Gather_5',
        #
        # 'Const_2', 'init_2', 'NonMaxSuppression_2/boxes', 'NonMaxSuppression_2/scores', 'NonMaxSuppression_2',
        #
        # 'Gather_6/params', 'Gather_6',
        # 'Gather_7/params', 'Gather_7',
        # 'Gather_8/params', 'Gather_8',
        #
        # 'Const_3', 'init_3', 'NonMaxSuppression_3/boxes', 'NonMaxSuppression_3/scores', 'NonMaxSuppression_3',
        #
        # 'Gather_9/params', 'Gather_9',
        # 'Gather_10/params', 'Gather_10', 'Gather_11/params', 'Gather_11', 'Const_4', 'init_4', 'NonMaxSuppression_4/boxes', 'NonMaxSuppression_4/scores', 'NonMaxSuppression_4', 'Gather_12/params', 'Gather_12', 'Gather_13/params', 'Gather_13', 'Gather_14/params', 'Gather_14', 'Const_5', 'init_5', 'NonMaxSuppression_5/boxes', 'NonMaxSuppression_5/scores', 'NonMaxSuppression_5', 'Gather_15/params', 'Gather_15', 'Gather_16/params', 'Gather_16', 'Gather_17/params', 'Gather_17', 'Const_6', 'init_6', 'NonMaxSuppression_6/boxes', 'NonMaxSuppression_6/scores', 'NonMaxSuppression_6', 'Gather_18/params', 'Gather_18', 'Gather_19/params', 'Gather_19', 'Gather_20/params', 'Gather_20', 'Const_7', 'init_7', 'NonMaxSuppression_7/boxes', 'NonMaxSuppression_7/scores', 'NonMaxSuppression_7', 'Gather_21/params', 'Gather_21', 'Gather_22/params', 'Gather_22', 'Gather_23/params', 'Gather_23', 'Const_8', 'init_8', 'NonMaxSuppression_8/boxes', 'NonMaxSuppression_8/scores', 'NonMaxSuppression_8', 'Gather_24/params', 'Gather_24', 'Gather_25/params', 'Gather_25', 'Gather_26/params', 'Gather_26', 'Const_9', 'init_9', 'NonMaxSuppression_9/boxes', 'NonMaxSuppression_9/scores', 'NonMaxSuppression_9', 'Gather_27/params', 'Gather_27', 'Gather_28/params', 'Gather_28', 'Gather_29/params', 'Gather_29', 'Const_10', 'init_10', 'NonMaxSuppression_10/boxes', 'NonMaxSuppression_10/scores', 'NonMaxSuppression_10', 'Gather_30/params', 'Gather_30', 'Gather_31/params', 'Gather_31', 'Gather_32/params', 'Gather_32', 'Const_11', 'init_11', 'NonMaxSuppression_11/boxes', 'NonMaxSuppression_11/scores', 'NonMaxSuppression_11', 'Gather_33/params', 'Gather_33', 'Gather_34/params', 'Gather_34', 'Gather_35/params', 'Gather_35']


        # now2:
        # All vars: ['Const', 'foo/max_boxes_tensor/initial_value', 'foo/max_boxes_tensor', 'foo/max_boxes_tensor/Assign',
        # 'foo/max_boxes_tensor/read', 'foo/init', 'NonMaxSuppression/boxes', 'NonMaxSuppression/scores',
        # 'NonMaxSuppression', 'Gather/params', 'Gather',
        # 'Gather_1/params', 'Gather_1', 'Gather_2/params', 'Gather_2', 'Const_1',
        # 'NonMaxSuppression_1/boxes', 'NonMaxSuppression_1/scores', 'NonMaxSuppression_1',
        #
        # 'Gather_3/params', 'Gather_3',
        # 'Gather_4/params', 'Gather_4',
        # 'Gather_5/params', 'Gather_5',
        #
        # 'Const_2', 'NonMaxSuppression_2/boxes', 'NonMaxSuppression_2/scores', 'NonMaxSuppression_2',
        #
        # 'Gather_6/params', 'Gather_6',
        # 'Gather_7/params', 'Gather_7',
        # 'Gather_8/params', 'Gather_8',
        #
        # 'Const_3', 'NonMaxSuppression_3/boxes', 'NonMaxSuppression_3/scores', 'NonMaxSuppression_3',
        #
        # 'Gather_9/params', 'Gather_9', 'Gather_10/params', 'Gather_10', 'Gather_11/params', 'Gather_11', 'Const_4', 'NonMaxSuppression_4/boxes', 'NonMaxSuppression_4/scores', 'NonMaxSuppression_4', 'Gather_12/params', 'Gather_12', 'Gather_13/params', 'Gather_13', 'Gather_14/params', 'Gather_14', 'Const_5', 'NonMaxSuppression_5/boxes', 'NonMaxSuppression_5/scores', 'NonMaxSuppression_5', 'Gather_15/params', 'Gather_15', 'Gather_16/params', 'Gather_16', 'Gather_17/params', 'Gather_17', 'Const_6', 'NonMaxSuppression_6/boxes', 'NonMaxSuppression_6/scores', 'NonMaxSuppression_6', 'Gather_18/params', 'Gather_18', 'Gather_19/params', 'Gather_19', 'Gather_20/params', 'Gather_20', 'Const_7', 'NonMaxSuppression_7/boxes', 'NonMaxSuppression_7/scores', 'NonMaxSuppression_7', 'Gather_21/params', 'Gather_21', 'Gather_22/params', 'Gather_22', 'Gather_23/params', 'Gather_23', 'Const_8', 'NonMaxSuppression_8/boxes', 'NonMaxSuppression_8/scores', 'NonMaxSuppression_8', 'Gather_24/params', 'Gather_24', 'Gather_25/params', 'Gather_25', 'Gather_26/params', 'Gather_26']

        # All vars: ['Const', 'foo/max_boxes_tensor/initial_value', 'foo/max_boxes_tensor', 'foo/max_boxes_tensor/Assign',
        # 'foo/max_boxes_tensor/read', 'Const_1', 'foo/nms_index/initial_value', 'foo/nms_index', 'foo/nms_index/Assign',
        # 'foo/nms_index/read', 'Const_2', 'foo/boxes/initial_value', 'foo/boxes', 'foo/boxes/Assign', 'foo/boxes/read',
        #
        # 'foo/Cast_1',
        # 'Const_3',
        #
        # 'foo/scores/initial_value', 'foo/scores', 'foo/scores/Assign', 'foo/scores/read',
        #
        # 'Const_4',
        #
        # 'foo/classes/initial_value', 'foo/classes', 'foo/classes/Assign', 'foo/classes/read', 'foo/init',
        #
        # 'foo/init_1', 'NonMaxSuppression', 'boxes_gather', 'scores_gather', 'classes_gather',
        #
        # 'foo_1/Cast', 'NonMaxSuppression_1', 'boxes_gather_1', 'scores_gather_1', 'classes_gather_1',
        #
        # 'foo_2/Cast', 'NonMaxSuppression_2', 'boxes_gather_2', 'scores_gather_2', 'classes_gather_2',
        #
        # 'foo_3/Cast', 'NonMaxSuppression_3', 'boxes_gather_3', 'scores_gather_3', 'classes_gather_3',
        #
        # 'foo_4/Cast', 'NonMaxSuppression_4', 'boxes_gather_4', 'scores_gather_4', 'classes_gather_4',
        #
        # 'foo_5/Cast', 'NonMaxSuppression_5', 'boxes_gather_5', 'scores_gather_5', 'classes_gather_5',
        #
        # 'foo_6/Cast', 'NonMaxSuppression_6', 'boxes_gather_6', 'scores_gather_6', 'classes_gather_6',
        #
        # 'foo_7/Cast', 'NonMaxSuppression_7', 'boxes_gather_7', 'scores_gather_7', 'classes_gather_7',
        #
        # 'foo_8/Cast', 'NonMaxSuppression_8', 'boxes_gather_8', 'scores_gather_8', 'classes_gather_8']


        test_arr_of_int = [0]
        with tf.control_dependencies(None):
            if self.running_for_the_first_time:
                with tf.variable_scope("foo") as self.scope:
                    tf_max_boxes_tensor = tf.get_variable("max_boxes_tensor", shape=None, dtype='int32',
                                                          initializer=max_boxes)
                    tf_nms_index = tf.get_variable("nms_index", shape=None, dtype='int32', initializer=test_arr_of_int)
                    tf_boxes = tf.get_variable("boxes", shape=None, dtype='float32',
                                               initializer=tf.cast(boxes, tf.float32))
                    # boxes = tf.cast(boxes, tf.float32)
                    tf_scores = tf.get_variable("scores", shape=None, dtype='float32', initializer=scores)
                    tf_classes = tf.get_variable("classes", shape=None, dtype='string', initializer=classes)

                    tf_boxes_gather = tf.get_variable("boxes_gather", shape=None, dtype='float32',
                                                      initializer=tf.cast(boxes, tf.float32))
                    tf_scores_gather = tf.get_variable("scores_gather", shape=None, dtype='float32', initializer=scores)
                    tf_classes_gather = tf.get_variable("classes_gather", shape=None, dtype='string',
                                                        initializer=classes)

                    self.sess.run(tf.variables_initializer([tf_nms_index]))
                    init = tf.global_variables_initializer()
                    self.sess.run(init)

                    self.running_for_the_first_time = False

            with tf.variable_scope(self.scope, reuse=True):
                self.scope.reuse_variables()

                tf_max_boxes_tensor = tf.get_variable("max_boxes_tensor", dtype='int32')
                tf_nms_index = tf.get_variable("nms_index", dtype='int32')
                tf_boxes = tf.get_variable("boxes", shape=None, dtype='float32', initializer=tf.cast(boxes, tf.float32))
                # boxes = tf.cast(boxes, tf.float32)
                tf_scores = tf.get_variable("scores", shape=None, dtype='float32', initializer=scores)
                tf_classes = tf.get_variable("classes", shape=None, dtype='string', initializer=classes)

                tf_boxes_gather = tf.get_variable("boxes_gather", shape=None, dtype='float32',
                                                  initializer=tf.cast(boxes, tf.float32))
                tf_scores_gather = tf.get_variable("scores_gather", shape=None, dtype='float32', initializer=scores)
                tf_classes_gather = tf.get_variable("classes_gather", shape=None, dtype='string',
                                                    initializer=classes)

                nms_index = tf.image.non_max_suppression(tf_boxes, tf_scores, tf_max_boxes_tensor,
                                                         iou_threshold=iou_threshold)

                tf_boxes_gather = tf.gather(tf_boxes, nms_index, name="boxes_gather")
                tf_scores_gather = tf.gather(tf_scores, nms_index, name="scores_gather")
                tf_classes_gather = tf.gather(classes, nms_index, name="classes_gather")

                with session.as_default():
                    boxes_out = tf_boxes_gather.eval()
                    scores_out = tf_scores_gather.eval()
                    classes_out = tf_classes_gather.eval()

    return boxes_out, scores_out, classes_out


def postprocess_nms(self, bboxes, iou_threshold=0.5):
    # towards 0.01 its more drastic and deletes more bboxes which are overlapped

    if len(bboxes) == 0:
        return bboxes

    arrays = []
    scores = []
    classes = []
    for j in range(0, len(bboxes)):
        bbox = bboxes[j]
        oldbbox = [bbox["label"],
                   [bbox["topleft"]["y"], bbox["topleft"]["x"], bbox["bottomright"]["y"], bbox["bottomright"]["x"]],
                   bbox["confidence"], 0]

        classes.append(oldbbox[0])
        score = oldbbox[2]
        arrays.append(list(oldbbox[1]))
        scores.append(score)

    arrays = np.array(arrays)

    allowed_number_of_boxes = 500
    nms_arrays, nms_scores, nms_classes = self.non_max_suppression_tf(self.sess, arrays, scores, classes,
                                                                      allowed_number_of_boxes, iou_threshold)

    """
    if self.running_for_the_first_time:
        from tensorflow.python.ops import variable_scope as var_scope

        #with tf.variable_scope('scope') as self.vs:
        with var_scope.variable_scope('scope', reuse=None) as self.vs:

            def non_max_suppression_tf(sess, boxes, scores, classes, max_boxes_tensor, iou_threshold):
                nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)

                boxes = tf.gather(boxes, nms_index, name="boxes_gather")
                scores = tf.gather(scores, nms_index, name="scores_gather")
                classes = tf.gather(classes, nms_index, name="classes_gather")

                with sess.as_default():
                    boxes_out = boxes.eval()
                    scores_out = scores.eval()
                    classes_out = classes.eval()

                return boxes_out, scores_out, classes_out

            self.nsm_template = tf.make_template('nms_temp', non_max_suppression_tf, unique_name_="FOO") #create_scope_now_=True
            self.running_for_the_first_time = False

    print("<Tensorflow for NMS>")
    print("All vars:",[n.name for n in tf.get_default_graph().as_graph_def().node])

    #if True:
    with tf.variable_scope(self.vs, reuse=True):
        nms_arrays, nms_scores, nms_classes = self.nsm_template(self.sess, arrays, scores, classes, allowed_number_of_boxes, iou_threshold)
    """

    reduced_bboxes = []
    for j in range(0, len(nms_arrays)):
        a = [nms_classes[j], nms_arrays[j], nms_scores[j], 0]
        reduced_bboxes.append(a)

    postprocessed_bboxes = []
    for bbox in reduced_bboxes:
        # from [bbox["label"], [bbox["topleft"]["y"], bbox["topleft"]["x"], bbox["bottomright"]["y"], bbox["bottomright"]["x"]], bbox["confidence"], 0]

        dictionary = {}
        dictionary["label"] = bbox[0]
        dictionary["confidence"] = bbox[2]
        dictionary["topleft"] = {}
        dictionary["topleft"]["y"] = bbox[1][0]
        dictionary["topleft"]["x"] = bbox[1][1]
        dictionary["bottomright"] = {}
        dictionary["bottomright"]["y"] = bbox[1][2]
        dictionary["bottomright"]["x"] = bbox[1][3]

        postprocessed_bboxes.append(dictionary)

    return postprocessed_bboxes