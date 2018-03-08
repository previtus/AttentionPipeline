import Settings, Connection, CropsCoordinates, VideoCapture, AttentionModel, History, Renderer, Evaluation, Debugger, Postprocess

def main_loop(args):
    print(args)

    settings = Settings.Settings(args)
    history = History.History(settings)
    connection = Connection.Connection(settings)
    #if connection.failed: return -1

    cropscoordinates = CropsCoordinates.CropsCoordinates(settings)
    videocapture = VideoCapture.VideoCapture(settings, history)
    evaluation = Evaluation.Evaluation(settings, connection, cropscoordinates, history)
    attentionmodel = AttentionModel.AttentionModel(settings, cropscoordinates, evaluation, history)
    postprocess = Postprocess.Postprocess(settings)

    renderer = Renderer.Renderer(settings)
    debugger = Debugger.Debugger(settings, cropscoordinates, evaluation)
    settings.set_debugger(debugger)

    for frame in videocapture.frame_generator():

        attention_coordinates = cropscoordinates.get_crops_coordinates('attention')
        #debugger.debug_coordinates_in_frame(attention_coordinates, frame[1],'attention')

        attention_evaluation = evaluation.evaluate(attention_coordinates, frame, 'attention')
        # attention_evaluation start in attention crops space (size of frame downscaled for attention evaluation
        # so that we can cut crops of 608x608 from it easily)

        projected_evaluation = cropscoordinates.project_evaluation_back(attention_evaluation, 'attention')
        #debugger.debug_evaluation_to_bboxes_after_reprojection(projected_evaluation, frame[1], 'attention', 'afterRepro')
        # projected_evaluation are now in original image space

        evaluation_coordinates = cropscoordinates.get_crops_coordinates('evaluation')
        # evaluation_coordinates are in evaluation space. (size of frame downscaled for regular evaluation
        # so that we can cut crops of 608x608 from it easily)
        #debugger.debug_coordinates_in_frame(evaluation_coordinates, frame[1], 'evaluation')

        #active_coordinates = attentionmodel.get_active_crops(projected_evaluation, evaluation_coordinates, frame)
        active_coordinates = attentionmodel.get_active_crops_faster(projected_evaluation, evaluation_coordinates, frame)

        #debugger.debug_coordinates_in_frame(active_coordinates, frame[1], 'evaluation', 'activeonly')
        # active_coordinates are in evaluation space

        if len(active_coordinates) == 0:
            print("Nothing left active - that's possibly ok, skip")
            continue

        final_evaluation = evaluation.evaluate(active_coordinates, frame, 'evaluation')
        # evaluation are in evaluation space
        projected_final_evaluation = cropscoordinates.project_evaluation_back(final_evaluation, 'evaluation')
        # projected back to original space

        projected_active_coordinates = cropscoordinates.project_coordinates_back(active_coordinates, 'evaluation')

        processed_evaluations = postprocess.postprocess_bboxes_along_splitlines(projected_active_coordinates,projected_final_evaluation, True)
        #debugger.debug_evaluation_to_bboxes_after_reprojection(processed_evaluations, frame[1], 'finalpostprocessed'+frame[0][-8:-4])

        #history.add_record(attention_evaluation, final_evaluation, active_coordinates)
        renderer.render(processed_evaluations, frame)