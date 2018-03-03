import Settings, Connection, CropsCoordinates, VideoCapture, AttentionModel, History, Renderer, Evaluation, Debugger

def main_loop(args):
    print(args)

    settings = Settings.Settings(args)
    connection = Connection.Connection(settings)
    connection.handshake()

    cropscoordinates = CropsCoordinates.CropsCoordinates(settings)
    videocapture = VideoCapture.VideoCapture(settings)
    evaluation = Evaluation.Evaluation(settings, connection, cropscoordinates)
    attentionmodel = AttentionModel.AttentionModel(settings, cropscoordinates, evaluation)


    history = History.History(settings)
    renderer = Renderer.Renderer(settings)
    debugger = Debugger.Debugger(settings, cropscoordinates, evaluation)
    settings.set_debugger(debugger)

    for frame in videocapture.frame_generator():
        attention_coordinates = cropscoordinates.get_crops_coordinates('attention')
        debugger.debug_coordinates_in_frame(attention_coordinates, frame[1],'attention')

        attention_evaluation = evaluation.evaluate(attention_coordinates, frame, 'attention')
        # attention_evaluation start in attention crops space (size of frame downscaled for attention evaluation
        # so that we can cut crops of 608x608 from it easily)

        projected_evaluation = cropscoordinates.project_evaluation_back(attention_evaluation, 'attention')
        debugger.debug_evaluation_to_bboxes_after_reprojection(projected_evaluation, frame[1], 'attention', 'afterRepro')
        # projected_evaluation are now in original image space

        evaluation_coordinates = cropscoordinates.get_crops_coordinates('evaluation')
        # evaluation_coordinates are in evaluation space. (size of frame downscaled for regular evaluation
        # so that we can cut crops of 608x608 from it easily)
        debugger.debug_coordinates_in_frame(evaluation_coordinates, frame[1], 'evaluation')

        active_coordinates = attentionmodel.get_active_crops(projected_evaluation, evaluation_coordinates, frame)
        debugger.debug_coordinates_in_frame(active_coordinates, frame[1], 'evaluation', 'activeonly')
        # active_coordinates are in evaluation space


        #final_evaluation = evaluation.evaluate(active_coordinates, frame, 'evaluation')
        #debugger.debug_evaluation_to_bboxes(final_evaluation, frame[1], 'evaluation')


        return 0

        final_evaluation = evaluation.evaluate(active_coordinates, frame, 'evaluation')

        #attentionmodel.debug_draw_bboxes_from_evaluation(final_evaluation, frame[1], 'evaluation')

        return 0

        history.add_record(attention_evaluation, final_evaluation, active_crops)
        renderer.render(final_evaluation, frame)