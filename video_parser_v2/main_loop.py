import Settings, Connection, CropsCoordinates, VideoCapture, AttentionModel, History, Renderer, Evaluation

def main_loop(args):
    print(args)

    settings = Settings.Settings(args)
    connection = Connection.Connection(settings)
    connection.handshake()
    evaluation = Evaluation.Evaluation(settings, connection)

    cropscoordinates = CropsCoordinates.CropsCoordinates(settings)
    videocapture = VideoCapture.VideoCapture(settings)
    attentionmodel = AttentionModel.AttentionModel(settings)

    history = History.History(settings)
    renderer = Renderer.Renderer(settings)

    for frame in videocapture.frame_generator():
        attention_coordinates = cropscoordinates.get_crops_coordinates('attention')
        attention_evaluation = evaluation.evaluate(attention_coordinates, frame)
        active_crops = attentionmodel.get_active_crops(attention_evaluation)
        evaluation_coordinates = cropscoordinates.get_crops_coordinates('evaluation', active_crops)
        final_evaluation = evaluation.evaluate(evaluation_coordinates, frame)

        history.add_record(attention_evaluation, final_evaluation, active_crops)
        renderer.render(final_evaluation, frame)