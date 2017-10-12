import matplotlib, os, errno
# IF WE ARE ON SERVER WITH NO DISPLAY, then we use Agg:
#print matplotlib.get_backend()
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

def path_exists(path):
    return os.path.exists(path)

def make_dir_if_doesnt_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def visualize_history(hi, show=True, save=False, save_path='', show_also='', custom_title=None):
    # Visualize history of Keras model run.
    '''
    Example calls:
    hi = model.fit(...)

    saveHistory(hi.history, 'tmp_saved_history.npy')
    visualize_history(loadHistory('tmp_saved_history.npy'))

    '''

    # list all data in history
    print(hi.keys())
    # summarize history for loss
    plt.plot(hi['loss'])
    plt.plot(hi['val_loss'])

    if show_also <> '':
        plt.plot(hi[show_also], linestyle='dotted')
        plt.plot(hi['val_'+show_also], linestyle='dotted')

    if custom_title is None:
        plt.title('model loss')
    else:
        plt.title(custom_title)

    plt.ylabel('loss')
    plt.xlabel('epoch')

    if show_also == '':
        plt.legend(['train', 'test'], loc='upper left')
    else:
        plt.legend(['train', 'test', 'train-'+show_also, 'test-'+show_also], loc='upper left')


    if save:
        filename = save_path #+'loss.png'
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        plt.savefig(filename)
        #plt.savefig(filename+'.pdf', format='pdf')

        print "Saved image to "+filename
    if show:
        plt.show()

    plt.clf()
    return plt