import matplotlib, os, errno
# IF WE ARE ON SERVER WITH NO DISPLAY, then we use Agg:
#print (matplotlib.get_backend())
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

def path_exists(path):
    return os.path.exists(path)

def make_dir_if_doesnt_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def short_summary(model, formating_lens=[10,10,20,20]):
    param_string = ""
    from keras import backend as K
    formating_lens = str(formating_lens)

    for layer in model.layers:
        trainable_count = int( np.sum([K.count_params(p) for p in set(layer.trainable_weights)]))
        non_trainable_count = int( np.sum([K.count_params(p) for p in set(layer.non_trainable_weights)]))

        if 'Dropout' in layer.__class__.__name__:
            param_string += str(layer.rate)+"-"
        if 'Dense' in layer.__class__.__name__:
            param_string += str(layer.units)+"-"

        if trainable_count == 0 and non_trainable_count == 0:
            print ('{:10}[{:10}]: {:20} => {:20}'.format(str(layer.name), str(layer.__class__.__name__), str(layer.input_shape), str(layer.output_shape)))
        else:
            print ('{:10}[{:10}]: {:20} => {:20}, with {} trainable + {} nontrainable'.format(str(layer.name), str(layer.__class__.__name__), str(layer.input_shape), str(layer.output_shape), str(trainable_count), str(non_trainable_count)))

    return param_string[0:-1]

def visualize_histories(histories, names, parameters, parameters_val, show=True, save=False, save_path='', custom_title=None, just_val=False):
    '''
    Visualize multiple histories.

    Example usage:
        h1 = loadHistory('history1.npy')
        h2 = loadHistory('history2.npy')
        visualize_histories([h1, h2], ['history1', 'history2'])
    '''
    import matplotlib.pyplot as plt

    if custom_title is None:
        custom_title = 'model'
    if just_val:
        custom_title = custom_title + ' (just validation results)'

    i = 0
    leg = []
    for hi in histories:
        n = names[i]
        print (parameters[i], parameters_val[i], hi.keys())
        # summarize history for loss
        if not just_val:
            p = plt.plot(hi[parameters[i]], linestyle='dashed')
            color = p[0].get_color()
            plt.plot(hi['val_' + parameters_val[i]], color=color, linewidth=2)
        else:
            plt.plot(hi['val_'+parameters_val[i]], linewidth=2)
        plt.title(custom_title)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        if not just_val:
            leg.append(n + '')
        leg.append(n + '_val')
        i += 1
    #plt.legend(leg, loc='lower left')
    plt.legend(leg, loc='best')
    if save:
        plt.savefig(save_path) #+plotvalues+'.png')

    if show:
        plt.show()

    plt.clf()
    return plt


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

    if show_also is not '':
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

        print ("Saved image to "+filename)
    if show:
        plt.show()

    plt.clf()
    return plt

def save_history(history_dict, filename, added=None):
    # Save history or histories into npy file
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname) and dirname is not '':
        try:
            os.makedirs(dirname)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    if added is None:
        to_be_saved = data = {'S': history_dict}
    else:
        to_be_saved = data = {'S': history_dict, 'A': added}
    np.save(filename, to_be_saved)

def load_history(filename):
    try:
        # Load history object
        loaded = np.load(open(filename))
        added = None
        try:
            added = loaded[()]['A']
        except:
            added = None

        return loaded[()]['S'], added

    except:
        return None