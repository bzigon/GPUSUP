
from keras.utils import np_utils
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from timeit import default_timer as timer
import os, sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
import pickle

def create_base_network(NumberOfFeatures, NumberOfClasses,init_mode='glorot_normal'):
    """
    This function will create the base network and return the model.

    This is a helper function that the other network creation functions will call.
    """
    network = Sequential()
    network.add(Dense(44, activation='sigmoid', kernel_initializer=init_mode,input_dim=NumberOfFeatures))
#    network.add(Dense(22, activation='sigmoid',kernel_initializer=init_mode))
    network.add(Dense(NumberOfClasses, activation='softmax',kernel_initializer=init_mode))
    return network


def create_neural_network(NumberOfFeatures, NumberOfClasses, optimizer_type, lr, moment, lr_decay):
    """
    This function will create the network and return the model.

    """
    model = create_base_network(NumberOfFeatures, NumberOfClasses)
    if optimizer_type == 'sgd':
        opt = optimizers.SGD(lr=lr, momentum=moment, decay=lr_decay)
    else:
        opt = optimizer_type

    model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    print(model.summary())
    return model


def graph_model(history, model, file_name_tv, file_name_lv, title):
    training_accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']

    training_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_accuracy) + 1)

    # Visualize training and validation accuracy
    fig = plt.figure()
    plt.plot(epoch_count, training_accuracy, 'r--')
    plt.plot(epoch_count, val_accuracy, 'b-')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.ylim([0.4, 1.0])
    plt.show()
    fig.savefig(file_name_tv)
    
    # Visualize training and validation loss
    fig = plt.figure()
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, val_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss Score')
    plt.title(title)
    plt.show()
    fig.savefig(file_name_lv)
    
    
#    print('Data to Graph')
#    for i in range(len(training_accuracy)):
#        print('%3d train %8.6f  test %8.6f' % (i,training_accuracy[i],val_accuracy[i]))


def read_samples(file_name, pflag=True):
    print('\nReading samples from ', file_name)
    df = pd.read_csv(file_name)
    X = df.iloc[:, 1:].values
    X = X.astype(np.float32)

    kinx = df.columns.get_loc('kernel')
    if kinx != 0:
        print('Warning: The logic assumes the kernel attribute is index 0')
    dfy = df.loc[:,'kernel'].values

    encoder = preprocessing.LabelEncoder()
    encoder.fit(dfy)
    
    if pflag:
      print('Encoder classes = \n')
      for i, c in enumerate(encoder.classes_):
         print('\tValue = {:2d}  Class = {}'.format(i, c))
    
    encoded_Y = encoder.transform(dfy)
    encoded_Y = encoded_Y.astype(np.float32)    # the labels are now an integer
    onehot_Y = np_utils.to_categorical(encoded_Y)
    onehot_Y = onehot_Y.astype(np.float32)      # the integer is now in binary format
    
    return X,onehot_Y,encoded_Y, dfy, df.columns

def do_trainXX(nFeatures,
             nClasses,
             model,
             cb_list,
             input_train_file,
             validation_file,
             batch_size,
             max_epochs):

    train_x, train_y, none, none, none  = read_samples(input_train_file)
    val_x, val_y, none, none, none = read_samples(validation_file)

    history = model.fit(train_x, train_y,
                        epochs=max_epochs,
                        verbose=2,
                        batch_size=batch_size,
                        callbacks=cb_list,
                        validation_data=(val_x, val_y))  # Data to use for evaluation
    return history, model, len(train_y)


# *********************************************************************************************
# *
# *********************************************************************************************
def perform_training(nFeatures,
                     nClasses,
                     mdlname,
                     input_train_file,
                     validation_file,
                     batch_size,
                     nepochs,
                     optimizer_type,
                     data_dir,
                     lr,
                     momentum,
                     lr_decay):

   train_x, train_y, encoded_y, none, none  = read_samples(input_train_file)
   val_x, val_y, none, none, none = read_samples(validation_file)
       
   if False:
       model = create_neural_network(nFeatures, nClasses, optimizer_type, lr, momentum, lr_decay)
       cb_list = []
       history = model.fit(train_x, train_y,
                        epochs=nepochs,
                        verbose=2,
                        batch_size=batch_size,
                        callbacks=cb_list,
                        validation_data=(val_x, val_y))  # Data to use for evaluation
                        
       #callback_list = []
       #history, predictive_model, num_samples = do_train(nFeatures,
       #                                             nClasses,
       #                                             model,
       #                                             callback_list,
       #                                             input_train_file,
       #                                             validation_file,
       #                                             batch_size,
       #                                             nepochs)
       num_samples = len(train_y)
       graph_model(history, model,
                   '{0}\\TrainValidateAcc-{1}-{2}.pdf'.format(data_dir, num_samples, optimizer_type),
                   '{0}\\TrainValidateLoss-{1}-{2}.pdf'.format(data_dir, num_samples, optimizer_type),
                   'Optimizer: {0} (ss={1})'.format(optimizer_type, num_samples))
       print("Writing model %s" % mdlname)
       model.save(mdlname)
    
   if True:
      model = RandomForestClassifier(n_estimators=200).fit(train_x, train_y)
      pickle.dump(model, open(mdlname, 'wb'))
      
   if False:
      model = GaussianNB().fit(train_x, encoded_y)
      pickle.dump(model, open(mdlname, 'wb'))
      
   return

# *********************************************************************************************
# *
# *********************************************************************************************
def perform_testing(nClasses, mdlname, validation_file):
    print('\n\nLoading model {} for testing '.format(mdlname))
#    perform_NNtesting(nClasses, mdlname, validation_file)
    perform_DTtesting(nClasses, mdlname, validation_file)
#    perform_NBtesting(nClasses, mdlname, validation_file)
   
# https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
# https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough

def perform_DTtesting(nClasses, mdlname, validation_file):
    model = pickle.load(open(mdlname, 'rb'))
    test_x, _, _, kname, cnames = read_samples(validation_file, False)
    classes = model.predict(test_x)
    
    estimator = model.estimators_[5]
   # Export as dot file
    export_graphviz(estimator, 
                out_file='tree.dot', 
                feature_names = cnames,
                class_names = kname,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    print("\nRandom Forest")
    show_results(nClasses, classes, kname)
    return
 
def perform_NNtesting(nClasses, mdlname, validation_file):
    predictive_model = load_model(mdlname)
    test_x, _, _, kname, none = read_samples(validation_file, False)
    classes = predictive_model.predict(test_x)

    print("\nNeural Network")
    show_results(nClasses, classes, kname)
    return
    
def perform_NBtesting(nClasses, mdlname, validation_file):
    model = pickle.load(open(mdlname, 'rb'))
    test_x, _, _, kname, none = read_samples(validation_file, False)
    classes = model.predict(test_x)
    classes_proba = model.predict_proba(test_x)

    print("\nNaive Bayes")
    show_results(nClasses, classes_proba, kname)
    return
    
def show_results(nClasses, classes, kname):
    print("     ", end='')
    for i in range(nClasses):
       print(' {:2d}'.format(i), end='')
    print()
    print("     ", end='')
    for i in range(nClasses):
       print('==='.format(i), end='')
    print()
    
    for i in range(len(classes)):
        v = list('0' * nClasses)
        c = np.argmax(classes[i])
        v[c] = '1'
        #print('%4d: ' % (i), '  '.join(v), end="\n")
        print('%4d: ' % (i), '  '.join(v), ' pred cls=%d' % (c), ' inference kname=%s' % (kname[i]), end="\n")
        #print('%4d: ' % (i), '  '.join(v), ' pred cls=%d' % (c), ' actual class=%d' % (enc_y[i]), end="\n")
        #
        # The next print statement only makes sense if the validation file is the test file.
        # Otherwise, you dont know the true class of the rows in the data file.
        #
        #            print('  act class=%d' % (int(enc_y[i])))
        
# *********************************************************************************************
# *
# *********************************************************************************************
class NetworkClass:
    def __init__(self, nw_func, opt_type, lr, moment, lr_decay, nfeatures, nclasses):
        self.nw_func    = nw_func
        self.opt_type   = opt_type
        self.lr         = lr
        self.moment     = moment
        self.lr_decay   = lr_decay
        self.nfeatures  = nfeatures
        self.nclasses   = nclasses

    def __call__(self):
        return self.nw_func(self.nfeatures, self.nclasses, self.opt_type, self.lr, self.moment, self.lr_decay)

def perform_crossvalidation(NumberOfFeatures,
                                   NumberOfClasses,
                                   NumberOfEpochs,
                                   NumberOfCrossValidations,
                                   input_train_file,
                                   data_dir,
                                   batch_size,
                                   lr_decay):

    train_x, train_y, none  = read_samples(input_train_file)

    network_parameters = [
    [create_network, 'rmsprop',  NumberOfEpochs,  0.001, 0.0, 'r-,'],
    [create_network, 'sgd',      NumberOfEpochs,  0.002, 0.9, 'b--o'],
    [create_network, 'adagrad',  NumberOfEpochs,  0.010, 0.0,'g-.v'],
    [create_network, 'adadelta', NumberOfEpochs,  1.000, 0.0,'c:^'],
    [create_network, 'adam',     NumberOfEpochs,  0.001, 0.0,'m-s'],
    [create_network, 'adamax',   NumberOfEpochs,  0.002, 0.0,'k--*'],
    [create_network, 'nadam',    NumberOfEpochs,  0.002, 0.0,'y-.x']
    ]
    
    start = timer()
    print("\nBegin %d-Fold Cross Validation of %d parameter sets" % (NumberOfCrossValidations,len(network_parameters)))
    legend = []
    ymax = 0.0
    ymin = 1.0
    with open("cross_validation.txt", "w") as fh:
        for (nw, opt_type, nepoch, lr, moment, linestyle) in network_parameters:
            nw_class = NetworkClass(nw, opt_type, lr, moment, lr_decay, NumberOfFeatures, NumberOfClasses)
            neural_network = KerasClassifier(build_fn=nw_class,
                                              epochs=nepoch,
                                              batch_size=batch_size,
                                              verbose=0)
            start = timer()
            scores = cross_val_score(neural_network, train_x, train_y, verbose=0, cv=NumberOfCrossValidations)
            stop = timer()
            for s in scores:
                print("%8.3f " % (s), end="")
                fh.write("%8.3f & " % (s))

            print(' <-- %8s  batchsize=%3d  numepoch=%3d mean=%5.2f stdev=%5.2f exectime=%6.1f' %
                  (opt_type, batch_size, nepoch, scores.mean()*100.0, scores.std()*100.0, stop-start))

            fh.write("%s  & %d  & %d & %5.2f & %5.2f & %6.1f\\\\\n\hline\n" %
                     (opt_type, batch_size, nepoch, scores.mean()*100.0, scores.std()*100.0, stop-start))

            ymax = max(min(max(scores)*1.1, 1.0), ymax)
            ymin = min(max(min(scores)*0.9, 0.0), ymin)

            # Create count of the number of epochs
            scores_count = range(1, len(scores)+1 )

            # Visualize accuracy history
            plt.plot(scores_count, scores, linestyle)
            ls = opt_type + ', lr={:5.3f}'.format(lr)
            if opt_type == 'sgd':
               ls = '{}, mnt={:5.2f}'.format(ls, moment)
            legend.append(ls)

    stop = timer()
    print('\nExecution time for cross validation=%6.1f seconds' % (stop-start))
        
    plt.legend(legend, loc=9, bbox_to_anchor=(0.7, 0.35))
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy Score')
    axes = plt.gca()
    axes.set_ylim([ymin, ymax])
    axes.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    plt.title('Cross Validation for Kernel Classes (bs={0},ss={1},ep={2})'.format(batch_size, len(train_y), nepoch))
    plt.savefig('{0}\\CrossValidation4kclass-{1}.pdf'.format(data_dir,len(train_y) ))
#    plt.ylim([0.4, 1.0])
    plt.show();
    print("\n**Note: Created cross_validation.txt and CrossValidation4kclass.pdf")
    return

#    graph_model(history, predictive_model,
#                '{0}\\TrainTest-{1}-{2}.pdf'.format(data_dir, num_samples, opt_type),
#                'Optimizer: {0} (ss={1})'.format(opt_type, num_samples))