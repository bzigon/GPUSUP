
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import figure

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from timeit import default_timer as timer
import os, sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
import pickle

def create_base_network(NumberOfFeatures, NumberOfClasses,init_mode='glorot_normal'):
    """
    This function will create the base network and return the model.

    This is a helper function that the other network creation functions will call.
    """
    network = Sequential()
    network.add(Dense(NumberOfFeatures, activation='relu', kernel_initializer=init_mode,input_dim=NumberOfFeatures))
#    network.add(Dense(NumberOfFeatures, activation='relu',kernel_initializer=init_mode))
    network.add(Dense(NumberOfFeatures/2, activation='relu',kernel_initializer=init_mode))
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
      print(df.groupby('kernel').size())
      
    encoded_Y = encoder.transform(dfy)
    encoded_Y = encoded_Y.astype(np.float32)    # the labels are now an integer (e.g. KernelAdd = 5)
    onehot_Y = to_categorical(encoded_Y)
    onehot_Y = onehot_Y.astype(np.float32)      # the integer is now in binary format (e.g. KernelAdd = 0101)
    
    return X, onehot_Y, encoded_Y, dfy, df.columns.tolist()

# *********************************************************************************************
# *
# *********************************************************************************************
def plot_confusion_matrix(y_true, y_pred, confuse_file, classes, 
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    
#    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

#    plt.figure(figsize=(8,8))
    fig, ax = plt.subplots(1,1,figsize=(10,8))
   
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '3.2f' if normalize else '3d'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
#    plt.savefig(confuse_file + '.png')
  
    return ax
    
# *********************************************************************************************
# *
# *********************************************************************************************
def perform_training(whichModel,
                     nFeatures,
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
                     lr_decay, dt):

   train_x, train_y, encoded_y, none, none  = read_samples(input_train_file)
   val_x, val_y, none, none, none = read_samples(validation_file)

   if whichModel == 1:
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
                   '{3}-TrainValAcc-{1}-{2}.pdf'.format(data_dir, num_samples, optimizer_type, dt),
                   '{3}-TrainValLoss-{1}-{2}.pdf'.format(data_dir, num_samples, optimizer_type, dt),
                   'Neural Network Training')
#                   'Optimizer: {0} (ss={1})'.format(optimizer_type, num_samples))
       print("Writing model %s" % mdlname)
       model.save(mdlname)
    
   if whichModel == 2:
      model = RandomForestClassifier(n_estimators=500, max_depth=6).fit(train_x, train_y)
      pickle.dump(model, open(mdlname, 'wb'))
      
   if whichModel == 3:
      model = GaussianNB().fit(train_x, encoded_y)
      pickle.dump(model, open(mdlname, 'wb'))
      
   return

#**********************************************************************************************
#*
# *********************************************************************************************

#
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#

def get_training_encoder(file_name):
   df = pd.read_csv(file_name)
   dfy = df.loc[:,'kernel'].values

   encoder = preprocessing.LabelEncoder()
   encoder.fit(dfy)
    
   return encoder
      

def perform_confusion(whichModel, nClasses, mdlname, confusion_file, training_file, dt):
   print('\n\nCompute confusion matrix\nLoading model {} for testing '.format(mdlname))

   if whichModel == 1:
      predictive_model = load_model(mdlname)   
      x, onehot_y, encoded_y, knames, _ = read_samples(confusion_file, False)
      pred_y = predictive_model.predict(x)
      title = 'Neural Network Confusion Matrix'
      confuse_file = 'NN'
    
   if whichModel == 2:
      predictive_model = pickle.load(open(mdlname, 'rb'))
      x, onehot_y, encoded_y, knames, _ = read_samples(confusion_file, False)
      pred_y = predictive_model.predict(x)
      title = 'Random Forrest Confusion Matrix'
      confuse_file = 'RF'
      
   if whichModel == 3:
      predictive_model = pickle.load(open(mdlname, 'rb'))
      x, onehot_y, encoded_y, knames,_ = read_samples(confusion_file, False)
      pred_y = predictive_model.predict(x)
#      pred_y_proba = predictive_model.predict_proba(x)
      pred_y = predictive_model.predict_proba(x)
      title = 'naive Bayes Confusion Matrix'
      confuse_file = 'NB'
      
   np.set_printoptions(precision=2)

   # Plot non-normalized confusion matrix
   encoded_y = encoded_y.astype(int)
   pylist = np.ndarray(encoded_y.shape, dtype=int)
   
   for i,py in enumerate(pred_y):
      pyc = np.argmax(py).astype(int)
      pylist[i]=pyc
   
   training_encoder = get_training_encoder(training_file)
   eclasses = training_encoder.classes_
 
   eclasses = np.array(['MatMult (K7)','MatMultFast (K8)','Stencil5 (K9)', 'Stencil5SM (K10)', 'Add (K1)','AddCB (K2)','AddCBTrig (K3)',
                              'AddCBTrigILP2 (K4)','AddCBTrigILP2_64 (K5)','AddCBTrigILP4_128 (K6)'], dtype=object)
                              
#   eclasses = np.array(['MatMult','MatMultFast','Stencil5', 'Stencil5SM', 'Add','AddCB','AddCBTrig',
#                              'AddCBTrigILP2','AddCBTrigILP2_64','AddCBTrigILP4','AddCBTrigILP4_128'], dtype=object)
 

   plot_confusion_matrix(encoded_y, pylist, confuse_file, eclasses, normalize=False,
                      title=title)

   plot_confusion_matrix(encoded_y, pylist, confuse_file, eclasses, normalize=True,
                      title=title)

   plt.savefig("{0}-{1}-confusion_matrix.pdf".format(mdlname, confuse_file))
   plt.show()

   classification_file = '{0}-{1}-classification_report.txt'.format(dt, confuse_file)
   with open(classification_file, 'w') as f:
      f.write(classification_report(encoded_y, pylist, target_names=eclasses))
          
   print('\nClassification file was written %s' % classification_file)
   
# *********************************************************************************************
# *
# *********************************************************************************************
def perform_testing(whichModel, nClasses, mdlname, testing_file, dt, training_file):
   print('\n\nTest the model\nLoading model {} for testing '.format(mdlname))
    
   training_encoder = get_training_encoder(training_file)
   eclasses = training_encoder.classes_
   
   if whichModel == 1:
      perform_NNtesting(nClasses, mdlname, testing_file, dt, eclasses)

   if whichModel == 2:
       perform_DecisionTreeTesting(nClasses, mdlname, testing_file, dt, eclasses)

   if whichModel == 3:
       perform_NaiveBayesTesting(nClasses, mdlname, testing_file, dt, eclasses)
   
# https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
# https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough

def keyFunc(e):
    return e[1]

def show_feature_importance(cnames, model):
    L = list(zip(cnames, model.feature_importances_))
    L.sort(reverse=True, key=keyFunc)
    for i, a in enumerate(L):
        print('{:3d}, {:8.5f}, {}'.format(i, a[1], a[0]))

def perform_NNtesting(nClasses, mdlname, testing_file, dt, eclasses):
    predictive_model = load_model(mdlname)
    test_x, _, _, kname, none = read_samples(testing_file, False)
    classes = predictive_model.predict(test_x)

    show_results(nClasses, classes, kname, "\nNeural Network",  dt+'-NN-TestResults.txt', eclasses)
    return
    
def perform_DecisionTreeTesting(nClasses, mdlname, testing_file, dt, eclasses):
    model = pickle.load(open(mdlname, 'rb'))
    test_x, _, _, kname, cnames = read_samples(testing_file, False)
    classes = model.predict(test_x)

    cnames.remove("kernel")
    show_feature_importance(cnames, model)

    estimator = model.estimators_[5]
   # Export as dot file
    export_graphviz(estimator, 
                out_file=dt+'-tree.dot', 
                feature_names = cnames,
                class_names = kname,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

    
    from subprocess import call
    call(['dot', '-Tpng', dt+'-tree.dot', '-o', dt+'-tree.png', '-Gdpi=200'], shell=True)

    show_results(nClasses, classes, kname, "\nRandom Forest",  dt+'-RF-TestResults.txt', eclasses)
    return
 
def perform_NaiveBayesTesting(nClasses, mdlname, testing_file, dt, eclasses):
    model = pickle.load(open(mdlname, 'rb'))
    test_x, _, _, kname, cnames = read_samples(testing_file, False)
    classes = model.predict(test_x)
    classes_proba = model.predict_proba(test_x)

    show_results(nClasses, classes_proba, kname, "\nnaive Bayes", dt+'-NB-TestResults.txt', eclasses)
    return
    
def show_results(nClasses, classes, kname, title, output_file, eclasses):
    with open(output_file, 'w') as f:
      f.write('Encoder classes\n\n')
      for i, c in enumerate(eclasses):
         f.write('Value = {0} Class = {1}\n'.format(i, c))
         
      f.write('\n'+title+"\n")
      f.write("    ")
      for i in range(nClasses):
         f.write(' {:2d}'.format(i))
      f.write("\n")
      f.write("     ")
      for i in range(nClasses):
         f.write('==='.format(i))
      f.write("\n")
    
      old_kname = ''
      for i in range(len(classes)):
         v = list('0' * nClasses)
         c = np.argmax(classes[i])
         v[c] = '1'
         if old_kname != kname[i]:
            f.write('\n')
            old_kname = kname[i]
            
         f.write('%4d: ' % (i))
         f.write('  '.join(v))
         f.write(' pred cls=%d' % (c))
         f.write(' inference kname=%s\n' % kname[i])
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

#    train_x, train_y, none  = read_samples(input_train_file)
    train_x, train_y, encoded_y, knames, _ = read_samples(input_train_file, False)
      
    network_parameters = [
    [create_neural_network, 'rmsprop',  NumberOfEpochs,  0.001, 0.0, 'r-,'],
    [create_neural_network, 'sgd',      NumberOfEpochs,  0.002, 0.9, 'b--o'],
    [create_neural_network, 'adagrad',  NumberOfEpochs,  0.010, 0.0,'g-.v'],
    [create_neural_network, 'adadelta', NumberOfEpochs,  1.000, 0.0,'c:^'],
    [create_neural_network, 'adam',     NumberOfEpochs,  0.001, 0.0,'m-s'],
    [create_neural_network, 'adamax',   NumberOfEpochs,  0.002, 0.0,'k--*'],
    [create_neural_network, 'nadam',    NumberOfEpochs,  0.002, 0.0,'y-.x']
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
        
    plt.legend(legend, loc=9, bbox_to_anchor=(0.7, 0.45))
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy Score')
    axes = plt.gca()
    axes.set_ylim([ymin, ymax])
    axes.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

#   plt.title('Cross Validation for Kernel Classes (bs={0},ss={1},ep={2})'.format(batch_size, len(train_y), nepoch))
    plt.title('Cross Validation for Kernel Classes')
    plt.savefig('{0}\\CrossValidation4kclass-{1}.pdf'.format(data_dir,len(train_y) ))
#    plt.ylim([0.4, 1.0])
    plt.show();
    print("\n**Note: Created cross_validation.txt and CrossValidation4kclass.pdf")
    return

#    graph_model(history, predictive_model,
#                '{0}\\TrainTest-{1}-{2}.pdf'.format(data_dir, num_samples, opt_type),
#                'Optimizer: {0} (ss={1})'.format(opt_type, num_samples))
