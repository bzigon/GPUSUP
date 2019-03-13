#
# plearn3.py
#
# This python file will either train a model or create a new model
# using a genetic algorithm.
#
# Bob Zigon, rzigon@iupui.edu
#

import os
import logging
from perf_train import perform_training, perform_testing, perform_crossvalidation, perform_confusion
from timeit import default_timer as timer
from datetime import datetime
import sys

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)
 
def get_datetime_label():
      now = datetime.now()
      S = now.strftime("%Y%j-")
      seconds_since_midnight = int((now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())
      return "{0}{1}".format(S, seconds_since_midnight)
      
######################################################################################################
##
######################################################################################################
#
# Sparse Matrices  http://www.mathcs.emory.edu/~cheung/Courses/561/Syllabus/3-C/sparse1.html
#
#
if __name__ == '__main__':

    data_dir        = "DataForPaper-TitanV"
    NumberOfEpochs  = 350
    learning_rate   = 0.001       # lr=1.0 for adadelta
    nFeatures       = 29-1    # was 44, 109, 46
    nClasses        = 10
    batch_size      = 32
    learning_rate_decay = learning_rate / NumberOfEpochs    # for SGD
    
    input_train_file = os.path.join(data_dir, 'train.txt')
    input_test_file = os.path.join(data_dir, 'test.txt')
    validation_file = os.path.join(data_dir, 'validate.txt')
    input_train_and_validate_file = os.path.join(data_dir, 'train_validate.txt')
    input_confusion_file = os.path.join(data_dir, 'confuse.txt')
    
    print("\nData Directory {}".format(data_dir))
    dt = get_datetime_label()

    
    if True:
        start = timer()
#       for optimizer_type in ['rmsprop','adagrad', 'adadelta','adam', 'adamax', 'nadam']:    # sgd, adadelta, adagrad, adagrad, rmsprop, nadam
        for optimizer_type in ['nadam']:    #  sgd, adadelta, adagrad, adamax, rmsprop, nadam
            
            whichModel = 3
            
            if whichModel == 1:
               modelname = 'gpumodel.' + optimizer_type
            elif whichModel == 2:
               modelname = 'gpumodel.dectree'
            elif whichModel == 3:
               modelname = 'gpumodel.naibayes'
            else:
               print('**Invalid model ** \n')
               sys.exit()
               
            modelname = dt + '-' + modelname
            
            if True:
               perform_training(whichModel, 
                                    nFeatures,                     # number of features
                                    nClasses,                      # number of classes
                                    modelname,
                                    input_train_file,
                                    validation_file,
                                    batch_size,                    # batch size
                                    NumberOfEpochs,
                                    optimizer_type,
                                    data_dir,
                                    learning_rate,
                                    0.90,                           # momentum for SGD
                                    learning_rate_decay, dt)        # learning rate decay for SGD

            perform_testing(whichModel,
                            nClasses,
                            modelname,
                            input_test_file, 
                            dt, input_train_file)
                            
            perform_confusion(whichModel,
                            nClasses,
                            modelname,
                            input_confusion_file,
                            input_train_file, dt)
                            
        stop = timer()
        print('\nExecution time for training and testing=%6.1f seconds' % (stop-start))

    if False:
        perform_crossvalidation(nFeatures,
                                       nClasses,
                                       NumberOfEpochs,
                                       7,               # number of cross validations
                                       input_train_and_validate_file,
                                       data_dir,
                                       batch_size,
                                       learning_rate_decay)


    if False:
        start = timer()
        print("DO GRID SEARCH")
        stop = timer()
        print('\nExecution time for grid search=%6.1f seconds' % (stop-start))

    # Note : LOG.TXT has the output in it when you run the genetic evolution.
    # This currently takes about 5 hours on a TitanV.
    #if False:
    #    start = timer()
    #
    #   datasets = {
    #       'train': input_train_file,
    #       'validate': validation_file
    #   }
    #
    #   # Set all the parameters in the body of the function perform_genetic_evolution.
    #   perform_genetic_evolution(datasets)
    #   stop = timer()
    #   print('\nExecution time for genetic evolution=%6.1f seconds' % (stop-start))




