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
#from perf_genetic import perform_genetic_evolution
from perf_train import perform_training, perform_testing, perform_crossvalidation
from timeit import default_timer as timer

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

######################################################################################################
##
######################################################################################################
#
# Sparse Matrices  http://www.mathcs.emory.edu/~cheung/Courses/561/Syllabus/3-C/sparse1.html
#
#
if __name__ == '__main__':

    data_dir        = "DataForPaper-TitanV"
    NumberOfEpochs  = 10
    learning_rate   = 0.001       # lr=1.0 for adadelta
    nFeatures       = 44-1    # was 44
    nClasses        = 11
    batch_size      = 32
    learning_rate_decay = learning_rate / NumberOfEpochs    # for SGD
    
    input_train_file = os.path.join(data_dir, 'train.txt')
    input_test_file = os.path.join(data_dir, 'test.txt')
    validation_file = os.path.join(data_dir, 'validate.txt')
    input_train_and_validate_file = os.path.join(data_dir, 'trainANDvalidate.txt')
    print("\nData Directory {}".format(data_dir))
    
    if True:
        start = timer()
#       for optimizer_type in ['rmsprop','adagrad', 'adadelta','adam', 'adamax', 'nadam']:    # sgd, adadelta, adagrad, adagrad, rmsprop, nadam
        for optimizer_type in ['nadam']:    #  sgd, adadelta, adagrad, adamax, rmsprop, nadam
            modelname = 'gpumodel.' + optimizer_type

            if True:
               perform_training(nFeatures,                     # number of features
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
                                    learning_rate_decay)            # learning rate decay for SGD

            perform_testing(nClasses,
                            modelname,
                            input_test_file)
                            
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




