#
# 1.py
#
from __future__ import print_function
import pandas as pd
import csv, sys, os
import glob
import PivotLogs as PL

#
# DoMerge will read a bunch of input files and write them to a single ouput file.
# Each line in an input file represents the profiler results for 1 kernel.
#
# If maxLineCount > 0, then we will only write the first maxLineCount files to the
# ouput file. We will mainly use this for the testing files so that we don't processes
# hundreds or thousands of lines.

def DoMerge(infiles, outfile, maxLineCount = 0):
   fhout = open(outfile, "w")

   try:
      for fcounter, f in enumerate(infiles):
         with open(f, "r") as fhin:
            lc = 0
            for line in fhin:
               if fcounter == 0:    # write every line of the first file
                  fhout.write(line)
               elif lc != 0:        # skip the first line of the other files
                  fhout.write(line)
               lc += 1
               
               if maxLineCount > 0:
                  if lc > maxLineCount:
                     break
                     
            print("File name {} wrote {} lines to {}".format(f, lc, outfile))
      
   except (FileNotFoundError) as e:
     sys.exit('\n***\n*** File Not Found Error in DoMerge {}***'.format(e.filename))
   fhout.close()

#trainkernels = ['KernelVectorAdd',               'KernelVectorAddCB',          'KernelVectorAddCBTrig',         'KernelVectorAddCBTrigILP2',
#            'KernelVectorAddCBTrigILP2_64',  'KernelVectorAddCBTrigILP4',  'KernelVectorAddCBTrigILP4_128', 'KernelMatMultFast',
#            'KernelStencil5', 'KernelStencil5SM',
#            'KernelMatMult']
#
#testkernels = ['KernelStencil9', 'KernelSpmv_csr_vector']


def include_column(colname):
   inc_list = [
               "sm_efficiency",              "achieved_occupancy",         "ipc",                  "issued_ipc", 
               "warp_execution_efficiency",  "branch_execution_efficiency","inst_replay_overhead", "warp_nonpred_execution_efficiency", 
               "shared_store_transactions",  "shared_load_transactions",   "shared_store_transactions_per_request", "shared_load_transactions_per_request", 
               "global_hit_rate",            "local_hit_rate",             "gld_throughput",       "gst_throughput", 
               "dram_read_throughput",       "dram_write_throughput",      "tex_cache_throughput", "shared_load_throughput", 
               "shared_store_throughput",    "gld_transactions_per_request", "gst_transactions_per_request",
               "shared_efficiency",          "inst_executed_shared_stores", "stall_inst_fetch", 
               "stall_exec_dependency",      "stall_memory_dependency",    "stall_texture",        "stall_sync", 
               "stall_other",                "stall_constant_memory_dependency", "stall_pipe_busy","stall_memory_throttle", 
               "stall_not_selected",         "tex_cache_hit_rate",         "l2_tex_read_hit_rate", "l2_tex_write_hit_rate", 
               "l2_read_throughput",         "l2_write_throughput",         
               "l2_tex_hit_rate",            "eligible_warps_per_cycle",   "flop_sp_efficiency",   "duration",
               ]

   return (colname in inc_list)
   
 #  if colname in inc_list:
 #     return True     
 #  return False
   
def compute_columns_to_include(input_file_name):
#   print('****compute_columns_to_include::Reading ', input_file_name)
   df = pd.read_csv(input_file_name)
   include = []
   
   for col in df.columns:
      if col != 'kernel':
          if include_column(col):
            max_value = df[col].max()
            min_value = df[col].min()
            if min_value != max_value:    # enable this line to eliminate columns of identical values
               include.append(col)
      elif col == 'kernel':
         include.append(col)

   return include
   
#
# Read a log file, extract the columns from column_list for kernels.
#
def read_log(input_file_name, column_list):
#    print('read_log::Reading ', input_file_name)
#    print('kernels = \n')
#    for k in kernels:
#        print('\t', k)

    df = pd.read_csv(input_file_name)
#   dfcopy = df[df.kernel.isin(kernels)]
#   dfnew = dfcopy[column_list]     # keep these columns
    dfnew = df[column_list]      # keep these columns
    T = dfnew.groupby('kernel').size()    # Pandas Series with kernel name, count
    
    #
    # Verify the the log file has the same number of columns as in the
    # column_list and their names are identical.
    #
    ColumnsMatch = True
    if len(dfnew.columns) != len(column_list):
        ColumnsMatch = False
    else:
        for c1,c2 in zip(dfnew.columns,column_list):
            if c1 != c2:
                ColumnsMatch = False

    return dfnew, ColumnsMatch, len(T)
    
def split_into_train_and_validate(df, train_percent):
   dfcopy = df.sample(frac=1.0).sample(frac=1.0)   # shuffle all of the rows of the dataframe
   train_rows = int(dfcopy.shape[0] * train_percent)
   train = dfcopy[0:train_rows]
   validate  = dfcopy[train_rows:]
   return train,validate

#
# If the columns are range normalized, X'_i = X_i - min_i{X_i}
#                                             ================
#                                             max_i{X_i} - min_i{X_i}
#
#  then X_min ->0, X_max -> 1,  so the first layer of your model should be RELU.
#
#
# If the columns are standard score normalized, X'_i = X_i - mu
#                                                    ===========
#                                                       sigma
#
#  then the resulting data has mean=0 and sigma=1.
#
#  In this case, the first layer should be TANH or SIGMOID.
#
def compute_normalization(df, columns):
    result = []

    for feature_name in columns:
        min_value   = 0
        max_value   = 0
        mean_value  = 0
        stdev_value = 0

        if feature_name != 'kernel':
            min_value   = df[feature_name].min()
            max_value   = df[feature_name].max()
            mean_value  = df[feature_name].mean()
            stdev_value = df[feature_name].std()

        result.append([feature_name, min_value, max_value, mean_value, stdev_value])

    return result

def apply_normalization(df, norm_result):
#    print('apply_normalization')
    for fea_name,fea_min,fea_max,fea_mean,fea_std in norm_result:
        if fea_min!=fea_max:
            df.loc[:,fea_name] = (df.loc[:, fea_name] - fea_mean) / fea_std
#            print('Normalizing to range [mu, sigma]')
#            df.loc[:,fea_name] = 2.0*((df.loc[:, fea_name] - fea_min) / (fea_max-fea_min)) - 1.0
#            print('Normalizing to range [-1,1]')
#            print(fea_name, fea_min, fea_max, fea_mean, fea_std)
    return df

def write_dataset(df, columns, file_name):
    print('write_dataset::Writing ', file_name)
    df.to_csv(file_name, float_format='%8.5f', columns=columns, index=False)
    return
    

def DoProcessColumns(trainingInputFile, DoUnion, DataDir, testingInputFile):
   print('\n****DoProcessColumns')
   if DoUnion:
       print('****Computing union of column lists ...')
   else:
       print('****Computing intersection of column lists ...')

   columns = []
   columns = compute_columns_to_include(trainingInputFile)
   
   data_dir = os.path.join(".", DataDir)
#   print('\n***Split data into training and validation sets, then normalize.')

   log_df, ColumnsMatch, numTrainingKernels = read_log(trainingInputFile, columns)

   if ColumnsMatch == False:
       print("**Error -- Columns don't match in ", input_file)
       sys.exit(1)

   test_df, ColumnsMatch, numTestingKernels = read_log(testingInputFile, columns)

   if ColumnsMatch == False:
       print("**Error -- Columns don't match in ", input_file)
       sys.exit(1)

   print('\nFilename {0} has {1} columns '.format(trainingInputFile, len(columns)))
   print('Number of features (columns) {}'.format(len(columns)))
   print('Number of classes {}\n'.format(numTrainingKernels))
   
   train_df, val_df = split_into_train_and_validate(log_df, 0.90)
   norm_result = compute_normalization(train_df, columns)

   apply_normalization(train_df, norm_result)
   apply_normalization(test_df, norm_result)
   apply_normalization(val_df, norm_result)
   apply_normalization(log_df, norm_result)

   write_dataset(train_df, columns, os.path.join(data_dir, 'train.txt'))
   write_dataset(val_df, columns, os.path.join(data_dir,  'validate.txt' ))
   write_dataset(test_df, columns, os.path.join(data_dir,  'test.txt' ))
   write_dataset(log_df, columns, os.path.join(data_dir,  'train_validate.txt' ))

#
# Main Program Logic
#
if __name__ == "__main__":

   # PivotLogs will parse a directory full of performance log files, 
   # pivot them, and write them out.
   # It's just a way of combining all of the profile data into a single file.
   
   PL.PivotLogs('DataForPaper-TitanV\\0LogsTitanV-Stencil5\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-Stencil5.txt')
   PL.PivotLogs('DataForPaper-TitanV\\0LogsTitanV-Stencil5SM\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-Stencil5SM.txt')

   # TO DO
   # Create a function called PivotAllDirectories
   #
   #
   # DoMerge will merge the individual .TXT files into a single .TXT so that
   # we can create the Train.txt, Validate.txt and Test.txt.
   
   inputTxtFilesForTraining = [
         "DataForPaper-TitanV\\0pivotlogs-titanv-1M.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-10M.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-100M.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-FMM.txt",
         "DataForPaper-TitanV\\1pivotlogs-titanv-FMM.txt",
         "DataForPaper-TitanV\\1pivotlogs-titanv-MM.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-Stencil5.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-Stencil5SM.txt",         
         ]
   
   trainingTxtFile = "DataForPaper-TitanV\\pivotlogs-titanv-TRAIN.txt"

   inputTxtFilesForTesting = [
         "DataForPaper-TitanV\\0pivotlogs-titanv-Stencil9.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-SpMV05.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-Stencil9SM.txt"
         ]
   
   testingTxtFile = "DataForPaper-TitanV\\pivotlogs-titanv-TEST.txt"

   DoMerge(inputTxtFilesForTraining, trainingTxtFile)
   DoMerge(inputTxtFilesForTesting, testingTxtFile, 10)
   
   DoProcessColumns(trainingTxtFile, True, "DataForPaper-TitanV\\", testingTxtFile)