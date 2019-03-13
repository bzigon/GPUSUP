#
# Log2Pivot.py -- convert a performance counter log file to a format that we can train and test with.
#
from __future__ import print_function
import os
import pandas as pd
import sys

tkernels = ['KernelVectorAdd',               'KernelVectorAddCB',          'KernelVectorAddCBTrig',         'KernelVectorAddCBTrigILP2',
            'KernelVectorAddCBTrigILP2_64',  'KernelVectorAddCBTrigILP4',  'KernelVectorAddCBTrigILP4_128', 'KernelMatMultFast',
            'KernelStencil5', 'KernelStencil5SM',
            'KernelMatMult']

testkernels = ['KernelStencil9', 'KernelSpmv_csr_vector']

def include_column(colname):
   inc_list = [
#               "blksz_x",                    "blksz_y",                    "grdsz_x",              "grdsz_y",
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
#               "flop_count_sp",              "flop_count_sp_add",    "flop_count_sp_fma", "flop_count_sp_mul","sysmem_write_throughput", "sysmem_read_throughput",
               ]

               
#   inc_list = []
#   if colname[0:7] != 'Unnamed':
#      return True
#   return False
   
   if colname in inc_list:
      return True     
   return False
   
def compute_columns_to_include(input_file_name):
   print('\ncompute_columns::Reading ', input_file_name)
   df = pd.read_csv(input_file_name)
   include = []
   for col in df.columns:
      if col != 'kernel':
#         if col[0:7] != 'Unnamed':
          if include_column(col):
            max_value = df[col].max()
            min_value = df[col].min()
            if min_value != max_value:    # enable this line to eliminate columns of identical values
               include.append(col)
      elif col == 'kernel':
         include.append(col)

#    print('Selected columns are ..')
#    for c in include:
#        print(c)
   return include

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
def normalize(df, mean_list, stdev_list, columns):
   result = df.copy()
   mean_listo = []
   stdev_listo = []
   for feature_name in columns:
      if feature_name != 'kernel':
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if len(mean_list) == 0:     # list len == 0 --> training
              mean_value = df[feature_name].mean()
              stdev_value= df[feature_name].std()
              mean_listo.append(mean_value)
              stdev_listo.append(stdev_value)
            else:
              mean_value = mean_list.pop(0)
              stdev_value = stdev_list.pop(0)
              mean_listo.append(mean_value)
              stdev_listo.append(stdev_value)

            if max_value == min_value:
               result[feature_name] = 0
            else:
    #            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
                result[feature_name] = (df[feature_name] - mean_value) / stdev_value
#                print('mean=', result[feature_name].mean(), feature_name, mean_value, stdev_value)
#                print('stdev=', result[feature_name].std(), feature_name)
      else:
         result[feature_name] = df[feature_name]

   return result, mean_listo, stdev_listo

#
# Read a log file, extract the columns from column_list for kernels.
#
def read_log(input_file_name, column_list, kernels):
    print('read_log::Reading ', input_file_name)
    print('kernels = \n')
    for k in kernels:
        print('\t', k)

    df = pd.read_csv(input_file_name)
    dfcopy = df[df.kernel.isin(kernels)]
    dfnew = dfcopy[column_list]     # keep these columns

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

    return dfnew,ColumnsMatch


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
    print('apply_normalization')
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

class Quad():
    def __init__(self, train_in, train_val_out, test_in, test_out):
        self.train_in       = train_in
        self.train_val_out = train_val_out
        self.test_in         = test_in
        self.test_out        = test_out


######################################################################################################
##
######################################################################################################
#
# Make a pass across each file to find the intersection of the column names.
# We need to do this because the number of non-zero columns can change as the
# GPU array size changes (like l2_tex_read_hit_rate, tex_cache_transactions, etc)

if __name__ == '__main__':

   suffix_list = [Quad('-TRAIN', '-TRAIN', '-TEST', '-TEST')]
   prefix = "pivotlogs-titanv"
   DataDir = "DataForPaper-TitanV\\"

   columns = []
   columns_next = []
   DoUnion = True

   #
   # first process the training list
   #
   if DoUnion:
       print('***Computing union of column lists ...')
   else:
       print('***Computing intersection of column lists ...')

   for q in suffix_list:
       input_file = prefix + q.train_in + '.txt'
       data_dir = os.path.join(".", DataDir)

       full_input_file_name = os.path.join(data_dir, input_file)

       if len(columns) == 0:
           columns = compute_columns_to_include(full_input_file_name)
           print('\tFilename {0}, number of columns {1}'.format(input_file, len(columns)))
       else:
           columns_next = compute_columns_to_include(full_input_file_name)
           print('\tFilename {0}, number of columns {1}'.format(input_file, len(columns_next)))

           if DoUnion:
               columns_tmp  = [c for c in columns_next if c in columns]
               columns_tmp1 = [c for c in columns_next if c not in columns]
               columns_tmp2 = [c for c in columns if c not in columns_next]
               columns_tmp += columns_tmp1
               columns_tmp += columns_tmp2
           else:
               if len(columns) < len(columns_next):
                   columns_tmp = [c for c in columns if c in columns_next]
               else:
                   columns_tmp = [c for c in columns_next if c in columns]

           columns = columns_tmp

   #
   # now process the test list
   #
   if 1==0:
       for q in suffix_list:
           input_file = prefix + q.test_in + '.txt'
           data_dir = os.path.join("..", DataDir)
           full_input_file_name = os.path.join(data_dir, input_file)

           columns_next = compute_columns_to_include(full_input_file_name)
           print('\tFilename {0}, number of columns {1}'.format(input_file, len(columns_next)))

           if DoUnion:
               columns_tmp  = [test for test in columns_next if test in columns]
               columns_tmp1 = [test for test in columns_next if test not in columns]
               columns_tmp2 = [test for test in columns if test not in columns_next]
               columns_tmp += columns_tmp1
               columns_tmp += columns_tmp2
           else:
               if len(columns) < len(columns_next):
                   columns_tmp = [test for test in columns if test in columns_next]
               else:
                   columns_tmp = [test for test in columns_next if test in columns]

           columns = columns_tmp

   print('\nNumber of features (columns) ', len(columns))
   print('Number of classes ', len(tkernels))


   data_dir = os.path.join(".", DataDir)

   print('\n\n***Pivot logs, split data into training and validation sets, then normalize ...')
   for q in suffix_list:
       input_file = prefix + q.train_in + '.txt'
       log_df, ColumnsMatch = read_log(os.path.join(data_dir, input_file), columns, tkernels)

       if ColumnsMatch == False:
           print("**Error -- Columns don't match in ", input_file)
           sys.exit(1)

       input_file = prefix + q.test_in + '.txt'
       test_df, ColumnsMatch = read_log(os.path.join(data_dir, input_file), columns, testkernels)

       if ColumnsMatch == False:
           print("**Error -- Columns don't match in ", input_file)
           sys.exit(1)

       train_df, val_df = split_into_train_and_validate(log_df, 0.90)
       norm_result = compute_normalization(train_df, columns)

       apply_normalization(train_df, norm_result)
       apply_normalization(test_df, norm_result)
       apply_normalization(val_df, norm_result)
       apply_normalization(log_df, norm_result)

       write_dataset(train_df, columns, os.path.join(data_dir, 'train.txt'))
       write_dataset(val_df, columns, os.path.join(data_dir,  'validate.txt' ))
       write_dataset(test_df, columns, os.path.join(data_dir,  'test.txt' ))
       write_dataset(log_df, columns, os.path.join(data_dir,  'trainANDvalidate.txt' ))



   print('Done')
