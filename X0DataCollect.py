#
# DataCollect.py -- this script will read a configuration file and then
#                   execute and log all of the profiler results.
#
import csv, sys
import subprocess
import time
from datetime import datetime

#
# Note : Log files are local to the directory you are running this
#        script from.
#

def RunAllKernels(config_filename, logFileTemplate, logFileDurationTemplate, testNumber):
    print("Opening file ", config_filename)
    
    # Count the number of executable lines in the configuration file.
    totalTestCount = 0
    with open(config_filename, 'r') as f:
       for line in f:
           words = line.split(':')
           kernelName = words[0]
           if kernelName[0] != '#' or kernelName[0] == '\n':
               totalTestCount += 1


    failCount = 0
    successCount = 0
    failCountDuration = 0
    successCountDuration = 0

    with open(config_filename, 'r') as f:
       for line in f:
           #    print(line,end='') # note, coma erases the "cartridge return"
           words = line.split(':')

           # Skip over the comment lines or blank lines.
           kernelName = words[0]
           if kernelName[0] == '#' or kernelName[0] == '\n':
               continue

           # A line will look like one of these.
           #
           # KernelVectorAdd:   compile: params -bs   32,1,1 -gs   64,1,1 -numele 1000000
           # KernelVectorAdd: nocompile: params -bs   64,1,1 -gs   64,1,1 -numele 1000000
           #
           # word[0] - kernel name
           # word[1] - The compile/nocompile switch is next. This was added in case a module needed to be compiled.
           # word[2] - params keyword
           # word[3] - bs  .. the block size keyword
           # word[4] - x,y,z .. the block size triple
           # word[5] - gs  .. the grid size keyword
           # word[6] - x,y,z .. the block size triple
           # word[7] - numele .. the number of elements keyword
           # word[8] - the number of elements

           compileOpt = words[1]
           paramsOpt = words[2]

           if compileOpt[0:10] == "   compile":
               print('*** Code compilation occurs here.')

           if paramsOpt[0:7] == " params":
               paramString = paramsOpt[7:len(paramsOpt) - 1]
               
               # Execute the nv profiler to generate all of the metrics.
               
               testNumber += 1
               logFile = logFileTemplate.format(testNumber)
               kernelstring = '--kernels ' + kernelName[0:1].lower() + kernelName[1:]
               execString = 'nvprof ' + kernelstring + ' --metrics all --log-file ' + logFile + ' --csv  --profile-api-trace none DataGen.exe -x ' + kernelName + ' ' + paramString
               print(execString); retcode = 0;
               retcode = subprocess.call(execString)

               print("\nTest Number {:d}-1 of {:d}, return code = {:d} ".format(testNumber, totalTestCount, retcode))
               if retcode == 0:
                   successCount += 1
               else:
                   failCount += 1

               # Execute the nv profiler to compute the duration of the kernel. This can not be done at the same
               # time the kernel metrics are being generated.
               
               logFileDuration = logFileDurationTemplate.format(testNumber)
               execString = 'nvprof  --log-file ' + logFileDuration + ' --csv  --profile-api-trace none DataGen.exe -x ' + kernelName + ' ' + paramString
               print(execString); retcode = 0;
               retcode = subprocess.call(execString)

               print("\nTest Number {:d}-2 of {:d}, return code = {:d} ".format(testNumber, totalTestCount, retcode))
               if retcode == 0:
                   successCountDuration += 1
               else:
                   failCountDuration += 1
                   
               print("Success count = {:d} Fail Count = {:d}".format(successCount, failCount))
               print("Success Duration count = {:d} Fail Duration Count = {:d}\n".format(successCountDuration, failCountDuration))

    #       if testNumber == 1:
    #           break

    return testNumber, successCount, failCount, successCountDuration, failCountDuration

def DoMain(filename, logFileTemplate, logFileDurationTemplate, testNumber):
   startTime = time.time()
   testNumber, successCount, failCount, successCountDuration, failCountDuration = RunAllKernels(filename, logFileTemplate, logFileDurationTemplate, testNumber)
   stopTime = time.time()
   deltaTime = stopTime - startTime

   print("\nTotal execution time for {:d} tests is {:f} seconds.".format(testNumber, deltaTime))
   print("Success count = {:d} Fail Count = {:d}".format(successCount, failCount))
   print("Success Duration count = {:d} Fail Duration Count = {:d}\n".format(successCountDuration, failCountDuration))
   print("Stop Time {}".format(str(datetime.now())))

   
#
# Main Program Logic
#
if __name__ == "__main__":


   #filename = 'TestConfigMM60.txt'
   #logFileTemplate = 'DataForPaper-TitanV\\1LogsTitanV-MM\log{:05d}.txt'
   #logFileDurationTemplate = 'DataForPaper-TitanV\\1LogsTitanV-MM\dur{:05d}.txt'
   #DoMain(filename, logFileTemplate, logFileDurationTemplate, 2000)

   #filename = 'TestConfigFMM60.txt'
   #logFileTemplate = 'DataForPaper-TitanV\\0LogsTitanV-FMM\log{:05d}.txt'
   #logFileDurationTemplate = 'DataForPaper-TitanV\\0LogsTitanV-FMM\dur{:05d}.txt'
   #DoMain(filename, logFileTemplate, logFileDurationTemplate, 9000)

   #filename = 'TestConfigILP4-1M10M100M.txt'
   #logFileTemplate = 'DataForPaper-TitanV\\0LogsTitanV-ILP4\log{:05d}.txt'
   #logFileDurationTemplate = 'DataForPaper-TitanV\\0LogsTitanV-ILP4\dur{:05d}.txt'
   #DoMain(filename, logFileTemplate, logFileDurationTemplate, 0)

   
   #filename = 'TestConfigStencil9SM.txt'
   #logFileTemplate = 'DataForPaper-TitanV\\0LogsTitanV-Stencil9SM\log{:05d}.txt'
   #logFileDurationTemplate = 'DataForPaper-TitanV\\0LogsTitanV-Stencil9SM\dur{:05d}.txt'
   #DoMain(filename, logFileTemplate, logFileDurationTemplate, 0)

   filename = 'TestConfigStencil5-120.txt'
   logFileTemplate = 'DataForPaper-TitanV\\1LogsTitanV-Stencil5\log{:05d}.txt'
   logFileDurationTemplate = 'DataForPaper-TitanV\\1LogsTitanV-Stencil5\dur{:05d}.txt'
   DoMain(filename, logFileTemplate, logFileDurationTemplate, 2000)

   filename = 'TestConfigStencil5SM.txt'
   logFileTemplate = 'DataForPaper-TitanV\\0LogsTitanV-Stencil5\log{:05d}.txt'
   logFileDurationTemplate = 'DataForPaper-TitanV\\0LogsTitanV-Stencil5\dur{:05d}.txt'
   DoMain(filename, logFileTemplate, logFileDurationTemplate, 1000)



