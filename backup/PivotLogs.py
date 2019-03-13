#
# PivotLogs.py -- this script will parse a directory full of performance log
#               files, pivot them, and write them out.
#
import csv, sys
import glob
from io import StringIO

class Triple:
    def __init__(self):
        self.name=""
        self.desc=""
        self.value=0

def AddTriple(n, d, v, tl):
    T = Triple()
    T.name = n
    T.desc = d
    T.value = v
    tl.append(T)

######################################################################################################
##
######################################################################################################
def ParseDuration(fname, TripleList):
    totalLineCountDur = 0
    actuaLineCountDur = 0
    headerFound = -1
    avgNameFound = -1
    nameFound = -1
    success = 0

    with open(fname, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            totalLineCountDur += 1
            T = Triple()

            if headerFound == 1:
                c = 0
                for col in row:
                    c = c + 1
                    if c == avgNameFound:
                        T.name = 'duration'
                        T.desc = 'duration'
                        T.value = col
                        #                           print('"', col, '"', ",", sep="", end="")
                    elif c == nameFound:
                        c1 = col.lower()
                        if c1[0:6] == 'kernel' or c1[0:11] == 'void kernel':
                            TripleList.append(T)
                            success = 1
            else:
                c = 0
                for col in row:
                    c = c + 1
                    if col == 'Time(%)':
                        headerFound = 1
                    elif col == 'Avg':
                        avgNameFound = c
                    elif col == 'Name':
                        nameFound = c

    return actuaLineCountDur, totalLineCountDur, TripleList, success

######################################################################################################
##
######################################################################################################
def ParseLog(fname):
    headerFound = -1
    metricNameFound = -1
    metricDescriptionFound = -1
    metricValueFound = -1
    totalLineCount = 0
    actuaLineCount = 0
    success = 0

    TripleList = []

    with open(fname, 'r') as f:
#           This is kind of strange. I open the file as a text file.
#           If it's the first line, I want to split it on spaces so I can retrieve bs, gs and numele.
#           If the header isnt found just yet, then split on commas.
#           If the header has been found, the split like a csv.
#
            for row in f:
                totalLineCount += 1
                T = Triple()

                if headerFound == 1:
                    buff = StringIO(row)
                    reader = csv.reader(buff)
#                    cols = row.rstrip('\n').split(',')
                    for line in reader:
                        #    print(row)
                        c = 0
                        for col in line:
                            if "_utilization" in col:   # skip any word with _utilization in it
                                break

                            c = c + 1
                            if c == metricNameFound:
                                T.name = col
    #                           print('"', col, '"', ",", sep="", end="")
                            elif c == metricDescriptionFound:
                                T.desc = col
    #                            print('"', col, '"', ",", sep="", end="")
                            elif c == metricValueFound:
                            #    print(col)
                                strValue = ""
                                for achar in col:
                                    if achar not in ['%','G','B','M','K','/','S','s'] :
                                        #print(achar, end="")
                                        strValue += achar
    #                            print(strValue)
                                actuaLineCount += 1
                                T.value=strValue
                                TripleList.append(T)
                                success = 1

                # Parse the first line of the file to grab bs, gs, and numele
                elif totalLineCount == 1:
                    cols = row.rstrip('\n').split(' ')
                    c = 0
                    bsFound = 0
                    gsFound = 0
                    neFound = 0
                    xFound  = 0
                    for col in cols:
                        c = c + 1

                        if xFound == 1:
                            xFound = 0
                            AddTriple("kernel", "kernel name", col, TripleList)

                        if bsFound == 1:
                            bsFound = 0
                            xyz = col.split(',')
                            bsx = xyz[0]
                            bsy = xyz[1]
                            bsz = xyz[2]
                            AddTriple("blksz_x", "block size x", bsx, TripleList)
                            AddTriple("blksz_y", "block size y", bsy, TripleList)
                            AddTriple("blksz_z", "block size z", bsz, TripleList)

                        if gsFound == 1:
                            gsFound = 0
                            xyz = col.split(',')
                            gsx = xyz[0]
                            gsy = xyz[1]
                            gsz = xyz[2]
                            AddTriple("grdsz_x", "grid size x", gsx, TripleList)
                            AddTriple("grdsz_y", "grid size y", gsy, TripleList)
                            AddTriple("grdsz_z", "grid size z", gsz, TripleList)

                        if neFound == 1:
                            neFound = 0
                            AddTriple("numele", "number of elements", col, TripleList)

                        if col == '-x':
                            xFound = 1
                        if col == '-bs':
                            bsFound = 1
                        if col == '-gs':
                            gsFound = 1
                        if col == '-numele':
                            neFound = 1
                else:
                    cols = row.rstrip('\n').split(',')
                    c = 0
                    for col in cols:
                        c = c + 1
                        if col == '"Device"':
                        #    print(row)
                            headerFound = 1
                        elif col == '"Metric Name"':
                            metricNameFound = c
                        elif col == '"Metric Description"':
                            metricDescriptionFound = c
                        elif col == '"Avg"':
                            metricValueFound = c

    return actuaLineCount,totalLineCount,TripleList,success

######################################################################################################
##
######################################################################################################
def ParseLogPair(fname):
    actuaLineCount, totalLineCount, TripleList, success1 = ParseLog(fname)

    durfname = fname.replace('log', 'dur')
    actualLineCountDur, totalLineCountDur, TripleList, success2 = ParseDuration(durfname, TripleList)

    return actuaLineCount, totalLineCount, TripleList, success1+success2

######################################################################################################
##
######################################################################################################
def PivotLogs(path, outputFile):

    files = glob.glob(path)
    fileCount = 0
    successCount = 0
    HeaderList = []

    print('\nProcessing ...', path)
    
    with open(outputFile, "w") as out:
       for fle in files:
           actual,total,TripleList,success = ParseLogPair(fle)
           
           TripleList[8:].sort(key=lambda triple:triple.name)
           fileCount += 1
           if ((fileCount%500) == 0):
               print(fileCount)

           # Write the column headers if this is the first file.
           if success == 2:
               if fileCount == 1:
                   HeaderList = TripleList
                   for T in TripleList:
                       out.write(T.name+',')
                   out.write('\n')

               # Verify that the HeaderList and the TripleList are the same
               # length and in the same order.
               if len(HeaderList) == len(TripleList):
                   MatchColumns = True
                   LenColumns = True
                   for c1,c2 in zip(HeaderList, TripleList):
                       if c1.name != c2.name:
                           MatchColumns = False
                           c1copy = c1
                           c2copy = c2

                       if len(c2.value.strip()) == 0:
                           LenColumns = False
                           c3copy = c2

                   if MatchColumns:
                       if LenColumns:
                           # Write the values
                           successCount += 1
                           for T in TripleList:
                               out.write(T.value+',')
                           out.write('\n')
                       else:
                           print('**Error %s has a value for column %s with length 0' % (fle,c3copy.name))
                   else:
                       print('**Error %s doesn''t match the column list c1=%s c2=%s' % (fle,c1copy.name,c2copy.name))
               else:
                   print('**Error %s doesn''t match the length of the list (headerlist=%d, triplelist=%d)' % (fle,len(HeaderList),len(TripleList)))
           else:
               print('**Warning: success should be 2, but it is %d' % success)

    if fileCount == 0:
      print('\n*******************\n***** WARNING .. fileCount == 0. Is there something wrong? \n*******************')
      
    print('Finished creating file %s, File count = %d, Success count = %d' % (outputFile, fileCount, successCount))


######################################################################################################
##
######################################################################################################

#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-1M-K1\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-1M-K1.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-1M-K2\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-1M-K2.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-1M-K3\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-1M-K3.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-1M-K4\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-1M-K4.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-1M-K5\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-1M-K5.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-1M-K6\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-1M-K6.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-1M-K7\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-1M-K7.txt')


#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-1M\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-1M.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-10M\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-10M.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-100M\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-100M.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-MM\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-MM.txt')
#PivotLogs('DataForPaper-TitanV\\1LogsTitanV-MM\log*.txt', 'DataForPaper-TitanV\\1pivotlogs-titanv-MM.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-FMM\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-FMM.txt')
#PivotLogs('DataForPaper-TitanV\\1LogsTitanV-FMM\log*.txt', 'DataForPaper-TitanV\\1pivotlogs-titanv-FMM.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-SpMV05\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-SpMV05.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-SpMV25\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-SpMV25.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-SpMV75\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-SpMV75.txt')

#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-Stencil5\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-Stencil5.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-Stencil5SM\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-Stencil5SM.txt')

#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-Stencil9\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-Stencil9.txt')
#PivotLogs('DataForPaper-TitanV\\0LogsTitanV-Stencil9SM\log*.txt', 'DataForPaper-TitanV\\0pivotlogs-titanv-Stencil9SM.txt')







