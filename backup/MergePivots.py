import csv, sys
import glob


infiles = [

         "DataForPaper-TitanV\\0pivotlogs-titanv-1M.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-10M.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-100M.txt",
         
         "DataForPaper-TitanV\\0pivotlogs-titanv-FMM.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-MM.txt",
         "DataForPaper-TitanV\\1pivotlogs-titanv-FMM.txt",
         "DataForPaper-TitanV\\1pivotlogs-titanv-MM.txt",

#         "DataForPaper-TitanV\\0pivotlogs-titanv-1M-K1.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-1M-K2.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-1M-K3.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-1M-K4.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-1M-K5.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-1M-K6.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-1M-K7.txt",
#         
#         "DataForPaper-TitanV\\0pivotlogs-titanv-10M-K1.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-10M-K2.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-10M-K3.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-10M-K4.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-10M-K5.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-10M-K6.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-10M-K7.txt",

         "DataForPaper-TitanV\\0pivotlogs-titanv-Stencil5.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-Stencil5SM.txt",         
#         "DataForPaper-TitanV\\0pivotlogs-titanv-Stencil9.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-Stencil9SM.txt",         
         ]
   
outfile = "DataForPaper-TitanV\\pivotlogs-titanv-TRAIN.txt"

xinfiles = [
#         "DataForPaper-TitanV\\0pivotlogs-titanv-Stencil5.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-Stencil9.txt",
         "DataForPaper-TitanV\\0pivotlogs-titanv-SpMV05.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-SpMV25.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-SpMV75.txt",
#         "DataForPaper-TitanV\\0pivotlogs-titanv-SpMV95.txt",
         
#         "DataForPaper-TitanV\\0pivotlogs-titanv-10M.txt",
#         "DataForPaper-TitanV\\1pivotlogs-titanv-FMM.txt"
#         "DataForPaper-TitanV\\1pivotlogs-titanv-FMA.txt"
         ]
   
xoutfile = "DataForPaper-TitanV\\pivotlogs-titanv-TEST.txt"

def domerge(infiles, outfile):
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
               
            print("File name {} wrote {} lines to {}".format(f, lc, outfile))
      
   except (FileNotFoundError) as e:
     sys.exit('\n***\n*** File Not Found Error in DoMerge {}***'.format(e.filename))
     
   fhout.close()

   
domerge(infiles, outfile)
domerge(xinfiles, xoutfile)
