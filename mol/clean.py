import os
import glob

file_list = glob.glob("./R*")
print(file_list)

for ff in file_list:
    os.system("rm -rf {}".format(ff))
