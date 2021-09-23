import os
import sys
import glob
from download import calc_sha1

file_directory = sys.argv[1]
file_directory = os.path.abspath(file_directory)
paths_to_hash = []

file_list = []
for filename in glob.iglob(file_directory + '**/**', recursive=True):
    if os.path.isfile(filename):
        file_list.append((calc_sha1(filename).hexdigest(), filename))

print(file_list)
