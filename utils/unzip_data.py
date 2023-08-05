import shutil

filename = "../data/raw/data.csv.zip"
extract_dir = "../data/prepared"
shutil.unpack_archive(filename, extract_dir)
