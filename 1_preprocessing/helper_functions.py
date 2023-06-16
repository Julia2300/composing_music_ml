from os import walk

# walk through directory and save files and directories

def get_file_and_dirnames(p):
    """
    Traverse the specified directory and get lists of filenames and directory names.

    :param p: path of the directory to traverse
    :return: a tuple containing a list of filenames and a list of directory names in the specified directory
    """
    f = []
    d = []
    for (dirpath, dirnames, filenames) in walk(p):
        f.extend(filenames)
        d.extend(dirnames)
        break
    return f,d

