from os import walk

# walk through directory and save files and directories

def get_file_and_dirnames(p):
    """
    Get filenames and directory names in a given directory.

    :param p: path of directory
    :return: list of filenames, list of directory names
    """
    f = []
    d = []
    for (dirpath, dirnames, filenames) in walk(p):
        f.extend(filenames)
        d.extend(dirnames)
        break
    return f,d

