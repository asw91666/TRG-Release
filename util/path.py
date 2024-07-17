import os

def get_file_dir_name(dir_path, target_name=None):
    """
    return
    file_dir_names: all file names and directory names in dir_path, sorted
    file_dir_path: all file path and directory path in dir_path, sorted
    target_list: target file names and directory names in dir_path, sorted
    target_path: target file path and directory path in dir_path, sorted
    """
    file_dir_names = os.listdir(dir_path)
    file_dir_names.sort()

    file_dir_path = []
    target_list = []
    target_path = []
    for name in file_dir_names:
        # return path
        tmp_path = os.path.join(dir_path, name)
        file_dir_path.append(tmp_path)

        if target_name is not None and target_name in name:
            # return name of file
            target_list.append(name)

            # return path
            tmp_path = os.path.join(dir_path, name)
            target_path.append(tmp_path)

    return file_dir_names, file_dir_path, target_list, target_path