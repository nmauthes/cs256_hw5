'''

Contains utilities for processing .abc files

'''

import glob
import os

def generate_data_file(abc_folder, out_file, out_folder=''):
    '''

    Concatenates all available .abc files in the specified folder and
    saves them to in a .txt file

    :param abc_folder: The folder containing the .abc files
    :param out_file: The name of the output file, e.g. 'data.txt'
    :param out_folder: The name of the ouput folder, default is cwd
    '''

    abc_files = glob.glob(os.path.join(abc_folder, '*.abc'))

    if not abc_files:
        raise Exception('NO ABC FILES FOUND')

    with open(os.path.join(out_folder, out_file), 'w') as f_out:
        for abc in abc_files:
            with open(abc, 'r') as f_in:
                f_out.write(f_in.read())

    print(f'Found {len(abc_files)} .abc files')
    print(f'Saved as {os.path.join(out_folder, out_file)}')
