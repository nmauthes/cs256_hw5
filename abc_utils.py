'''

Contains utilities for processing .abc files

'''

import glob
import os
import re


HEADER = '`' * 16 + 'BEGINABC' + '`' * 16


def generate_data_file(abc_folder, out_file, out_folder='', add_headers=False):
    '''

    Concatenates all available .abc files in the specified folder and
    saves them to in a .txt file

    :param abc_folder: The folder containing the .abc files
    :param out_file: The name of the output file, e.g. 'data.txt'
    :param out_folder: The name of the ouput folder, default is cwd
    :param add_headers: If true, this will add a header indicating the beginning of ABC text. For use with training.
    '''

    abc_files = glob.glob(os.path.join(abc_folder, '*.abc'))

    if not abc_files:
        raise Exception('NO ABC FILES FOUND')

    lines = []

    for abc in abc_files:
        with open(abc, 'r') as f_in:
            for line in f_in:
                lines.append(line)

                if add_headers and re.match('X: [0-9]+', line):
                    lines.insert(len(lines) - 2, HEADER)

    with open(os.path.join(out_folder, out_file), 'w') as f_out:
        for line in lines:
            f_out.write(line)

    print(f'Found {len(abc_files)} .abc files')
    print(f'Saved as {os.path.join(out_folder, out_file)}')
