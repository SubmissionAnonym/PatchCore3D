import os
import shutil
from tqdm import tqdm

from dpipe.io import load


# paths to downloaded and unzipped IXI data
t1_data_path = '/path/to/IXI-T1'
t2_data_path = '/path/to/IXI-T2'

# just a tmp folder that will be deleted
tmp_path = 'tmp'  # '/path/to/tmp'
# output path to preprocessed files
final_path = '/path/to/preproc_ixi'

# path where captk is installed
path_to_captk = '/path/to/captk/CaPTk/1.9.0'

os.makedirs(tmp_path, exist_ok=True)
os.makedirs(final_path, exist_ok=True)

for image_name in tqdm(sorted(os.listdir(t1_data_path))):

    uid = image_name.split('-T1.nii.gz')[0]
    t1_image_file = os.path.join(t1_data_path, image_name)
    t2_image_file = os.path.join(t2_data_path, uid + '-T2.nii.gz')

    if not os.path.exists(t2_image_file):
        print(f'{t2_image_file} does not exist')
        continue

    command = f'{os.path.join(path_to_captk, "captk")} {os.path.join(path_to_captk, "BraTSPipeline.cwl")} -t1 {t1_image_file} -t1c {t1_image_file} -t2 {t2_image_file} -fl {t2_image_file} -o {tmp_path} -b 0 -p {uid} -i 0 -s 0 -d 0'

    code_exit = os.system(command)
    if code_exit != 0:
        print(code_exit)

    t1_tmp_path = os.path.join(tmp_path, f'T1_to_SRI.nii.gz')
    t2_tmp_path = os.path.join(tmp_path, f'T2_to_SRI.nii.gz')
    t1 = load(t1_tmp_path)
    t2 = load(t2_tmp_path)
    print(f'\n\n{uid}: {t1.min()}, {t1.max()}, {t2.min()}, {t2.max()}\n\n')

    os.rename(t1_tmp_path, os.path.join(final_path, f'{uid}_t1_to_SRI.nii.gz'))
    os.rename(t2_tmp_path, os.path.join(final_path, f'{uid}_t2_to_SRI.nii.gz'))

    shutil.rmtree(tmp_path)
