import os
import subprocess
from pathlib import Path
import file_manager
from tqdm import tqdm
import zipfile

if __name__ == '__main__':
    """
    The main program of the project.
    The flow is as follows:
    -> Lists with name initialization 
    -> Calling GDrive API 
    -> Retrieval of file system 
    -> Query to our needs 
    -> Download ZIP files of ALL models (including not supported baseline models)
    -> UnZipping all models ZIPs
    -> Calling streamlit using subprocess - output to main's console
    """
    gpt_names = ['gpt_neo_sampled30m_no_hashtags',
                 'gpt_neo_sampled10m_no_hashtags',
                 'gpt_neo_sampled1m',
                 'gpt_neo_sample2m']

    baselines_names = ['baseline1',
                       'baseline2']

    prefix = 'model/'
    suffix = '.zip'

    # load
    drive = file_manager.DriveAPI()
    gpts = [i for i in drive.items if 'gpt' in i['name'] and '.zip' in i['name']]
    baselines = [i for i in drive.items if 'baseline' in i['name'] and '.zip' in i['name']]
    print(*gpts, sep="\n", end="\n\n")
    print(*baselines, sep="\n", end="\n\n")

    print('Downloading all models, the procedure can take some time', end='\n')
    # download
    print('Downloading GPT-NEO models from project drive', end='\n')
    for g in tqdm(gpts):
        if not Path(f"{prefix}{g['name']}").exists():
            drive.FileDownload(g['id'], g['name'], 'model')

    print('Downloading GPT-NEO models from project drive')
    for b in tqdm(baselines):
        if not Path(f"{prefix}{b['name']}").exists():
            drive.FileDownload(b['id'], b['name'], 'model')

    # unzip
    print('\n\n Unzipping GPT', end='\n\n')
    for g in gpt_names:
        dir_path = f'{prefix}{g}'.replace('.zip', 'gpt_neo_sampled30m_no_hashtags')
        if not Path(dir_path).exists():
            os.mkdir(f"{dir_path}")
            with zipfile.ZipFile(f'{prefix}{g}{suffix}', 'r') as zip_ref:
                zip_ref.extractall(f"{dir_path}")
        print(f'\n Unzipped {g}', end='\n')

    print('\n\n Unzipping BaseLine', end='\n')
    for b in baselines_names:
        dir_path = f'{prefix}{b}'.replace('.zip', '')
        if not Path(f'{prefix}{b}').exists():
            os.mkdir(f"{dir_path}")
            with zipfile.ZipFile(f'{prefix}{b}{suffix}', 'r') as zip_ref:
                zip_ref.extractall(f"{dir_path}")
            print(f'\n Unzipped {b}', end='\n')

    subprocess.run(['streamlit', 'run', 'main_streamlit_page.py'])


