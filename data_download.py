import functools
import json
import multiprocessing as mp
import requests
import subprocess
import tqdm

from bs4 import BeautifulSoup as bs
from pathlib import Path as path

WORKERS = 4
BATCHSIZE = 30



def main() -> None:
    with open('data_source.json', 'r') as fp:
        datasets = json.load(fp)
    
    for _name, _dataset in datasets.items():
        print(f'Working on {_name}...')
        nc_files = walk_nc(_dataset['source'])
        nc_batches = [nc_files[30 * i:30 * (i + 1)] for i in range(len(nc_files) // 30)] + [nc_files[30 * (len(nc_files) // 30):]]

        if not ('data' / path(_name)).exists(): ('data' / path(_name)).mkdir(parents=True)

        with mp.Pool(WORKERS) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(
                    functools.partial(
                        download,
                        command=f'wget --no-parent --recursive --no-host-directories --cut-dirs {_dataset["cutdirs"]} --timestamping --execute robots=off',
                        destination_root='data' / path(_name)
                    ),
                    nc_batches
                )
            ):
                pass


def download(url_list, command, destination_root) -> None:
    """ Download all files in url_list """
    proc = subprocess.Popen(command.split() + url_list, cwd=destination_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.communicate()


def walk_nc(url) -> list:
    """ Find all .nc files in all subdirectories of url and return as list
    
    This was developed for https://noaadata.apps.nsidc.org/NOAA/.
    It probably won't work in any other context.
    """
    nc_files = []

    r = requests.get(url)
    for link in bs(r.content, 'html.parser').find_all('a'):
        href = link.get('href')
        if href and href.endswith('.nc'):  # nc file
            nc_files.append(url + href)
        elif href and href != '../' and href.endswith('/'):  # subdirectory
            nc_files.extend(walk_nc(url + href))

    return nc_files


if __name__ == '__main__':
    main()
