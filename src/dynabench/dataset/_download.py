import requests
import shutil
import glob
import os
import hashlib
from tqdm.auto import tqdm


DATASETS_WUEDATA = {
    'advection-cloud-high': 'zbqcxuyReznAgcph',
    'advection-cloud-low': 'GaRdhfPCWxHUzxlQ',
    'advection-cloud-medium': 'owRAjEeBzNIGqvdK',
    'advection-grid-full': 'yykBSNdbcqItyWDh',
    'advection-grid-high': 'nwJlWdrofVkYuCts',
    'advection-grid-low': 'FGItmswlpuHempEw',
    'advection-grid-medium': 'zmciLlnTwTlPVZCz',
    'burgers-cloud-high': 'aGrDKZabZGEXKZYO',
    'burgers-cloud-low': 'wpOlFjtZEGmwiavw',
    'burgers-cloud-medium': 'rDBEKoNFokVeAeRc',
    'burgers-grid-full': 'PDOojyfucyMRRLXu',
    'burgers-grid-high': 'oWQFcnHDPFVTPuFF',
    'burgers-grid-low': 'llIjxBIGcADVYxqP',
    'burgers-grid-medium': 'uINfiLobMAwkqczG',
    'gasdynamics-cloud-high': 'pxXsRMxabNQcHuLc',
    'gasdynamics-cloud-low': 'DZkQFcCKGEKwWQvQ',
    'gasdynamics-cloud-medium': 'oXcHGHypRtrfsFLG',
    'gasdynamics-grid-full': 'PmXYFqMThqzQUHDr',
    'gasdynamics-grid-high': 'mCTBjrtqJUcavlXF',
    'gasdynamics-grid-low': 'zKFoiADdlABeAfkG',
    'gasdynamics-grid-medium': 'ZqNhoHNmhQTPdEwR',
    'kuramotosivashinsky-cloud-high': 'qCgdghMRDesLMCVA',
    'kuramotosivashinsky-cloud-low': 'ApqXiYzrJeKUlrwV',
    'kuramotosivashinsky-cloud-medium': 'phAeHeEJqUgtHGYS',
    'kuramotosivashinsky-grid-full': 'MZwiHmAEZMpDbNTt',
    'kuramotosivashinsky-grid-high': 'ELRRJbfPWZDzZJeS',
    'kuramotosivashinsky-grid-low': 'OAqFMPiCplVdZfPI',
    'kuramotosivashinsky-grid-medium': 'CuyNqLlxtHxXbRss',
    'reactiondiffusion-cloud-high': 'QWuksjgCrrAtLvUf',
    'reactiondiffusion-cloud-low': 'wnJBGyCRApQVYNej',
    'reactiondiffusion-cloud-medium': 'riXsWXMdJAVrEiiw',
    'reactiondiffusion-grid-full': 'eBpfBneUjyylNipB',
    'reactiondiffusion-grid-high': 'IyISptaSPhILTNKG',
    'reactiondiffusion-grid-low': 'IfnaSfmyjxkwHDJs',
    'reactiondiffusion-grid-medium': 'mtBwUKUzUhiQrLLr',
    'wave-cloud-high': 'FNrHjaYEfEXzqvDI',
    'wave-cloud-low': 'YQwODtvCBYffajww',
    'wave-cloud-medium': 'tNMywrnmQlUatFra',
    'wave-grid-full': 'OVNOzAPatjIHbmmD',
    'wave-grid-high': 'eVzNMGmGnuFYMSRM',
    'wave-grid-low': 'IaBtcJtpAriSciCd',
    'wave-grid-medium': 'TnRqCkHNoseAJemZ'
}

BASE_URL = "https://wuedata.uni-wuerzburg.de/radar/api/datasets/%s/download"




def download_raw(equation: str, structure: str, resolution: str, tmp_dir: str = "data/tmp/"):
    """
        Download the raw tar-file from the WUEData API.

        Parameters
        ----------
        equation : str
            Name of the equation to download.
        structure : str
            Description of how the observation points are structured. Can be "cloud" or "grid".
        resolution : str
            Resolution of the dataset. Can be "low", "medium", or "high".
        tmp_dir : str
            Directory where the temporary files should be saved. Defaults to "data/tmp/".
        
            
        Returns
        -------
        None
    """


    # make an HTTP request within a context manager
    url = BASE_URL % DATASETS_WUEDATA[f"{equation}-{structure}-{resolution}"]
    with requests.get(url, stream=True) as r:
        
        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))
        
        # implement progress bar via tqdm
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
        
            # save the output to a file
            out_path = os.path.join(tmp_dir, f"{equation}-{structure}-{resolution}.tar")
            with open(out_path, 'wb') as output:
                shutil.copyfileobj(raw, output)
            

            #shutil.rmtree("data/tmp")

def download_equation(equation: str, structure: str, resolution: str, out_dir: str = "data", tmp_dir: str = "data/tmp/"):
    """
        Download a dataset and unpack it to the right place.

        Parameters
        ----------
        equation : str  
            Name of the equation to download.
        structure : str
            Description of how the observation points are structured. Can be "cloud" or "grid".
        resolution : str
            Resolution of the dataset. Can be "low", "medium", or "high".
        out_dir : str
            Directory where the dataset should be saved. Defaults to "data/".
        tmp_dir : str
            Directory where the temporary files should be saved. Defaults to "data/tmp/". This directory will be deleted after the dataset is unpacked.
        
        Returns
        -------
        None
    """


    # paths
    tmp_dir = os.path.join(out_dir, tmp_dir)

    os.makedirs(tmp_dir, exist_ok=True)

    # download the tar file
    print("Downloading data...")
    download_raw(equation, structure, resolution, tmp_dir)

    # unpack the tar file
    print("Unpacking data...")
    tar_path = os.path.join(tmp_dir, f"{equation}-{structure}-{resolution}.tar")
    tmp_target_path = os.path.join(tmp_dir, f"{equation}-{structure}-{resolution}")
    shutil.unpack_archive(tar_path, tmp_target_path)

    # check md5 sums
    print("Checking md5 sums...")
    manifest_template = os.path.join(tmp_target_path, "*", "manifest-md5.txt")
    manifest_path = glob.glob(manifest_template)[0]
    with open(manifest_path, 'r') as f:
        manifest = f.read()
    
    file_template = os.path.join(tmp_target_path, "*", "data", "dataset", "*.h5")
    for file in glob.glob(file_template):
        with open(file, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash not in manifest:
            print(f"MD5 sum of {file} does not match the manifest.")
    
    # move the data to the right place
    print("Moving data...")
    target_path = os.path.join(out_dir, equation, structure, resolution)
    os.makedirs(target_path, exist_ok=True)
    for file in glob.glob(file_template):
        try:
            shutil.move(file, target_path)
        except shutil.Error:
            print(f"File {file} cannot be moved to {target_path}. Skipping.")

    # clean up
    print("Cleaning up...")
    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    download_equation("wave", "cloud", "low")