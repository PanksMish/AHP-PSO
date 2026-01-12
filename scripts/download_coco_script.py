"""
Download MS COCO 2017 dataset
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for download"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    print(f"Downloading {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True,
                              miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path, extract_to):
    """Extract zip file with progress"""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.namelist(), desc="Extracting"):
            zip_ref.extract(member, extract_to)


def main():
    parser = argparse.ArgumentParser(description='Download COCO 2017 dataset')
    parser.add_argument('--output', type=str, default='data/coco',
                        help='Output directory')
    parser.add_argument('--train', action='store_true',
                        help='Download training set')
    parser.add_argument('--val', action='store_true', default=True,
                        help='Download validation set')
    parser.add_argument('--annotations', action='store_true', default=True,
                        help='Download annotations')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = 'http://images.cocodataset.org/zips/'
    anno_url = 'http://images.cocodataset.org/annotations/'
    
    files_to_download = []
    
    if args.train:
        files_to_download.append(('train2017.zip', base_url))
    
    if args.val:
        files_to_download.append(('val2017.zip', base_url))
    
    if args.annotations:
        files_to_download.append(('annotations_trainval2017.zip', anno_url))
    
    # Download files
    for filename, url in files_to_download:
        file_url = url + filename
        output_path = output_dir / filename
        
        if output_path.exists():
            print(f"{filename} already exists. Skipping download.")
        else:
            download_url(file_url, output_path)
        
        # Extract
        extract_to = output_dir
        print(f"Extracting {filename}...")
        extract_zip(output_path, extract_to)
        
        # Remove zip file to save space
        print(f"Removing {filename}...")
        output_path.unlink()
    
    print("\n" + "="*50)
    print("COCO 2017 dataset downloaded successfully!")
    print(f"Location: {output_dir.absolute()}")
    print("="*50)


if __name__ == '__main__':
    main()
