def download(data_dict: dict = {
        "test.csv":"1GMMnuJlDkoBRrPq31AAbSmmyoQf5-qrm",
        "train.csv":"14g-ZnZ2Csrw6z-rP4rnIiswwkgoAT-OJ",
        "val.csv":"1NDtZleuWq1RXonU75GIyxvvN4zgf-M6V",
    }) -> None:
    '''
    Downloads the data from the given dictionary of file names and file ids from google drive
    input:
        data_dict: dict - dictionary of file names and file ids
    example usage: 
        data_dict = {
            "test.csv":"1GMMnuJlDkoBRrPq31AAbSmmyoQf5-qrm",
            "train.csv":"14g-ZnZ2Csrw6z-rP4rnIiswwkgoAT-OJ",
            "val.csv":"1NDtZleuWq1RXonU75GIyxvvN4zgf-M6V",
        }
        download(data_dict)
    '''
    import gdown
    import os
    print("Data Dict", data_dict)
    for file_name, file_id in data_dict.items():
        if not os.path.exists(file_name):
            url = f'https://drive.google.com/uc?id={file_id}'
            print(f"Downloading {file_name} from {url}")
            gdown.download(url, file_name, quiet=False)