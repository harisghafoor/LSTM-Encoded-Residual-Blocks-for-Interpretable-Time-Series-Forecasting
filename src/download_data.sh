# Run the following commands for downloading the data
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1QuoP44tF5GFDkxXt7RCAZvOKWe4esO_D', dest_path=f"./{DATASETS_PATH}/Electricity-train.csv", unzip=False)
gdd.download_file_from_google_drive(file_id='1qD6w_s4SnNYYds67eAEq3j9PZs67bw-m', dest_path=f"./{DATASETS_PATH}/Electricity-test.csv", unzip=False)
gdd.download_file_from_google_drive(file_id='12WUb2S-mDpD8mgbPJrP0npTmGcoySRnx', dest_path=f"./{DATASETS_PATH}/MAPE.csv", unzip=False)