import drms
import requests
import os
import time
from datetime import datetime, timedelta
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Download SDO data from DRMS.")
    parser.add_argument("--email", type=str, default='jonnypat25@gmail.com', required=False, help="Email address for DRMS client.")
    parser.add_argument("--series", type=str, default='hmi.M_720s', help="[hmi.Ic_noLimbDark_720s, hmi.M_720s, aia.lev1_euv_12s, aia.lev1_uv_24s]")
    parser.add_argument("--wavelength", type=str, default=None, help="Wavelength for AIA data (e.g., '171', '193', '304')")
    parser.add_argument("--start_date", type=str, default='2016.07.04', help="Start date in 'YYYY.MM.DD' format.")
    parser.add_argument("--end_date", type=str, default='2017.05.14', help="End date in 'YYYY.MM.DD' format.")
    parser.add_argument("--folder_path", type=str, default="/home/gpatane/Dataset/Magnetogram", required=False, help="Folder path to save downloaded files.")
    parser.add_argument("--cadence", type=str, default='1d', help="Download cadence: 1d (daily), 1w (weekly), 1h (hourly), 3d (every 3 days)")
    parser.add_argument("--time_of_day", type=str, default='00:00:00', help="Time of day to download (HH:MM:SS)")
    
    return parser.parse_args()

def parse_cadence(cadence_str):
    """
    Parse cadence string and return timedelta object
    Supported formats: 1d, 2d, 1w, 1h, 30m, etc.
    """
    cadence_mapping = {
        # Days
        '1d': timedelta(days=1),
        '2d': timedelta(days=2),
        '3d': timedelta(days=3),
        '7d': timedelta(days=7),
        
        # Weeks
        '1w': timedelta(weeks=1),
        '2w': timedelta(weeks=2),
        
        # Hours
        '1h': timedelta(hours=1),
        '2h': timedelta(hours=2),
        '6h': timedelta(hours=6),
        '12h': timedelta(hours=12),
        
        # Minutes
        '30m': timedelta(minutes=30),
        '60m': timedelta(hours=1),
    }
    
    if cadence_str in cadence_mapping:
        return cadence_mapping[cadence_str]
    else:
        # Try to parse custom format like "5d", "3h", etc.
        try:
            if cadence_str.endswith('d'):
                days = int(cadence_str[:-1])
                return timedelta(days=days)
            elif cadence_str.endswith('w'):
                weeks = int(cadence_str[:-1])
                return timedelta(weeks=weeks)
            elif cadence_str.endswith('h'):
                hours = int(cadence_str[:-1])
                return timedelta(hours=hours)
            elif cadence_str.endswith('m'):
                minutes = int(cadence_str[:-1])
                return timedelta(minutes=minutes)
            else:
                print(f"Warning: Unknown cadence format '{cadence_str}', using daily")
                return timedelta(days=1)
        except ValueError:
            print(f"Warning: Invalid cadence format '{cadence_str}', using daily")
            return timedelta(days=1)

def download_file(url, file_name, folder_path):
    try:
        if not file_name.endswith(".fits"):
            file_name += ".fits"
        full_path = os.path.join(folder_path, file_name)
        
        # Skip if file already exists
        if os.path.exists(full_path):
            print(f"File {file_name} already exists, skipping...")
            return
            
        response = requests.get(url, timeout=300)
        
        if response.status_code == 200:
            with open(full_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {file_name}")
        else:
            print(f"Error downloading from {url}: Status code {response.status_code}")
    
    except Exception as e:
        print(f"Error during download: {e}")

def main():
    args = parse_args()
    client = drms.Client(email=args.email)
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y.%m.%d")
    end_date = datetime.strptime(args.end_date, "%Y.%m.%d")
    
    # Parse cadence
    cadence = parse_cadence(args.cadence)
    
    # Create folder
    os.makedirs(args.folder_path, exist_ok=True)
    
    print(f"Downloading {args.series} data from {start_date.date()} to {end_date.date()}")
    print(f"Cadence: {args.cadence} ({cadence})")
    print(f"Time of day: {args.time_of_day}")
    
    current_date = start_date
    total_downloads = 0
    
    while current_date <= end_date:
        # Format date with specified time
        date_str = current_date.strftime(f'%Y.%m.%d_{args.time_of_day}')
        
        # Build query for specific time
        if 'aia' in args.series.lower() and args.wavelength:
            # For AIA data with specific wavelength
            query = f"{args.series}[{date_str}][{args.wavelength}]"
        else:
            # For HMI data or AIA without wavelength filter
            query = f"{args.series}[{date_str}]"
        
        print(f"Processing {current_date.date()} at {args.time_of_day}...")
        
        try:
            # Export data
            export_request = client.export(query, protocol='fits')
            
            # Check if data is available
            if hasattr(export_request, 'urls') and len(export_request.urls) > 0:
                # Download the first (closest) image for this time
                url = export_request.urls.url.iloc[0]
                record = export_request.data.record.iloc[0]
                
                print(f"Downloading {record}")
                download_file(url, record, args.folder_path)
                total_downloads += 1
            else:
                print(f"No data available for {current_date.date()} at {args.time_of_day}")
                
        except Exception as e:
            print(f"Error processing {current_date.date()}: {e}")
            
        # Move to next time point based on cadence
        current_date += cadence
        time.sleep(1)  # Small delay to be respectful to the server
    
    print(f"\nDownload completed! Total files downloaded: {total_downloads}")

if __name__ == "__main__":
    main()