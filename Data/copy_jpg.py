import os
import shutil

def copy_jpg_files(source_dir, dest_dir):
    '''function that will take all .jpg files from a specified directory (including its subdirectories) and copy them to
     another specified directory. '''
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Construct the destination file path
                dest_file_path = os.path.join(dest_dir, file)
                # Copy the file to the destination directory
                shutil.copy(file_path, dest_file_path)
                print(f"Copied {file} to {dest_dir}")

if __name__ == "__main__":
    source_directory_path = r'/Users/danbenami/Desktop/תמונות מלחמה 2'
    destination_directory_path = '/Users/danbenami/Desktop/MeSpotter/Data/reserve'

    copy_jpg_files(source_directory_path, destination_directory_path)
