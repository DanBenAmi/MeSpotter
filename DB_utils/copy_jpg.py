import os
import shutil


def copy_jpg_files(source_dir, dest_dir):
    '''function to copy .jpg files from one directory to another, including subdirectories'''
    # Checks and creates the destination directory if it does not exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walks through the source directory, including all subdirectories
    for root, dirs, files in os.walk(source_dir):
        # Loops through each file in directories
        for file in files:
            # Checks if the file ends with a .jpg extension (case insensitive)
            if file.lower().endswith('.jpg'):
                # Constructs the full path of the source file
                file_path = os.path.join(root, file)
                # Constructs the full path for the destination of the file
                dest_file_path = os.path.join(dest_dir, file)
                # Copies the file from the source to the destination
                shutil.copy(file_path, dest_file_path)
                # Prints a message indicating the file has been copied
                print(f"Copied {file} to {dest_dir}")



if __name__ == "__main__":
    # Define the source and destination directories
    source_directory_path = r'/Users/danbenami/Desktop/תמונות מלחמה 2'
    destination_directory_path = '/DataBase/reserve'

    # Calls the function with the specified source and destination directories
    copy_jpg_files(source_directory_path, destination_directory_path)
