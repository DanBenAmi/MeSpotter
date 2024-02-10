import os


def delete_non_photo_files(directory):
    # Define a set of valid photo file extensions
    valid_extensions = {'.jpeg', '.jpg', '.png', '.gif', '.tiff', '.bmp', '.webp'}

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # List all files in the directory
    files = os.listdir(directory)

    # Loop through files and delete those not in valid_extensions
    for file in files:
        # Get the file extension
        ext = os.path.splitext(file)[1].lower()
        # Check if the file is not a photo
        if ext not in valid_extensions:
            # Construct full file path
            file_path = os.path.join(directory, file)
            # Check if it's a file and not a directory
            if os.path.isfile(file_path):
                print(f"Deleting {file_path}...")
                os.remove(file_path)  # Delete the file
            else:
                print(f"Skipped {file_path}, it is not a file.")

    print("Cleanup complete.")

if __name__ == '__main__':
    # Example usage
    directory_path = '../DataBase/ImageDB/phone'  # Change this to your directory
    delete_non_photo_files(directory_path)
