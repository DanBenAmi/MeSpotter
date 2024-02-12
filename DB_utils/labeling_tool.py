import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import os
from PIL import Image, ImageTk
import time


class LabelingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Labeling Tool")
        self.master.geometry("530x900+400+100")

        self.new_df = False
        self.df = None
        self.df_path = ''
        self.image_index = 0
        self.images = []
        self.directory_path = ''

        self.init_ui()


    def init_ui(self):
        instructions = tk.Label(self.master,
                                text="Welcome to Labeling Tool\n\nplease select the images folder.")
        instructions.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5, padx=10)

        load_folder_button = tk.Button(self.master, text="Load Image Folder", command=self.load_images_dir)
        load_folder_button.grid(row=1, column=1, pady=10)
        self.dir_loading_success = tk.Label(self.master, text="")
        self.dir_loading_success.grid(row=2, column=0, columnspan=3)

        load_df_button = tk.Button(self.master, text="Load Existing Labels table", command=self.load_existing_labels)
        or_label = tk.Label(self.master, text="or")
        init_new_df = tk.Button(self.master, text="Initial New Labels Table", command=self.initial_new_df)
        self.df_success = tk.Label(self.master, text="")
        self.df_success.grid(row=5, column=0, columnspan=3)
        load_df_button.grid(row=4, column=0, padx=2, pady=10)
        or_label.grid(row=4, column=1, padx=2, pady=10)
        init_new_df.grid(row=4, column=2, padx=2, pady=10)

        self.img_label = tk.Label(self.master)
        self.img_label.grid(row=7, column=0, columnspan=3, pady=10)

        self.class_label = tk.Label(self.master)
        self.class_label.grid(row=8, column=0, columnspan=3, pady=10)

        save_bt = tk.Button(self.master, text="Save labels table", command=self.save_dataframe)
        save_bt.grid(row=9, column=2, pady=10, padx=10)

        exit_bt = tk.Button(self.master, text="Exit", command=self.exit_app)
        exit_bt.grid(row=9, column=0, pady=10, padx=10)

        self.save_success = tk.Label(self.master, text="")
        self.save_success.grid(row=10, column=2)






    def load_images_dir(self):
        self.directory_path = filedialog.askdirectory()
        if self.directory_path:
            self.images = [img for img in os.listdir(self.directory_path) if img.endswith(('.jpg', '.png'))]
            self.dir_loading_success.config(text="Great. Please load a labels excel file or initial a new labels table.")
        else:
            self.dir_loading_success.config(text="No directory selected. Please select a directory to proceed.")

    def load_existing_labels(self):
        self.df_path = filedialog.askopenfilename(title="Select the Excel file",
                                                  filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
        if self.df_path:
            self.df = pd.read_excel(self.df_path)

            # Filter out images already in the DataFrame
            labeled_images = self.df['Image'].tolist()
            self.images = [img for img in self.images if img not in labeled_images]

            if self.images:
                self.df_success.config(text="Labels table loaded successfully.\n\nYou may start labeling the images:")
                self.start_labeling()
            else:
                self.df_success.config(text="Labels table loaded successfully. all images are already labeled")
                return
        else:
            self.df_success.config(text="Warning: No file selected.")

    def initial_new_df(self):
        self.new_df = True
        self.df_path = os.path.join(self.directory_path, 'labels.xlsx')
        self.df = pd.DataFrame(columns=['Image', 'Label'])
        self.df_success.config(text="Initialized new Data Frame for labels table.\n\nYou may start labeling the images:")
        self.start_labeling()

    def start_labeling(self):
        self.class_label.configure(text="Press the button of the class:\n1=Your face,  2=Other face,  3=Not a face")
        self.update_image()

    def update_image(self):
        if self.image_index < len(self.images):
            image_path = os.path.join(self.directory_path, self.images[self.image_index])
            img = Image.open(image_path).resize((500, 500), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(img)

            self.img_label.configure(image=photo)
            self.img_label.image = photo

            # instructions = tk.Label(self.labeling_window, text="Press '0', '1', or '2' to label the image.")
            # instructions.pack()

            self.master.bind('<Key>', self.handle_keypress)
        else:
            self.class_label.configure(text="Completion: All images have been labeled.")
            self.exit_app()

    def handle_keypress(self, event):
        if event.char in ['1', '2', '3']:
            new_row = pd.DataFrame({'Image': [self.images[self.image_index]], 'Label': int(event.char)-1})
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            self.image_index += 1

            if self.image_index % 10 == 0 or self.image_index == len(self.images):
                self.save_dataframe()

            if self.img_label:
                self.img_label.pack_forget()
            self.update_image()

    def save_dataframe(self):
        if not self.df_path:
            self.df_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        self.df.to_excel(self.df_path, index=False)
        self.save_success.configure(text="Labels table has been saved")
        time.sleep(1)
        self.save_success.configure(text="")



    def exit_app(self):
        self.save_dataframe()
        self.master.destroy()
        self.master.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingApp(root)
    root.mainloop()
