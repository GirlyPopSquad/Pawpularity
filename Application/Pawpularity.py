import tkinter as tk
from tkinter import ttk
import pandas as pd
import os
from PIL import Image, ImageTk


class CSVViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Viewer")
        self.root.geometry("700x500")

        # Buttons to load specific CSV files
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        self.file1_button = ttk.Button(self.button_frame, text="Load test.csv", command=lambda: self.load_csv("Application/Data/test.csv"))
        self.file1_button.grid(row=0, column=0, padx=5)

        self.file2_button = ttk.Button(self.button_frame, text="Load train.csv", command=lambda: self.load_csv("Application/Data/train.csv"))
        self.file2_button.grid(row=0, column=1, padx=5)

        # Treeview (Table)
        self.tree = ttk.Treeview(root, show="headings")
        self.tree.pack(expand=False, fill="both")

        # Scrollbars
        self.vsb = ttk.Scrollbar(root, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(root, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        
        self.canvas = tk.Canvas(root)
        self.canvas.pack(pady=10)
        
        self.tree.bind("<<TreeviewSelect>>", self.on_row_selected)
    
    def load_csv(self, file_name):
        try:
            # Read CSV
            df = pd.read_csv(file_name)

            # Clear existing data
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = list(df.columns)

            # Configure columns
            for col in df.columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100)

            # Insert data
            for _, row in df.iterrows():
                self.tree.insert("", "end", values=list(row))

        except Exception as e:
            print(f"Error loading file: {e}")
            
            
    def on_row_selected(self, event):
        """Triggered when a row is selected in the table."""
        selected_item = self.tree.selection()
        if selected_item:
            row_values = self.tree.item(selected_item[0])["values"]
            
            # Assuming 'id' is the first column
            image_id = str(row_values[0])  
            
            possible_folders = ["Application/Data/train/", "Application/Data/test/"]

            image_path = None
            for folder in possible_folders:
                possible_path = folder + image_id + ".jpg"
                if os.path.exists(possible_path):
                    image_path = possible_path
                    break 

            if os.path.exists(image_path):
                self.show_image(image_path)
            else:
                self.image_label.config(text="Image not found", image="")
      
    def show_image(self, image_path):
        """Displays the selected image."""
        image = Image.open(image_path)

        # Get image dimensions
        img_width, img_height = image.size

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate the aspect ratio of the image
        img_aspect_ratio = img_width / img_height
        canvas_aspect_ratio = canvas_width / canvas_height

        if img_aspect_ratio > canvas_aspect_ratio:
            # Image is wider than canvas
            new_width = canvas_width
            new_height = int(new_width / img_aspect_ratio)
        else:
            # Image is taller than canvas
            new_height = canvas_height
            new_width = int(new_height * img_aspect_ratio)

        # Resize image to fit the canvas dimensions while maintaining aspect ratio
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(image)  # Keep a reference to avoid garbage collection
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor="center", image=self.photo)
        

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVViewerApp(root)
    root.mainloop()
