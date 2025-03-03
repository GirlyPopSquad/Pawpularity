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
        
        self.image_label = tk.Label(root, text="No image selected", bg="gray", width=50, height=20)
        self.image_label.pack(pady=10)
        
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
            image_path = "Application/Data/train/" + image_id + ".jpg"

            if image_path:
                self.show_image(image_path)
            else:
                self.image_label.config(text="Image not found", image="")
      
    def show_image(self, image_path):
        """Displays the selected image."""
        image = Image.open(image_path)

        # Resize image to a larger size (e.g., 400x400)
        image = image.resize((400, 400), Image.Resampling.LANCZOS)  
        photo = ImageTk.PhotoImage(image)

        self.image_label.config(image=photo, text="") 
        self.image_label.image = photo  # Keep reference to avoid garbage collection

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVViewerApp(root)
    root.mainloop()
