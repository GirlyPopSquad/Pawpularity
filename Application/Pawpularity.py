import tkinter as tk
from tkinter import ttk
import pandas as pd
import os
from PIL import Image, ImageTk
import Regression as reg

pawpularity_model = reg.train_pawpularity_model()
human_model = reg.train_human_model()

class CSVViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Viewer")
        self.root.geometry("800x600")
        

        # Buttons to load specific CSV files
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        # Button for each CSV file in the Data folder
        csv_files = self.load_csv_files()
        for file in csv_files:
            button = ttk.Button(self.button_frame, text=f"Load {file}", command=lambda f=file: self.load_csv(f"Application/Data/{f}"))
            button.pack(side="left", padx=5)

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
        
        self.pawpularityLabel = tk.Label(root)
        self.pawpularityLabel.pack(expand=True)
        
        self.humanLabel = tk.Label(root)
        self.humanLabel.pack(expand=True)
    
    def load_csv_files(self):
        path = "Application/Data"
        dir_list = os.listdir(path)
        csv_list =  filter(lambda x: x.endswith('.csv'), dir_list)
        return csv_list
    
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

            self.tree.bind("<<TreeviewSelect>>", lambda event: self.on_row_selected(event, file_name))

        except Exception as e:
            print(f"Error loading file: {e}")
            
    def on_row_selected(self,event, file):
        """Triggered when a row is selected in the table."""
        selected_item = self.tree.selection()
        if selected_item:
            row_values = self.tree.item(selected_item[0])["values"]
            
            # Assuming 'id' is the first column
            image_id = str(row_values[0])  

            # Check if folder exists with the name of the file
            filename = file.replace(".csv", "")
            folder = f"{filename}"
        
            possible_path = folder + "/" + image_id + ".jpg"

            self.show_text("Image Loading", self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2)

            if os.path.exists(possible_path):
                self.show_image(possible_path)
                self.show_pawpularity_score(image_id, file)
                self.remove_if_human(image_id, file)
            else:
                self.show_text("Image not found", self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2)

    def show_text(self, text, x, y):
        """Displays text on the canvas at the specified position."""
        self.canvas.delete("all")
        self.canvas.create_text(x, y, text=text, anchor="center")

    def show_image(self, image_path):
        """Displays the selected image."""
        self.canvas.delete("all")
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
        image = image.resize((new_width, new_height))

        self.photo = ImageTk.PhotoImage(image)  # Keep a reference to avoid garbage collection
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor="center", image=self.photo)
        
    def show_pawpularity_score(self,image_id, file):
        
        df = pd.read_csv(file)
    
        image_data = df.loc[df['Id'] == image_id]
        
        if "test.csv" in file:
            image_data = image_data.drop(columns=['Id'])
        else:
            image_data = image_data.drop(columns=['Id', 'Pawpularity'])
        
        result = pawpularity_model.predict(image_data)
        
        self.pawpularityLabel.configure(text="Pawpularity score: {}".format(result))   

    def remove_if_human(self,image_id, file):
        
        df = pd.read_csv(file)
    
        image_data = df.loc[df['Id'] == image_id]
        
        if "test.csv" in file:
            image_data = image_data.drop(columns=['Id', 'Human'])
        else:
            image_data = image_data.drop(columns=['Id', 'Human', 'Pawpularity'])
        
        result = human_model.predict(image_data) 
        self.humanLabel.configure(text="Human score: {}".format(result))  
        
if __name__ == "__main__":
    root = tk.Tk()
    app = CSVViewerApp(root)
    root.mainloop()
