import tkinter as tk
from tkinter import ttk
import pandas as pd
import os
from PIL import Image, ImageTk
import Regression as reg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pawpularity_model, mse, r2, pawpularity_best_param, pawpularity_best_score = reg.train_pawpularity_model()
human_model, accuracy, loss, human_best_param, human_best_score = reg.train_human_model()

class CSVViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Viewer")
        self.root.geometry("800x600")

        csv_files = self.load_csv_files()
        options = csv_files

        # Dropdown
        chosen_file = tk.StringVar()
        dropdown = ttk.OptionMenu(root, chosen_file, 'Choose csv', *options, command=lambda file: self.load_csv(f"Application/Data/{file}"))
        dropdown.pack()

        # Treeview (Table) with Scrollbar
        self.tree_frame = tk.Frame(root)
        self.tree_frame.pack(expand=False, fill="both")

        self.tree_scroll = tk.Scrollbar(self.tree_frame)
        self.tree_scroll.pack(side="right", fill="y")

        self.tree = ttk.Treeview(self.tree_frame, show="headings", yscrollcommand=self.tree_scroll.set)
        self.tree.pack(expand=False, fill="both", padx=(20 , 0)) #Added paddign to make it look symmetrical against the scrollbar

        self.tree_scroll.config(command=self.tree.yview)
        
        self.setup_details(root)

    def setup_details(self, root):
        self.details_frame = tk.Frame(root)
        self.details_frame.pack(expand=False, fill="both")

        #Canvas Frame
        self.canvas_frame = tk.Frame(self.details_frame)
        self.canvas_frame.pack(padx=20, pady=10, side=tk.RIGHT)

        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(expand=False)

        self.pawpularityLabel = tk.Label(self.canvas_frame)
        self.pawpularityLabel.pack(expand=False)

        self.humanLabel = tk.Label(self.canvas_frame)
        self.humanLabel.pack(expand=False)

        #Metrics Frame
        self.metrics_frame = tk.Frame(self.details_frame)
        self.metrics_frame.pack(padx=20, pady=10, side=tk.LEFT)

        self.metricsLabel1 = tk.Label(self.metrics_frame)
        self.metricsLabel1.pack(anchor="w", side="bottom", expand=False)
        self.metricsLabel2 = tk.Label(self.metrics_frame)
        self.metricsLabel2.pack(anchor="w", side="bottom", expand=False)
        self.metricsLabel3 = tk.Label(self.metrics_frame)
        self.metricsLabel3.pack(anchor="w", side="bottom", expand=False)
        self.metricsLabel4 = tk.Label(self.metrics_frame)
        self.metricsLabel4.pack(anchor="w", side="bottom", expand=False)
        
        self.hyperparamterlabel1 = tk.Label(self.metrics_frame)
        self.hyperparamterlabel1.pack(anchor="w", side="bottom")
        self.hyperparamterlabel2 = tk.Label(self.metrics_frame)
        self.hyperparamterlabel2.pack(anchor="w", side="bottom")
        self.hyperparamterlabel3 = tk.Label(self.metrics_frame)
        self.hyperparamterlabel3.pack(anchor="w", side="bottom")
        self.hyperparamterlabel4 = tk.Label(self.metrics_frame)
        self.hyperparamterlabel4.pack(anchor="w", side="bottom")
    
    def load_csv_files(self):
        path = "Application/Data"
        dir_list = os.listdir(path)
        csv_list =  filter(lambda x: x.endswith('.csv'), dir_list)
        return list(csv_list)
    
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
        self.metricsLabel1.configure(text="MSE: {}".format(mse))
        self.metricsLabel2.configure(text="R2: {}".format(r2))
        self.metricsLabel3.configure(text="Accuracy: {}".format(accuracy))
        self.metricsLabel4.configure(text="Log Loss: {}".format(loss))
        
        self.hyperparamterlabel1.configure(text="PBA: {}".format(pawpularity_best_param))
        self.hyperparamterlabel2.configure(text="PBS: {}".format(pawpularity_best_score))
        self.hyperparamterlabel3.configure(text="HBA: {}".format(human_best_param))
        self.hyperparamterlabel4.configure(text="HBS: {}".format(human_best_score))
        
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

        if result ==[1]:
            self.show_text("Human detected on image, removed image", self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2)
 
        
if __name__ == "__main__":
    root = tk.Tk()
    app = CSVViewerApp(root)
    root.mainloop()
