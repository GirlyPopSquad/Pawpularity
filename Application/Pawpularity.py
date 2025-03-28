import tkinter as tk
from tkinter import ttk
import pandas as pd
import os
from PIL import Image, ImageTk
import Regression as reg
import CsvManager as csvManager
import OcclusionProbability as occlusionProbability
import PawpularityNaiveBayes as pawpularityNaiveBayes

pawpularity_model, mse, r2, pawpularity_best_param, pawpularity_best_score = reg.train_pawpularity_model()
human_model, accuracy, loss, human_best_param, human_best_score = reg.train_human_model()
occlusion_model = occlusionProbability.train_occlusion_bayes()
pawpularity_naive_bayes_model, pawpularity_naive_bayes_accuracy = pawpularityNaiveBayes.train_pawpularity_bayes()

data_path = "Application/Data"

class CSVViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Viewer")
        self.root.geometry("1000x700")

        self.setup_dropdown(root)
        self.setup_treeview_w_scrollbar(root)     
        self.setup_overview(root)

    def setup_dropdown(self, root):
        csv_files = csvManager.load_csv_files(data_path)
        options = csv_files

        chosen_file = tk.StringVar()
        dropdown = ttk.OptionMenu(root, chosen_file, 'Choose csv', *options, command=lambda file: self.load_csv(f"{data_path}/{file}"))
        dropdown.pack()

    def setup_treeview_w_scrollbar(self, root):
        self.tree_frame = tk.Frame(root)
        self.tree_frame.pack(expand=False, fill="both")

        self.tree_scroll = tk.Scrollbar(self.tree_frame)
        self.tree_scroll.pack(side="right", fill="y")

        self.tree = ttk.Treeview(self.tree_frame, show="headings", yscrollcommand=self.tree_scroll.set)
        self.tree.pack(expand=False, fill="both", padx=(20 , 0))

        self.tree_scroll.config(command=self.tree.yview)


    def setup_overview(self, root):
        self.overview_frame = tk.Frame(root)
        self.overview_frame.pack(expand=True, fill="both")

        #Canvas Frame
        self.canvas_frame = tk.Frame(self.overview_frame)
        self.canvas_frame.pack(padx=10, pady=10, side=tk.RIGHT)

        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(expand=False)

        self.pawpularityLabel = tk.Label(self.canvas_frame)
        self.pawpularityLabel.pack(expand=False)

        self.humanLabel = tk.Label(self.canvas_frame)
        self.humanLabel.pack(expand=False)

        self.setup_details_frame(self.overview_frame)

    def setup_details_frame(self, parent):
        # Details Frame
        self.details_frame = tk.Frame(parent, bg="lightblue")
        self.details_frame.pack(padx=10, pady=10, side=tk.LEFT, fill="x", expand=True)

        # Metrics Frame
        self.metrics_frame = tk.Frame(self.details_frame)
        self.metrics_frame.pack(padx=10, pady=10, fill="x")

        self.metrics_title_label = tk.Label(self.metrics_frame, font=("Arial", 12, "bold"), text="Metrics")
        self.metrics_title_label.pack()

        self.metrics_pawpularity_frame = tk.Frame(self.metrics_frame)
        self.metrics_pawpularity_frame.pack(padx=10, pady=10, side=tk.LEFT) 

        self.metrics_pawpularity_title = tk.Label(self.metrics_pawpularity_frame, font=("Arial", 10, "bold"),  text="Pawpularity Model")    
        self.metrics_pawpularity_title.pack(anchor="w") 

        self.mse_label = tk.Label(self.metrics_pawpularity_frame, text="MSE: {}".format(mse))
        self.mse_label.pack(anchor="w")

        self.r2_label = tk.Label(self.metrics_pawpularity_frame, text="R2: {}".format(r2))
        self.r2_label.pack(anchor="w")

        self.metrics_human_frame = tk.Frame(self.metrics_frame)
        self.metrics_human_frame.pack(padx=10, pady=10, side=tk.RIGHT)

        self.metrics_human_title = tk.Label(self.metrics_human_frame, font=("Arial", 10, "bold"), text="Human Model")
        self.metrics_human_title.pack(anchor="w")

        self.accuracy_label = tk.Label(self.metrics_human_frame, text="Accuracy: {}".format(accuracy))
        self.accuracy_label.pack(anchor="w")
        
        self.loss_label = tk.Label(self.metrics_human_frame, text="Log Loss: {}".format(loss))
        self.loss_label.pack(anchor="w")

        # Hyperparameters Frame
        self.hyperparams_frame = tk.Frame(self.details_frame)
        self.hyperparams_frame.pack(padx=10, pady=10, fill="x")

        self.hyperparams_title_label = tk.Label(self.hyperparams_frame, font=("Arial", 12, "bold"), text="Hyperparameters")
        self.hyperparams_title_label.pack()

        self.hyperparams_pawpularity_frame = tk.Frame(self.hyperparams_frame)
        self.hyperparams_pawpularity_frame.pack(padx=10, pady=10, side=tk.LEFT)

        self.hyperparams_pawpularity_title = tk.Label(self.hyperparams_pawpularity_frame, font=("Arial", 10, "bold"), text="Pawpularity Model")
        self.hyperparams_pawpularity_title.pack(anchor="w")
        
        # TODO: what is "PBA"
        self.PBA_label = tk.Label(self.hyperparams_pawpularity_frame, text="PBA: {}".format(pawpularity_best_param))
        self.PBA_label.pack(anchor="w")

        self.paw_best_score = tk.Label(self.hyperparams_pawpularity_frame, text="Best Score: {}".format(pawpularity_best_score))
        self.paw_best_score.pack(anchor="w")

        self.hyperparams_human_frame = tk.Frame(self.hyperparams_frame)
        self.hyperparams_human_frame.pack(padx=10, pady=10, side=tk.RIGHT)

        self.hyperparams_human_title = tk.Label(self.hyperparams_human_frame, font=("Arial", 10, "bold"), text="Human Model")
        self.hyperparams_human_title.pack(anchor="w")

        # TODO: what is "HBA"
        self.HBA = tk.Label(self.hyperparams_human_frame, text="HBA: {}".format(human_best_param))
        self.HBA.pack(anchor="w")

        self.human_best_score = tk.Label(self.hyperparams_human_frame, text="Best Score: {}".format(human_best_score))
        self.human_best_score.pack(anchor="w")
        
        #Occulsion Probability
        self.occlusion_probability_frame = tk.Frame(self.details_frame)
        self.occlusion_probability_frame.pack(padx=10, pady=10, fill="x")
        
        self.occlusion_probability_title_label = tk.Label(self.occlusion_probability_frame, font=("Arial", 12, "bold"), text="Occulsion Probability")
        self.occlusion_probability_title_label.pack()
        
        self.occlusion_probability_label = tk.Label(self.occlusion_probability_frame, font=("Arial", 10, "bold"), text="Select image to see probability")
        self.occlusion_probability_label.pack(anchor="w")

        #Pawpularity Probability
        self.pawpularity_probability_frame = tk.Frame(self.details_frame)
        self.pawpularity_probability_frame.pack(padx=10, pady=10, fill="x")
        
        self.pawpularity_probability_title_label = tk.Label(self.pawpularity_probability_frame, font=("Arial", 12, "bold"), text="Pawpularity Naive Bayes")
        self.pawpularity_probability_title_label.pack()
        
        self.pawpularity_probability_label = tk.Label(self.pawpularity_probability_frame, font=("Arial", 10, "bold"), text="Accuracy: {}".format(pawpularity_naive_bayes_accuracy))
        self.pawpularity_probability_label.pack(anchor="w")
        
        
    
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

            self.show_text("Image Loading")

            if os.path.exists(possible_path):
                self.show_image(possible_path)
                self.show_pawpularity_score(image_id, file)
                self.remove_if_human(image_id, file)
                self.show_occlusion_probability(image_id, file)
            else:
                self.show_text("Image not found")

    def show_text(self, text):
        """Displays text in the center of the canvas."""
        self.canvas.delete("all")
        self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, text=text, anchor="center")

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
        
    def show_occlusion_probability(self,image_id, file):
        df = pd.read_csv(file)
    
        image_data = df.loc[df['Id'] == image_id]
        
        image_data = image_data[['Human']]
        
        result = occlusion_model.predict_proba(image_data)
        probability_of_happening = (result[0, 1] * 100).round(3)
        
        self.occlusion_probability_label.configure(text="Occlusion probability: {}".format(probability_of_happening))
        

    def remove_if_human(self,image_id, file):
        
        df = pd.read_csv(file)
    
        image_data = df.loc[df['Id'] == image_id]
    
        coloumns_to_drop = ['Id', 'Human']
        if 'Pawpularity' in image_data.columns:
            coloumns_to_drop.append('Pawpularity')

        image_data = image_data.drop(columns=coloumns_to_drop)
        
        result = human_model.predict(image_data)
        self.humanLabel.configure(text="Human: {}".format(result))

        if result ==[1]:
            self.show_text("Human detected - image will not be shown")
 
        
if __name__ == "__main__":
    root = tk.Tk()
    app = CSVViewerApp(root)
    root.mainloop()
