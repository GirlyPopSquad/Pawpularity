import tkinter as tk
from tkinter import ttk
import pandas as pd
import os

class CSVViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Viewer")
        self.root.geometry("700x500")

        # Buttons to load specific CSV files
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        self.file1_button = ttk.Button(self.button_frame, text="Load test.csv", command=lambda: self.load_csv(os.path.join("Data", "test.csv")))
        self.file1_button.grid(row=0, column=0, padx=5)

        self.file2_button = ttk.Button(self.button_frame, text="Load train.csv", command=lambda: self.load_csv(os.path.join("Data", "train.csv")))
        self.file2_button.grid(row=0, column=1, padx=5)

        # Treeview (Table)
        self.tree = ttk.Treeview(root, show="headings")
        self.tree.pack(expand=True, fill="both")

        # Scrollbars
        self.vsb = ttk.Scrollbar(root, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(root, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")

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
            
            


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVViewerApp(root)
    root.mainloop()
