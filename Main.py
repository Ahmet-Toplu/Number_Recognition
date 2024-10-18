import os
import json
from tkinter import *
from tkinter import messagebox
from tkinter.simpledialog import askinteger
from PIL import ImageGrab, Image
from Recognition import AI

class App:
    def __init__(self):
        self.last_x, self.last_y = None, None

        # tkinter window values
        self.width, self.height = 500, 500
        self.canvas_height, self.canvas_width = (self.width*85/100), self.height
        self.window = Tk()
        self.window.title("Number Recognition")
        self.window.geometry("%dx%d" % (self.width, self.height))
        self.window.resizable(False, False)
        self.canvas = Canvas(self.window, bg="black", height=self.canvas_height, width=self.canvas_width)
        self.window.attributes('-topmost', True)
        self.ai = AI()
        self.ai.load_model(str(self.ai.model_version))

        # File to store counters
        self.COUNTERS_FILE = './Number_Recognition/counters.json'

        # Load the counters when the app starts
        self.c_counter, self.i_counter = self.load_counters()

    # Function to load counters from file
    def load_counters(self):
        if os.path.exists(self.COUNTERS_FILE):
            with open(self.COUNTERS_FILE, 'r') as file:
                counters = json.load(file)
                return counters.get('correct', 0), counters.get('incorrect', 0)
        else:
            # Return initial values if the file does not exist
            return 0, 0

    # Function to save counters to file
    def save_counters(self):
        with open(self.COUNTERS_FILE, 'w') as file:
            json.dump({'correct': self.c_counter, 'incorrect': self.i_counter}, file)

    def mainloop(self):
        # creating buttons
        self.detectB = Button(self.window, text="Detect", command=self.detect_number)
        self.detectB.place(x=(self.width*80/100), y=(self.height*90/100))
        self.clearB = Button(self.window, text="Clear", command=self.clear_canvas)
        self.clearB.place(x=(self.width*5/100), y=(self.height*90/100))

        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Button-1>", self.on_click)

        # creating the main loop
        self.window.mainloop()

    def on_drag(self, event):
        x, y = event.x, event.y
        if x != self.last_x or y != self.last_y:
            if self.last_x is not None and self.last_y is not None:
                self.canvas.create_line(self.last_x, self.last_y, x, y, fill="white", width=10)
            self.last_x, self.last_y = x, y

        self.window.attributes('-topmost', False)

    def on_click(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")

    def detect_number(self):
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        image = ImageGrab.grab(bbox=(x, y, x1, y1))

        new_width = 28
        new_height = 28
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Ensure the directory exists before saving the file
        output_dir = "./Number_Recognition/images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        resized_image.save(f"{output_dir}/resized_canvas_image.bmp")
        prediction = self.ai.predict(f"{output_dir}/resized_canvas_image.bmp")
        message = messagebox.askyesno("True or False", "Is the output correct?")
        if message:
            self.correct(prediction)
        else:
            self.incorrect(prediction)

    # ask user input for the correct number
    def incorrect(self, prediction):
        self.entry = askinteger("Correct Number", "Enter the correct number")
        new_filename = f"./Number_Recognition/images/incorrect_{self.i_counter}_predicted_{prediction}_label_{self.entry}.bmp"
        os.rename("./Number_Recognition/images/resized_canvas_image.bmp", new_filename)
        self.i_counter += 1
        self.save_counters()  # Save the updated counters
        self.ai.fine_tune_model(new_filename, int(self.entry))
        self.clear_canvas()

    def correct(self, prediction):
        self.entry = askinteger("Correct Number", "Enter the correct number")
        new_filename = f"./Number_Recognition/images/correct_{self.c_counter}_predicted_{prediction}_label_{self.entry}.bmp"
        os.rename("./Number_Recognition/images/resized_canvas_image.bmp", new_filename)
        self.c_counter += 1
        self.save_counters()  # Save the updated counters
        self.clear_canvas()

if __name__ == "__main__":
    App().mainloop()