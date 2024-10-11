import unittest
from tkinter import Event
from PIL import Image
from Main import App

class TestNumberRecognition(unittest.TestCase):

    def setUp(self):
        # Set up the NumberRecognition instance for each test
        self.app = App()
        self.app.window.update()  # Ensure the window is fully loaded for accurate testing

    def test_on_drag(self):
        # Simulate a drag event
        event = Event()
        event.x, event.y = 10, 10
        
        self.app.on_drag(event)
        self.assertEqual(self.app.last_x, 10, "X value is incorrect after drag")
        self.assertEqual(self.app.last_y, 10, "Y value is incorrect after drag")
        print("Test passed: on_drag updates last_x and last_y correctly")

    def test_on_click(self):
        # Simulate a click event
        event = Event()
        event.x, event.y = 20, 20
        
        self.app.on_click(event)
        self.assertIsNone(self.app.last_x, "last_x should be None after click")
        self.assertIsNone(self.app.last_y, "last_y should be None after click")
        print("Test passed: on_click sets last_x and last_y to None correctly")

    def test_clear_canvas(self):
        # Draw something on the canvas
        self.app.canvas.create_line(0, 0, 100, 100)
        # Assert that the canvas has items
        self.assertNotEqual(len(self.app.canvas.find_all()), 0, "Canvas should have items before clearing")
        print("Test passed: clear_canvas has items before clearing")

        # Clear the canvas
        self.app.clear_canvas()
        # Assert that the canvas is empty after clearing
        self.assertEqual(len(self.app.canvas.find_all()), 0, "Canvas should be empty after clearing")
        print("Test passed: clear_canvas clears the canvas correctly")

    def test_detect_number(self):
        # Mock the image grab and resizing process
        x, y, x1, y1 = (self.app.canvas.winfo_rootx(),
                        self.app.canvas.winfo_rooty(),
                        self.app.canvas.winfo_rootx() + self.app.canvas.winfo_width(),
                        self.app.canvas.winfo_rooty() + self.app.canvas.winfo_height())

        # Test the detect_number method
        self.app.detect_number()

        # Verify that the image was saved correctly
        try:
            img_path = "./images/resized_canvas_image.bmp"
            with Image.open(img_path) as img:
                self.assertEqual(img.size, (28, 28), "Image size should be 28x28 after resizing")
                print("Test passed: detect_number saves and resizes the image correctly")
        except FileNotFoundError:
            self.fail("File not found: The image was not saved correctly")

    def tearDown(self):
        # Destroy the Tkinter window after each test
        self.app.window.destroy()

if __name__ == "__main__":
    unittest.main()
