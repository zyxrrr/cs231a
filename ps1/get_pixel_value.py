# CS231A Problem Set 1 
# Prints the pixel value upon mouse click
from Tkinter import *
from PIL import Image, ImageTk
import argparse

class CoordinatePrinter:
    def __init__(self, oval_size, resize_val):
        self.oval_size = oval_size
        self.oval = None
        self.resize_val = resize_val

    # callback function for mouse click
    def print_coordinates(self, event):
        print (event.x*self.resize_val, event.y*self.resize_val)
        if self.oval is not None:
            canvas.delete(self.oval)
        self.oval = canvas.create_oval(event.x-self.oval_size, event.y-self.oval_size, 
                event.x+self.oval_size, event.y+self.oval_size, outline="red", fill="red")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the pixel location in'
            ' an image')
    parser.add_argument('filepath', type=str, nargs='?', default='front.png', 
            help='The path to the image')
    parser.add_argument('-resize-val', type=int, default=2, 
            help='How much to resize the image')
    parser.add_argument('-oval-size', type=int, default=2, 
            help='How big the red dot is when clicked')

    args = parser.parse_args()

    root = Tk()

    # Creating tkinter canvas with image added
    frame = Frame(root, bd=0)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    
    printer = CoordinatePrinter(args.oval_size, args.resize_val)
    img = Image.open(args.filepath)
    width, height = img.size
    width /= printer.resize_val
    height /= printer.resize_val
    img = img.resize((width, height), Image.ANTIALIAS)
    canvas = Canvas(frame, bd=0, width=width, height=height)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    frame.pack(fill=BOTH,expand=1)

    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=img, anchor="nw")

    canvas.bind("<Button 1>", printer.print_coordinates)

    root.mainloop()
