import matplotlib.pyplot as pyplot
from PIL import Image, ImageDraw
import napari
import numpy as np 
import pandas as pd 
import time

path_image = "D:\\Tommy\\em_targeting\\Jelly_fish1.png"

image = Image.open(path_image)
image = np.array(image)

viewer = napari.Viewer()
viewer.add_image(image)
viewer.add_shapes(name="tissue")
napari.run()

mask = np.zeros_like(image[:,:,0])

for polygon in viewer.layers["tissue"].data:
    polygon = [(e[0], e[1]) for e in polygon]
    img = Image.new('L', image.shape[:2], 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

    mask += np.array(img).T

mask = mask > 0

viewer = napari.Viewer()
viewer.add_image(image)
viewer.add_labels(mask)
napari.run()