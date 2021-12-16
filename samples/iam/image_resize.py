from PIL import Image
import os

basewidth = 359
directory = '../forms'
new_directory = 'forms/'
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = Image.open(os.path.join(directory, filename))
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img.save(os.path.join(new_directory, filename))
    else:
        continue
