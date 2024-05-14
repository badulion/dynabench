import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

cdict = {'red':   [[0.0,  33/255, 33/255],
                   [0.3,  1.0, 1.0],
                   [0.65,  1.0, 1.0],
                   [0.8,  1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  79/255, 33/255],
                    [0.3,  1.0, 1.0],
                    [0.65,  1.0, 1.0],
                    [0.8,  1.0, 1.0],
                    [1.0,  1.0, 1.0]],
         'blue':  [[0.0,  0.0, 0.0],
                    [0.3,  1.0, 1.0],
                    [0.65,  1.0, 1.0],
                    [0.8,  1.0, 1.0],
                    [1.0,  1.0, 1.0]],}

def hex_to_rgb(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def cgrad(color_list, anchors):
    colors = [hex_to_rgb(c) for c in color_list]
    anchors = np.array(anchors).reshape((-1,1))
    reds = np.array([c[0] for c in colors]).reshape((-1,1))
    green = np.array([c[1] for c in colors]).reshape((-1,1))
    blue = np.array([c[2] for c in colors]).reshape((-1,1))
    cdict = {
        'red': np.hstack([anchors, reds, reds]),
        'green': np.hstack([anchors, green, green]),
        'blue': np.hstack([anchors, blue, blue]),
    }
    return LinearSegmentedColormap('neuralODE', segmentdata=cdict, N=256)



NEURALPDE_COLORMAP = cgrad(["#214F26", "#41B54F", "#F0BD35", "#DF721A", "#BA1F16"], [0,0.3,0.65,0.8,1])


if __name__ == "__main__":
    x, y = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    z = np.sin(x)*np.sin(y)

    plt.imshow(z, cmap=NEURALPDE_COLORMAP)
    plt.show()

