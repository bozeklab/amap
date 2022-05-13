import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def set_style():
    #font_dirs = ['/Users/kasiabozek/fonts', ]
    #font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    #font_list = font_manager.createFontList(font_files)
    #font_manager.fontManager.ttflist.extend(font_list)

    plt.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.1, wspace=0.2, hspace=0.5)
    params = {'axes.labelsize': 8,
              'axes.labelpad': 1.0,
              'axes.titlepad' : 3.0,
              'axes.titlesize': 9,
              'xtick.labelsize': 7,
              'xtick.major.pad' : -1,
              'ytick.labelsize': 7,
              'axes.linewidth': 0.5,
              'ytick.major.pad': 1.0,
              'legend.fontsize': 9,
              'legend.title_fontsize': 12,
              'legend.labelspacing': 0.1,
              'legend.markerscale':0.5,
              "legend.frameon": True,
              'font.family': "sans-serif",
              'font.sans-serif': 'Arial',
              'grid.linewidth': 0.5
              }
    pylab.rcParams.update(params)
    sns.set_style("white")

def composite(background, foreground, alpha):
    for color in range(3):
        background[:, :, color] = alpha * foreground[:, :, color] + background[:, :, color] * (1 - alpha)

    return background.astype(np.uint8)

