import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Get top n indices of an array
get_top_n = lambda array, n: (-array).argsort()[:,:n]

# Get the labels for a given list of predictions
get_predictions = lambda predictions, categories: [[categories[i] for i in p] for p in predictions]

# zip the predictions, result and labels
get_zipped = lambda result , predictions, top_n: [list(zip(result[i,a], predictions[i])) for i,a in enumerate(top_n)]


def  _plot_image(ax, img):
    ax.imshow(img)
    ax.tick_params(axis='both',       
                   which='both',      
                   bottom=False,      
                   top=False,         
                   left=False,
                   right=False,
                   labelleft=False,
                   labelbottom=False) 
    return ax


def _plot_prediction_bar(ax, result):
    result = result[::-1]
    perf = [category[0] for category in result]
    ax.barh(range(len(perf)), perf, align='center', color='#33ccff')
    ax.tick_params(axis='both',       
                   which='both',      
                   bottom=False,      
                   top=False,         
                   left=False,
                   right=False,
                   labelbottom=False) 
    tick_labels = [category[1].split(',')[0] for category in result]
    ax.yaxis.set_ticks(range(len(perf)))
    ax.yaxis.set_ticklabels(tick_labels, position=(0.5,0), minor=False, horizontalalignment='center')

    
def plot_predictions(images, results, categories, N):
    """Plot a list of images with associated top-N predictions
        
        arguments:
        images -- an array of images
        results -- a list of np arrays of shape [1,N_Categories]
        categories -- an array of str representing the labels
        N -- the number of predictions to display
    """
    
    factor = int((len(images)/6)+1)
    top_n = get_top_n(results, N)
    predictions = get_predictions(top_n, categories)
    zipped = get_zipped(results, predictions, top_n)
    gs = gridspec.GridSpec(factor+1, 3)
    fig = plt.figure(figsize=(15, int(5*(factor+1)+N/3)))
    gs.update(hspace=0.1, wspace=0.001)
    
    for gg, results, img in zip(gs, zipped, images):
        gg2 = gridspec.GridSpecFromSubplotSpec(6+int(N/3), 10, subplot_spec=gg)
        ax = fig.add_subplot(gg2[0:5, :])
        _plot_image(ax, img)
        ax = fig.add_subplot(gg2[5:6+int(N/3), 1:9])
        _plot_prediction_bar(ax, results)
