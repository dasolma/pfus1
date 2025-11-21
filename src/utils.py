from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import random
import os
import json
import cv2


COLORS = [(255,0,0),(0,255,0),(255,255,0),(0,0,255),(255,0,255),(0,255,255),(128,255,0),(128,0,255)]
FOLD_COLORS = ['blue', 'green', 'orange', 'red', 'purple']
ALPHA = 0.4

def read_image_and_masks(data_dir, file, output_image_size):
    # Read image in grayscale
    im = cv2.imread(os.path.join(data_dir, f"{file}.png"), cv2.IMREAD_GRAYSCALE)

    # Calculate resize factors and resize the image to input dimensions
    wfactor, hfactor = output_image_size[0] / im.shape[0], output_image_size[1] / im.shape[1]
    im = cv2.resize(im, tuple(reversed(output_image_size)))

    # Read polygon data from JSON file and preprocess polygons
    polys = json.load(open(os.path.join(data_dir, f"{file}.json")))
    polys = sorted(polys, key=lambda x: x['label'])
    polys = [np.array([[x * hfactor, y * wfactor] for x, y in p['pol']]) for p in polys]

    # Initialize masks
    masks = np.zeros(output_image_size + (9,))

    # Draw polygons on masks for each polygon
    for j, poly in enumerate(polys):
        poly = poly.round().astype(int)
        mask = np.zeros(output_image_size)
        cv2.fillPoly(mask, pts=[poly], color=(1,))
        masks[:, :, j] = mask

    # Create a background mask as the 9th channel
    masks[:, :, 8] = 1 - masks.sum(axis=-1).clip(0, 1)

    return im, masks

def read_norm_image_and_polys(_dir, frame, im_size=(128, 128)):
    im = cv2.imread(os.path.join(_dir, f"frame_{frame:03}.png"), cv2.IMREAD_GRAYSCALE)
    
    wfactor, hfactor = im_size[0] / im.shape[0], im_size[1] / im.shape[1]
    im_model = cv2.resize(im,tuple(reversed((128, 128))))
    im = cv2.resize(im,tuple(reversed(im_size)))

    # read polys
    polys = json.load(open(os.path.join(_dir, f"frame_{frame:03}.json")))
    polys = sorted(polys, key=lambda x: x['label'])
    polys = [{'pol': np.array([[x * hfactor, y * wfactor] for x, y in p['pol']]), 
              'label': p['label']} for p in polys]
    
    return im, polys, im_model


def draw_poly(image, ps, colors, thickness=1, dashed = False):
    
    ini = None
    for i, (pol, color) in enumerate(zip(ps, colors)):
        pol = pol.round().astype(int)
        for p1, p2 in zip(pol[0:], pol[1:]):

            if dashed:
                drawline(image, tuple(p1), tuple(p2), color=color, thickness=thickness)
            else:
                cv2.line(image, tuple(p1), tuple(p2), color=color, thickness=thickness)
           
        ini = pol[0]
        if dashed:
            drawline(image, tuple(p2), tuple(ini), color=color, thickness=thickness)
        else:
            cv2.line(image, tuple(p2), tuple(ini), color=color, thickness=thickness)

def show_predictions(model, cases, axs, fold, data_dir, num_samples=6):
            
    #model = models[0]['model']
    #case = random.sample(models[0]['val_cases'], 3)
    cases = random.sample(cases, num_samples)
    for patient, ax in zip(cases, axs):
        _dir = os.path.join(data_dir, patient)
        n = len(os.listdir(_dir)) // 2
        i = random.randint(0, n-1)

        m, polys, im_model = read_norm_image_and_polys(_dir, i, im_size=(512, 720)) # cv2.imread(os.path.join(_dir, f"frame_{i:03}.png"), cv2.IMREAD_GRAYSCALE)
        polys = [p['pol'] for p in polys]

        rgb_im = np.stack((m,m,m), axis=-1)

        draw_poly(rgb_im, polys, COLORS, thickness=3)

        prediction = model.predict(np.expand_dims(im_model, 0))[0] 
        prediction = cv2.resize(prediction.astype('float32'), (720,512), interpolation = cv2.INTER_AREA)


        prediction_segment = np.zeros(shape=(512, 720, 3))
        for i, c in enumerate(COLORS):
            #c = np.array(c) / 255
            mask = (np.stack((prediction[:,:,i],)*3, axis=-1) * c * ALPHA)
            prediction_segment = prediction_segment + mask


        prediction_segment = prediction_segment.astype(int)


        ax.imshow(np.clip(rgb_im + prediction_segment, 0, 255))
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        
    axs[0].set_title(f"Fold {fold}", fontsize=20, color=FOLD_COLORS[fold-1])
        

    
def genenerate_epoch_graph(models, data_dir, output_dir):
    
    fig = plt.figure(figsize=(20, 10))

    gs = gridspec.GridSpec(6, 8)

    axs = []
    ax1 = fig.add_subplot(gs[0:2,0:3])
    ax2 = fig.add_subplot(gs[2:4,0:3])
    ax3 = fig.add_subplot(gs[4:6,0:3])
    axs.append([ax1, ax2, ax3])

    for f in range(5):
        col = 3 + f
        fold_ax = []
        for i in range(6):
            ax = fig.add_subplot(gs[i:i+1,col])
            fold_ax.append(ax)
        axs.append(fold_ax)
    axs = np.array(axs)
    
    for i in range(len(models)):
        model = models[i]
        
        # plot metrics
        nmetrics = len(model['scores'])
        for mi, (ax, (metric, score)) in enumerate(zip(axs[0], model['scores'].items())):
            ax.plot(model['scores'][metric], c=FOLD_COLORS[i])
            ax.set_ylabel(metric.capitalize(), fontsize=20)
            if mi < nmetrics-1:
                ax.set_xticklabels([])
                ax.set_xticks([])
            else:
                ax.set_xlabel('Epochs', fontsize=20)
                
        # plot validation samples
        show_predictions(model['model'], model['val_cases'], axs[-(5-i)], i+1, data_dir=data_dir)

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(os.path.join(output_dir, f"epoch_{len(model['scores'][metric])}"))


def visualize_image_and_masks(im, masks, organ_names=ORGAN_NAMES):
    """
    Visualize the original image and the 8 organ masks.
    
    Args:
        im (numpy.ndarray): Grayscale image (H, W).
        masks (numpy.ndarray): Segmentation masks (H, W, 9).
        organ_names (list): List of organ names for the 8 classes.
    """
    # Create figure with 3 rows and 3 columns
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()
    
    # Plot original image in the first subplot
    axes[0].imshow(im, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot each of the 8 organ masks
    for i in range(8):
        axes[i + 1].imshow(masks[:, :, i], cmap='gray')
        axes[i + 1].set_title(organ_names[i], fontsize=12, fontweight='bold')
        axes[i + 1].axis('off')
    
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.show()
