import os
import glob
import argparse
import ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.io import imread
from tensorflow import keras

def run_model(orig_img):
    """
    Run model locally to get predicted heatmap.
    """
    img = orig_img.astype('float32') / 255
    model_path = os.path.join('models/h5/best_model.h5')
    model = keras.models.load_model(model_path)
    pred_heatmap = np.squeeze(model.predict(img[None,:,:,:]))
    pred_heatmap = np.clip(pred_heatmap, 0, 1)
    pred_heatmap = (pred_heatmap * 255).astype('uint8')
    return pred_heatmap


def parse_text_image(text_preds_path):
    """
    Parse the result from GCP deployment to get predicted heatmap.
    """
    with open(text_preds_path) as f:
        lines = f.readlines()
    img_list = ast.literal_eval(lines[1])
    pred_heatmap = np.squeeze(np.array(img_list))
    pred_heatmap = np.clip(pred_heatmap, 0, 1)
    pred_heatmap = (pred_heatmap*255).astype('uint8')
    return pred_heatmap


def create_figure(img, heatmap, pred_heatmap, output_name):
    """
    Save a figure showing the predicted heatmap.
    """
    fig_dir = 'figs'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_path = os.path.join(fig_dir, 'preds_{}.jpg'.format(output_name))

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Image')
    plt.subplot(1,3,2)
    plt.imshow(heatmap, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Ground Truth Heatmap')
    plt.title('{} predictions'.format(output_name))
    plt.subplot(1,3,3)
    plt.imshow(pred_heatmap, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Predicted Heatmap')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def process_input(image_path, heatmap_path, output_name, text_preds_path):
    img = imread(image_path)
    heatmap = imread(heatmap_path)
    if text_preds_path is None:  # run the model locally
        pred_heatmap = run_model(img)
    else:
        pred_heatmap = parse_text_image(text_preds_path)
    create_figure(img, heatmap, pred_heatmap, output_name)


def main():
    parser = argparse.ArgumentParser(description='Tool for viewing model predictions.')
    parser.add_argument('--image_path', help='Path to test image (required)')
    parser.add_argument('--heatmap_path', help='Path to test heatmap (required)')
    parser.add_argument('--output_name', help='Name for type of model predictions (required)')
    parser.add_argument('--text_preds_path', help='Path to prediction image in a text file (optional)', default=None)
    args = parser.parse_args()
    process_input(args.image_path, args.heatmap_path, args.output_name, args.text_preds_path)


if __name__ == "__main__":
    main()

