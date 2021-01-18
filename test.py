# test/inference script
import numpy as np
import torch
from torch import nn
import argparse
import time
import os
import sys
import cv2
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.hrnet import hrnet
from tqdm import tqdm


def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


def preprocess_image(image):
    image = normalize(image)
    return torch.unsqueeze(image, dim=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_images",
        type=str,
        required=True,
        default="./",
        help="Path where all the test images are located or you can give path to video, it will break into each frame and write as a video",
    )

    parser.add_argument(
        "--path_to_weights",
        type=str,
        required=True,
        default="./",
        help="Path to weights for which inference needs to be done",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="path to save checkpoints and wandb, final output path will be this path + wandbexperiment name so the output_dir should be root directory",
    )


    parser.add_argument(
        "--is_video",
        action="store_true",
        help="If true path to image will be video and code will write a video of mask with same name.",
    )

    args = parser.parse_args()


    model = hrnet()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(args.path_to_weights)["state_dict"])
    model.eval()

    if args.is_video:
        print("testing on video")
        current_video = cv2.VideoCapture(args.path_to_images)
        # width = current_video.get(cv2.CAP_PROP_FRAME_WIDTH)
        # height = current_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = current_video.get(cv2.CAP_PROP_FPS)
        out_video = cv2.VideoWriter(
            os.path.join(args.output_dir, args.path_to_images.split("/")[-1]),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (
                1920,
                1080 * 2,
            ),  # change it to width,height if there is no cropping operation performed on image
        )

        for idx, frame in enumerate(tqdm(frame_extract(args.path_to_images))):
            # remove this line if there is no preprocessing step for video
            frame1 = frame
            frame = cv2.resize(frame, (256, 144))
            image = preprocess_image(frame, cfg)
            if torch.cuda.is_available():
                image = image.cuda()
            st = time.time()
            prediction = model(image, (frame1.shape[0], frame1.shape[1]))
            if idx == 0:
                print(time.time()-st)

            prediction = (
                torch.argmax(prediction["output"][0], dim=1)
                .detach()
                .cpu()
                .squeeze(0)
                .numpy()
                .astype(np.uint8)
            ).reshape((frame1.shape[0], frame1.shape[1], 1))
            out_video.write(np.concatenate((frame1, frame1 * prediction), axis=0))
            # print(np.concatenate((frame, frame * prediction), axis=0).shape)
        out_video.release()
        return
    os.makedirs(os.path.join(args.output_dir, "pred_mask"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "pred_plot"), exist_ok=True)
    print("testing on {} images".format(len(os.listdir(args.path_to_images))))

    for image_path in tqdm(os.listdir(args.path_to_images)):
        image = cv2.imread(os.path.join(args.path_to_images, image_path))
        image = preprocess_image(image)
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            prediction = model(image, (args.height, args.dataset.width))
        prediction = (
            torch.argmax(prediction["output"][0], dim=1)
            .cpu()
            .squeeze(dim=0)
            .numpy()
            .astype(np.uint8)
            * 255
        )
        cv2.imwrite(os.path.join(args.output_dir, "pred_mask", image_path), prediction)


if __name__ == "__main__":
    main()
