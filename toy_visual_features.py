import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def _assert_imgs(imgs: torch.Tensor):
    assert imgs.ndim == 5 and imgs.shape[0] == 1 and imgs.shape[1] == 1 and imgs.shape[2] == 3, \
        f"Expected (1,1,3,H,W), got {tuple(imgs.shape)}"


@torch.no_grad()
def _gray01_from_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: (1,1,3,H,W), float in [0,1]
    returns gray: (H,W) float in [0,1]
    """
    _assert_imgs(imgs)
    x = imgs[0, 0]  # (3,H,W)
    gray = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
    return gray.clamp(0, 1)


@torch.no_grad()
def _u8_from_gray01(gray01: torch.Tensor) -> np.ndarray:
    """
    gray01: (H,W) float in [0,1]
    returns uint8 (H,W) in [0,255]
    """
    return (gray01.mul(255.0).round().to(torch.uint8).cpu().numpy())

def load_single_image_as_imgs(image_path: str, device="cpu") -> torch.Tensor:
    """
    Returns imgs tensor with shape (1, 1, 3, H, W), float32 in [0,1]
    """

    # load with OpenCV (BGR)
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)

    # BGR -> RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # uint8 -> float32 [0,1]
    rgb01 = rgb.astype(np.float32) / 255.0  # (H,W,3)

    # to torch, channel-first
    x = torch.from_numpy(rgb01).permute(2, 0, 1)  # (3,H,W)

    # add (B=1, N=1)
    imgs = x.unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)

    return imgs.to(device)


if __name__ == "__main__":
    VIS = True

    image_path = "/data/wanghaoxuan/sintel/evaluation/final/temple_3/frame_0001.png"
    imgs = load_single_image_as_imgs(image_path)

    # binarization with Otsu
    x = imgs[0, 0]  # (3,H,W)
    gray = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
    gray = gray.clamp(0, 1)
    gray_u8 = gray.mul(255.0).round().to(torch.uint8).cpu().numpy()
    _, bin_u8 = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    # optional visualization
    if VIS:
        plt.figure(figsize=(6, 6))
        plt.imshow(bin_u8, cmap="gray")
        plt.title("Otsu Binarization")
        plt.axis("off")
        plt.show()
        plt.imsave("otsu_binarization_example.png", bin_u8, cmap="gray")

    # Canny edge detection
    gray_blur = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    edges_u8 = cv2.Canny(
        gray_blur,
        threshold1=50,
        threshold2=150,
        apertureSize=3,
        L2gradient=True
    )

    # optional visualization
    if VIS:
        plt.figure(figsize=(6, 6))
        plt.imshow(edges_u8, cmap="gray")
        plt.title("Canny Edge Detection")
        plt.axis("off")
        plt.show()
        plt.imsave("canny_edge_detection_example.png", edges_u8, cmap="gray")
    
