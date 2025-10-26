import torch
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor # Assuming you have a helper function


if __name__ == "__main__":
    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`