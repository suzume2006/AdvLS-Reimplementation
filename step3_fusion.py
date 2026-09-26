"""
Step 3 of AdvLS (digital-only implementation): the fusion function
S(X, G_theta) -- takes a clean image X and a spot group G_theta (from step 2)
and produces the adversarial image by rendering glowing laser-spot blobs
onto it.

Usage:
    pip install pillow numpy
    python step3_fusion.py --image "C:\\path\\to\\one\\image.jpeg" --num_spots 20 --output adversarial_sample.png

This reads one image, generates a random spot group (using step2_spots.py),
renders it, and saves the result so you can visually inspect it -- exactly
like Figure 4 in the paper ("Adversarial samples in the digital environment").
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from step2_spots import LaserSpot, random_spot_group, SPOT_RADIUS


def render_spot_layer(image_size, group):
    """
    Draw all spots in the group onto a transparent layer, with a soft
    Gaussian-blurred glow rather than a hard-edged circle (closer to how a
    real laser spot looks: bright core, soft falloff).
    Returns an RGBA PIL Image the same size as the target photo.
    """
    width, height = image_size
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for spot in group:
        bbox = [
            spot.m - SPOT_RADIUS, spot.n - SPOT_RADIUS,
            spot.m + SPOT_RADIUS, spot.n + SPOT_RADIUS,
        ]
        draw.ellipse(bbox, fill=(spot.r, spot.g, spot.b, 255))

    # Blur the whole spot layer so each dot has a soft glowing edge instead
    # of a hard-edged circle -- closer to a real laser spot's appearance.
    layer = layer.filter(ImageFilter.GaussianBlur(radius=SPOT_RADIUS / 2))
    return layer


def fuse(clean_image: Image.Image, group, alpha=0.75):
    """
    S(X, G_theta): alpha-blend the rendered spot layer onto the clean image.
    alpha controls how strong/opaque the spots appear (0 = invisible,
    1 = fully opaque laser color, no original image showing through at
    the spot centers). 0.75 gives a bright but not fully saturated spot,
    similar to the paper's example images.
    """
    clean_rgba = clean_image.convert("RGBA")
    spot_layer = render_spot_layer(clean_image.size, group)

    # Only blend where the spot layer has non-zero alpha (i.e. where a spot
    # actually is), scaled by our chosen blend strength.
    spot_array = np.array(spot_layer).astype(np.float32)
    clean_array = np.array(clean_rgba).astype(np.float32)

    spot_alpha = (spot_array[:, :, 3:4] / 255.0) * alpha  # shape (H, W, 1)
    blended_rgb = clean_array[:, :, :3] * (1 - spot_alpha) + spot_array[:, :, :3] * spot_alpha
    result_array = np.concatenate([blended_rgb, clean_array[:, :, 3:4]], axis=2)
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)

    return Image.fromarray(result_array, mode="RGBA").convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Step 3: render a laser spot group onto an image")
    parser.add_argument("--image", required=True, help="Path to a clean image (e.g. one from correctly_classified.json)")
    parser.add_argument("--num_spots", type=int, default=20)
    parser.add_argument("--color_mode", default="random", choices=["random", "red", "green", "blue"])
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--output", default="adversarial_sample.png")
    args = parser.parse_args()

    clean_image = Image.open(args.image).convert("RGB")
    print(f"Loaded {args.image}, size {clean_image.size}")

    group = random_spot_group(
        image_width=clean_image.size[0],
        image_height=clean_image.size[1],
        num_spots=args.num_spots,
        color_mode=args.color_mode,
    )
    print(f"Generated {len(group)} spots (color_mode={args.color_mode})")

    adversarial_image = fuse(clean_image, group, alpha=args.alpha)
    adversarial_image.save(args.output)
    print(f"Saved adversarial sample to {args.output}")
    print("Open both the original and this file side by side to see the effect of the spots.")


if __name__ == "__main__":
    main()
