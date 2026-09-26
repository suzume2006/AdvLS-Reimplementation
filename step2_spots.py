"""
Step 2 of AdvLS (digital-only implementation): define a single laser spot
theta = (m, n, r, g, b) and a spot group G_theta = [theta_1, theta_2, ...].

This module gives you:
  - LaserSpot: one spot's parameters, with bounds-checking.
  - random_spot(): generate one random spot within the image bounds.
  - random_spot_group(): generate a full group (the paper uses 10-50 spots).
  - Conversions to/from a flat numeric vector, since the genetic algorithm
    in step 4 works on flat "chromosomes" rather than nested objects.

Usage example (run this file directly to see it in action):
    python step2_spots.py
"""

import random
from dataclasses import dataclass
from typing import List


# Fixed radius for every spot, in pixels. The paper does not optimize this --
# it's a constant you choose. Tune it to taste; larger spots are more visible
# but might fool the classifier more easily.
SPOT_RADIUS = 8


@dataclass
class LaserSpot:
    """One laser spot: theta = (m, n, r, g, b)."""
    m: int      # horizontal center position (pixels)
    n: int      # vertical center position (pixels)
    r: int      # red channel, 0-255
    g: int      # green channel, 0-255
    b: int      # blue channel, 0-255

    def to_vector(self):
        """Flatten to [m, n, r, g, b] for GA chromosome use."""
        return [self.m, self.n, self.r, self.g, self.b]

    @staticmethod
    def from_vector(vec):
        """Rebuild a LaserSpot from a flat [m, n, r, g, b] list."""
        m, n, r, g, b = vec
        return LaserSpot(int(m), int(n), int(r), int(g), int(b))


def random_spot(image_width, image_height, color_mode="random"):
    """
    Generate one random spot within the image bounds.
    color_mode: "random" (any RGB), or "red"/"green"/"blue" to restrict
    the spot to that single channel (matching the paper's ablation study,
    Table 1 / Figure 8, which tests random vs. red vs. green vs. blue).
    """
    m = random.randint(0, image_width - 1)
    n = random.randint(0, image_height - 1)

    if color_mode == "red":
        r, g, b = 255, 0, 0
    elif color_mode == "green":
        r, g, b = 0, 255, 0
    elif color_mode == "blue":
        r, g, b = 0, 0, 255
    else:  # random color
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

    return LaserSpot(m, n, r, g, b)


def random_spot_group(image_width, image_height, num_spots=20, color_mode="random"):
    """
    Generate a full spot group G_theta = [theta_1, ..., theta_num_spots].
    The paper uses group sizes from 10 to 50 -- pick a number in that range.
    """
    return [random_spot(image_width, image_height, color_mode) for _ in range(num_spots)]


def group_to_vector(group: List[LaserSpot]):
    """Flatten an entire spot group into one long chromosome:
    [m1,n1,r1,g1,b1, m2,n2,r2,g2,b2, ...]. This is what the GA will mutate
    and crossover in step 4."""
    flat = []
    for spot in group:
        flat.extend(spot.to_vector())
    return flat


def vector_to_group(vec):
    """Rebuild a spot group from a flat chromosome (inverse of group_to_vector)."""
    group = []
    for i in range(0, len(vec), 5):
        group.append(LaserSpot.from_vector(vec[i:i + 5]))
    return group


if __name__ == "__main__":
    # Demo: generate a group of 20 random-colored spots for a 224x224 image
    # (224x224 is ResNet50's standard input size).
    group = random_spot_group(image_width=224, image_height=224, num_spots=20, color_mode="random")

    print(f"Generated a spot group with {len(group)} spots:\n")
    for i, spot in enumerate(group):
        print(f"  Spot {i+1}: {spot}")

    flat = group_to_vector(group)
    print(f"\nFlattened chromosome length: {len(flat)} (should be {len(group)} spots x 5 values)")

    rebuilt = vector_to_group(flat)
    print(f"Rebuilt group matches original: {rebuilt == group}")
