"""
Step 4 of AdvLS (digital-only implementation): the genetic algorithm that
searches for a spot group G_theta which fools the classifier on a given
image, following Algorithm 1 in the paper.

How it works, matching the paper's Section 3.2/3.4:
    1. Start with a population of random spot groups ("Seed" candidates).
    2. Render each onto the image (step 3's fuse()) and check the model's
       confidence in the TRUE label. Lower confidence = more adversarial
       = better fitness (we are trying to minimize this).
    3. If any candidate causes misclassification, stop immediately -- success.
    4. Otherwise: replace the worst 10% of the population with copies of the
       best 10% (the paper's selection strategy), then crossover pairs of
       spot-group chromosomes with probability P_c=0.7, and mutate individual
       values with probability P_m=0.1.
    5. Repeat until success or a maximum number of generations is reached.

Usage:
    pip install torch torchvision pillow numpy
    python step4_ga.py --image "C:\\path\\to\\image.JPEG" --true_label_idx 1
                        --num_spots 20 --population 20 --generations 50
                        --output adversarial_success.png

--true_label_idx: the ImageNet class index for this image (from your
correctly_classified.json manifest -- each entry already has "true_label_idx").
"""

import argparse
import random

import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

from step2_spots import random_spot_group, group_to_vector, vector_to_group, SPOT_RADIUS
from step3_fusion import fuse
from ga_core import mutate, crossover, select_and_replace


def load_model(device):
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval().to(device)
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    return model, preprocess, categories


def evaluate_confidence(model, preprocess, device, clean_image, group, true_label_idx):
    """
    Render the spot group onto the image, run it through the model, and
    return (confidence_on_true_label, predicted_label_idx).
    This is one "query" to the model -- the paper tracks how many of these
    are needed on average (Table 1's "Query" row).
    """
    adv_image = fuse(clean_image, group)
    x = preprocess(adv_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence_true = float(probs[0, true_label_idx])
        pred_idx = int(probs.argmax(dim=1))
    return confidence_true, pred_idx, adv_image


def run_ga(model, preprocess, device, clean_image, true_label_idx,
           num_spots=20, population_size=20, max_generations=50,
           color_mode="random", verbose=True):
    """
    Runs the full genetic algorithm loop for ONE image.
    Returns: (success: bool, best_group, num_queries, best_adv_image)
    """
    width, height = clean_image.size

    # Step 1: initialize population of random spot groups (as flat vectors).
    population = [
        group_to_vector(random_spot_group(width, height, num_spots, color_mode))
        for _ in range(population_size)
    ]

    num_queries = 0
    best_adv_image = None
    best_ever_vector = None
    best_ever_confidence = float("inf")

    for generation in range(max_generations):
        fitness_scores = []  # (confidence, index) -- lower confidence is "more fit" here

        for i, vector in enumerate(population):
            group = vector_to_group(vector)
            confidence, pred_idx, adv_image = evaluate_confidence(
                model, preprocess, device, clean_image, group, true_label_idx
            )
            num_queries += 1
            fitness_scores.append((confidence, i))

            if confidence < best_ever_confidence:
                best_ever_confidence = confidence
                best_ever_vector = list(vector)

            if pred_idx != true_label_idx:
                if verbose:
                    print(f"  [SUCCESS] Generation {generation}, individual {i}: "
                          f"misclassified after {num_queries} queries "
                          f"(confidence on true label dropped to {confidence:.4f})")
                return True, group, num_queries, adv_image, pred_idx

        # No success yet this generation -- log the best candidate so far.
        best_confidence = min(fitness_scores, key=lambda t: t[0])[0]
        if verbose:
            print(f"  Generation {generation}: best confidence on true label = "
                  f"{best_confidence:.4f} (best ever: {best_ever_confidence:.4f}, "
                  f"queries so far: {num_queries})")

        # Step 4: selection -- replace worst 10% with copies of best 10% (ga_core.py, tested separately).
        population = select_and_replace(fitness_scores, population)

        # Crossover: pair up random individuals and produce children.
        new_population = []
        for _ in range(population_size - 1):  # leave one slot for elitism
            parent_a = random.choice(population)
            parent_b = random.choice(population)
            child = crossover(parent_a, parent_b)
            child = mutate(child, width, height)
            new_population.append(child)

        # Elitism: carry the single best solution EVER found, unchanged, into
        # the next generation. Without this, a good solution found by chance
        # can get mutated away and the search effectively forgets its own progress.
        new_population.append(list(best_ever_vector))
        population = new_population

    if verbose:
        print(f"  [FAILURE] No misclassification found after {max_generations} generations "
              f"({num_queries} queries). Best confidence reached: {best_ever_confidence:.4f}")
    return False, None, num_queries, best_adv_image, None


def main():
    parser = argparse.ArgumentParser(description="Step 4: genetic algorithm attack on one image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--true_label_idx", type=int, required=True,
                         help="ImageNet class index for this image (see correctly_classified.json)")
    parser.add_argument("--num_spots", type=int, default=20)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--color_mode", default="random", choices=["random", "red", "green", "blue"])
    parser.add_argument("--output", default="adversarial_result.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, preprocess, categories = load_model(device)
    clean_image = Image.open(args.image).convert("RGB")

    print(f"Attacking '{args.image}' (true label: {categories[args.true_label_idx]}) "
          f"with {args.num_spots} spots, population={args.population}, "
          f"max {args.generations} generations...\n")

    success, best_group, num_queries, adv_image, pred_idx = run_ga(
        model, preprocess, device, clean_image, args.true_label_idx,
        num_spots=args.num_spots, population_size=args.population,
        max_generations=args.generations, color_mode=args.color_mode,
    )

    if success and adv_image is not None:
        adv_image.save(args.output)
        print(f"\nAttack SUCCEEDED after {num_queries} queries. "
              f"Misclassified as: '{categories[pred_idx]}'. Saved to {args.output}")
    else:
        print(f"\nAttack FAILED after {num_queries} queries. Try more generations, "
              f"a larger population, or more spots.")


if __name__ == "__main__":
    main()
