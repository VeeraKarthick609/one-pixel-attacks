#!/usr/bin/env python3

import numpy as np
import pandas as pd
from keras.datasets import cifar10 # type: ignore
import pickle
import os
from PIL import Image
import time  # New import for timing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Helper functions
from differential_evolution import differential_evolution
import helper
from networks.resnet import ResNet

class PixelAttacker:
    def __init__(self, model, data, class_names, dimensions=(32, 32)):
        # Load data and model
        self.model = model
        self.x_test, self.y_test = data
        self.class_names = class_names
        self.dimensions = dimensions

        network_stats, correct_imgs = helper.evaluate_models([self.model], self.x_test, self.y_test)
        self.correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
        self.network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    def predict_classes(self, xs, img, target_class, minimize=True):
        imgs_perturbed = helper.perturb_image(xs, img)
        predictions = self.model.predict(imgs_perturbed)[:, target_class]
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, targeted_attack=False, verbose=False):
        attack_image = helper.perturb_image(x, img)
        confidence = self.model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)

        if verbose:
            print('Confidence:', confidence[target_class])
        return ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class))

    def attack(self, img_id, target=None, pixel_count=1, maxiter=75, popsize=400, verbose=False, plot=False):
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img_id, 0]
        dim_x, dim_y = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count
        popmul = max(1, popsize // len(bounds))

        def predict_fn(xs):
            return self.predict_classes(xs, self.x_test[img_id], target_class, target is None)

        def callback_fn(x, convergence):
            return self.attack_success(x, self.x_test[img_id], target_class, targeted_attack, verbose)

        # Time the perturbation process
        start_perturb_time = time.time()
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)
        end_perturb_time = time.time()
        perturb_time = end_perturb_time - start_perturb_time

        attack_image = helper.perturb_image(attack_result.x, self.x_test[img_id])[0]
        prior_probs = self.model.predict(np.array([self.x_test[img_id]]))[0]
        predicted_probs = self.model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img_id, 0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        if plot:
            helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)

        return [self.model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs, attack_result.x, attack_image, perturb_time]

    def attack_all(self, samples=100, pixels=(1, 3, 5), targeted=False, maxiter=75, popsize=400, verbose=True):
        results = []
        valid_imgs = self.correct_imgs.img
        img_samples = np.random.choice(valid_imgs, samples)
        total_start_time = time.time()  # Start total timing

        for pixel_count in pixels:
            for i, img in enumerate(img_samples):
                print(self.model.name, '- image', img, '-', i + 1, '/', len(img_samples))
                targets = [None] if not targeted else range(10)

                for target in targets:
                    if targeted and target == self.y_test[img, 0]:
                        continue
                    start_time = time.time()
                    result = self.attack(img, target, pixel_count,
                                         maxiter=maxiter, popsize=popsize, verbose=verbose)
                    end_time = time.time()
                    attack_time = end_time - start_time
                    result.append(attack_time)
                    results.append(result)

        total_end_time = time.time()
        total_time = total_end_time - total_start_time  # Calculate total time
        avg_time_per_attack = total_time / len(results) if results else 0

        return results, total_time, avg_time_per_attack

if __name__ == '__main__':
    _, test = cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    model = ResNet(load_weights=True)

    attacker = PixelAttacker(model, test, class_names)
    print('Starting attack')

    results, total_time, avg_time_per_attack = attacker.attack_all(samples=100, pixels=(1, 3, 5), targeted=False,
                                                                   maxiter=75, popsize=400, verbose=True)

    columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs',
               'perturbation', 'perturbed_image', 'perturb_time', 'attack_time']
    results_table = pd.DataFrame(results, columns=columns)
    results_table['total_time'] = total_time
    results_table['avg_time_per_attack'] = avg_time_per_attack

    successful_attacks = results_table[results_table['success'] == True]
    csv_filename = 'successful_attacks.csv'
    successful_attacks[['model', 'pixels', 'image', 'true', 'predicted', 'cdiff', 'perturb_time', 'attack_time',
                        'total_time', 'avg_time_per_attack']].to_csv(csv_filename, index=False)
    print(f'Saved successful attacks to {csv_filename}')

    output_dir = 'perturbed_images'
    os.makedirs(output_dir, exist_ok=True)
    for index, row in successful_attacks.iterrows():
        img = row['perturbed_image']
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        filename = f"{output_dir}/{row['model']}_{row['image']}_{row['true']}_{row['predicted']}.png"
        img.save(filename)
    print(f'Saved {len(successful_attacks)} perturbed images to {output_dir}')

    results_file = 'results.pkl'
    with open(results_file, 'wb') as file:
        pickle.dump(results, file)
    print(f'Saved results to {results_file}')
