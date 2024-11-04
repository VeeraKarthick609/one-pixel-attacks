#!/usr/bin/env python3

import numpy as np
import pandas as pd
from keras.datasets import cifar10  # type: ignore
import pickle
import os
from PIL import Image
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from differential_evolution import differential_evolution
import helper
from networks.resnet import ResNet
import cma  # CMA-ES package

class PixelAttacker:
    def __init__(self, model, data, class_names, dimensions=(32, 32)):
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

    def differential_evolution_attack(self, predict_fn, bounds, callback_fn, maxiter, popsize):
        return differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popsize, recombination=1, atol=-1,
            callback=callback_fn, polish=False)

    def cma_es_attack(self, predict_fn, bounds, maxiter):
        # Flatten bounds to find global min and max across all parameters
        min_bound = min(b[0] for b in bounds)
        max_bound = max(b[1] for b in bounds)

        dim = len(bounds)
        x0 = np.random.rand(dim) * (max_bound - min_bound) + min_bound  # Initial guess
        sigma = 0.5  # Initial standard deviation
        options = {'maxiter': maxiter, 'bounds': [[min_bound], [max_bound]]}  # Global bounds

        result = cma.fmin2(predict_fn, x0, sigma, options=options)
        return result[0]


    def attack(self, img_id, target=None, pixel_count=1, maxiter=75, popsize=400, algorithm='de', verbose=False):
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img_id, 0]
        dim_x, dim_y = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count

        def predict_fn(xs):
            return self.predict_classes(xs, self.x_test[img_id], target_class, target is None)

        def callback_fn(x, convergence):
            return self.attack_success(x, self.x_test[img_id], target_class, targeted_attack, verbose)

        start_time = time.time()
        
        if algorithm == 'de':
            attack_result = self.differential_evolution_attack(predict_fn, bounds, callback_fn, maxiter, popsize)
        elif algorithm == 'cma-es':
            attack_result = self.cma_es_attack(predict_fn, bounds, maxiter)
        else:
            raise ValueError("Algorithm not supported. Choose 'de' or 'cma-es'.")
        
        end_time = time.time()
        attack_time = end_time - start_time

        attack_image = helper.perturb_image(attack_result, self.x_test[img_id])[0]
        prior_probs = self.model.predict(np.array([self.x_test[img_id]]))[0]
        predicted_probs = self.model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img_id, 0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        return [self.model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs, attack_result, attack_image, attack_time]

    def attack_all(self, samples=100, pixels=(1, 3, 5), targeted=False, maxiter=75, popsize=400, algorithm='cma-es', verbose=True):
        results = []
        valid_imgs = self.correct_imgs.img
        img_samples = np.random.choice(valid_imgs, samples)
        total_start_time = time.time()

        for pixel_count in pixels:
            for i, img in enumerate(img_samples):
                print(self.model.name, '- image', img, '-', i + 1, '/', len(img_samples))
                targets = [None] if not targeted else range(10)

                for target in targets:
                    if targeted and target == self.y_test[img, 0]:
                        continue
                    result = self.attack(img, target, pixel_count,
                                         maxiter=maxiter, popsize=popsize, algorithm=algorithm, verbose=verbose)
                    results.append(result)

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        avg_time_per_attack = total_time / len(results) if results else 0

        return results, total_time, avg_time_per_attack

if __name__ == '__main__':
    _, test = cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    model = ResNet(load_weights=True)

    attacker = PixelAttacker(model, test, class_names)

    for algorithm in ['cma-es']:
        print(f'Starting attack with {algorithm.upper()}')
        results, total_time, avg_time_per_attack = attacker.attack_all(
            
        )

        # Format results to the required columns
        formatted_results = []
        for result in results:
            model_name = result[0]
            pixel_count = result[1]
            img_id = result[2]
            true_class = result[3]
            predicted_class = result[4]
            success = result[5]
            cdiff = result[6]
            perturb_time = result[11]
            attack_time = result[11]

            formatted_result = [
                model_name,
                pixel_count,
                img_id,
                true_class,
                predicted_class,
                cdiff,
                perturb_time,
                attack_time,
                total_time,
                avg_time_per_attack
            ]
            formatted_results.append(formatted_result)

        # Convert to DataFrame for easy CSV saving
        columns = ['model', 'pixels', 'image', 'true', 'predicted', 'cdiff', 'perturb_time', 'attack_time', 'total_time', 'avg_time_per_attack']
        results_df = pd.DataFrame(formatted_results, columns=columns)

        # Save results to CSV
        csv_filename = f'results_{algorithm}.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f'Saved formatted results to {csv_filename}')
