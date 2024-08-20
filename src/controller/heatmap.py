from .base import BaseController
from kb_ops.predict import PredictOutput
import PIL
from PIL.Image import Image
from model.concept import Concept

class ControllerHeatmapMixin(BaseController):
    ######################
    # Heatmap Comparison #
    ######################
    def heatmap_image_comparison(self, image1: Image, image2: Image):
        '''
            Implements: "What are the differences between these two images"
        '''
        # Choose the highest ranking concepts for visualization regardless of whether they're unknown
        concept1_pred: PredictOutput = self.predict_hierarchical(image1)['prediction_path'][-1]
        concept2_pred: PredictOutput = self.predict_hierarchical(image2)['prediction_path'][-1]

        concept1 = self.retrieve_concept(concept1_pred.predicted_label)
        concept2 = self.retrieve_concept(concept2_pred.predicted_label)

        if concept1.name == concept2.name:
            image1_heatmap = self.heatmap(image1, concept1.name)
            image2_heatmap = self.heatmap(image2, concept1.name)

            c1_minus_c2_image1 = c2_minus_c1_image1 = image1_heatmap
            c1_minus_c2_image2 = c2_minus_c1_image2 = image2_heatmap

        else:
            c1_minus_c2_image1, c2_minus_c1_image1 = self.heatmap_visualizer.get_difference_heatmap_visualizations(concept1, concept2, image1)
            c1_minus_c2_image2, c2_minus_c1_image2 = self.heatmap_visualizer.get_difference_heatmap_visualizations(concept1, concept2, image2)

        return {
            'concept1_prediction': concept1_pred, # PredictOutput
            'concept2_prediction': concept2_pred, # PredictOutput
            'concept1_minus_concept2_on_image1': c1_minus_c2_image1, # PIL.Image.Image
            'concept2_minus_concept1_on_image1': c2_minus_c1_image1, # PIL.Image.Image
            'concept1_minus_concept2_on_image2': c1_minus_c2_image2, # PIL.Image.Image
            'concept2_minus_concept1_on_image2': c2_minus_c1_image2  # PIL.Image.Image
        }

    def heatmap(
        self, image: Image,
        concept_name: str,
        only_score_increasing_regions: bool = False,
        only_score_decreasing_regions: bool = False,
        return_detection_score: bool = False
    ) -> Image:
        '''
            If only_score_increasing_regions is True, implements:
                "Why is this a <class x>"

            If only_score_decreasing_regions is True, implements:
                "What are the differences between this and <class x>"

            If neither is true, shows the full heatmap (both increasing and decreasing regions).

            If return_detection_score is True, returns the detection score along with the heatmap.
        '''
        if only_score_increasing_regions and only_score_decreasing_regions:
            raise ValueError('At most one of only_score_increasing_regions and only_score_decreasing_regions can be True.')

        if return_detection_score and (only_score_increasing_regions or only_score_decreasing_regions):
            raise NotImplementedError('return_detection_score is not yet implemented for only_score_increasing_regions or only_score_decreasing_regions.')

        concept = self.retrieve_concept(concept_name)

        if only_score_increasing_regions:
            heatmap = self.heatmap_visualizer.get_positive_heatmap_visualization(concept, image)
        elif only_score_decreasing_regions:
            heatmap = self.heatmap_visualizer.get_negative_heatmap_visualization(concept, image)
        else: # Visualize all regions
            heatmap = self.heatmap_visualizer.get_heatmap_visualization(concept, image, return_detection_score=return_detection_score)

            if return_detection_score:
                heatmap, detection_score = heatmap
                return heatmap, detection_score

        return heatmap

    def heatmap_class_difference(self, concept1_name: str, concept2_name: str, image: Image = None):
        '''
            If image is provided, implements:
                "Why is this <class x> and not <class y>"

            Otherwise, implements:
                "What is the difference between <class x> and <class y>"
        '''
        concept1 = self.retrieve_concept(concept1_name)
        concept2 = self.retrieve_concept(concept2_name)

        if image is None:
            def load_positive_image(concept: Concept) -> Image:
                for example in concept.examples:
                    if not example.is_negative:
                        return PIL.Image.open(example.image_path)

            concept1_image = load_positive_image(concept1)
            concept2_image = load_positive_image(concept2)

            concept1_minus_concept2_image1, concept2_minus_concept1_image1 = self.heatmap_visualizer.get_difference_heatmap_visualizations(concept1, concept2, concept1_image)
            concept1_minus_concept2_image2, concept2_minus_concept1_image2 = self.heatmap_visualizer.get_difference_heatmap_visualizations(concept1, concept2, concept2_image)

            return {
                'concept1_minus_concept2_on_concept1_image': concept1_minus_concept2_image1, # PIL.Image.Image
                'concept2_minus_concept1_on_concept1_image': concept2_minus_concept1_image1, # PIL.Image.Image
                'concept1_minus_concept2_on_concept2_image': concept1_minus_concept2_image2, # PIL.Image.Image
                'concept2_minus_concept1_on_concept2_image': concept2_minus_concept1_image2  # PIL.Image.Image
            }

        else:
            concept1_minus_concept2, concept2_minus_concept1 = self.heatmap_visualizer.get_difference_heatmap_visualizations(concept1, concept2, image)

            return {
                'concept1_minus_concept2': concept1_minus_concept2, # PIL.Image.Image
                'concept2_minus_concept1': concept2_minus_concept1  # PIL.Image.Image
            }

    def heatmap_class_intersection(self, concept1_name: str, concept2_name: str, image: Image = None):
        '''
            If image is not provided, implements:
                "What is in common between <class x> and <class y> on this image?"

            Not yet implemented: If image is provided, implements:
                "What is in common between <class x> and <class y>?"
        '''
        concept1 = self.retrieve_concept(concept1_name)
        concept2 = self.retrieve_concept(concept2_name)

        if image is None:
            # TODO concatenate two images from concepts side by side as image
            raise NotImplementedError('This functionality is not yet implemented.')

        heatmap = self.heatmap_visualizer.get_intersection_heatmap_visualization(concept1, concept2, image)

        return heatmap