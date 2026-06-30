import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import Any, Optional, Tuple, Type, Dict, List
import cv2
from sam_utils import (
   MaskData,
   area_from_rle,
   batch_iterator,
   batched_mask_to_box,
   box_xyxy_to_xywh,
   build_all_layer_point_grids,
   calculate_stability_score,
   coco_encode_rle,
   generate_crop_boxes,
   is_box_near_crop_edge,
   mask_to_rle_pytorch,
   remove_small_regions,
   rle_to_mask,
   uncrop_boxes_xyxy,
   uncrop_masks,
   uncrop_points,
)


# this is for showing masked img from automask
def show_anns(anns):
   if len(anns) == 0:
       return
   sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
   ax = plt.gca()
   ax.set_autoscale_on(False)


   img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
   img[:,:,3] = 0
   for ann in sorted_anns:
       m = ann['segmentation']
       # color_mask = np.concatenate([np.random.random(3), [0.35]])
       color_mask = np.concatenate([np.random.random(3), [1]])
       img[m] = color_mask
   ax.imshow(img)


# this is for showing masked img from labeld ids mask
def show_mask(mask_with_ids):
   unique_ids = np.unique(mask_with_ids)
   num_ids = len(unique_ids)
   colored_mask = np.zeros((mask_with_ids.shape[0], mask_with_ids.shape[1], 3))


   for i, unique_id in enumerate(unique_ids):
       if unique_id == 0:
           continue
       colored_mask[mask_with_ids == unique_id] = np.random.random(3)
   plt.imshow(colored_mask)


# this is for labeling ids on mask and removing overlapped mask
def labeled_mask_2(anns):
   if len(anns) == 0:
       return
  
   sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
   combined_mask = np.zeros_like(sorted_anns[0]['segmentation'], dtype=np.int32)
   already_masked = np.zeros_like(sorted_anns[0]['segmentation'], dtype=bool)
   boolean_mask = np.zeros_like(sorted_anns[0]['segmentation'], dtype=bool)
   count = 0


   for ann in sorted_anns:
       m = ann['segmentation']
       non_overlapping = m & ~already_masked 
      
       if non_overlapping.sum() > 0:
           combined_mask[non_overlapping] = count + 1
           boolean_mask[non_overlapping] = True
           already_masked[non_overlapping] = True 
           count += 1



   return combined_mask, boolean_mask


def labeled_mask(anns, min_area=800, min_non_overlap_area=800):
   """Label masks after filtering small regions.

   This keeps the original overlap-aware strategy in labeled_mask(), but removes
   tiny masks and tiny remaining fragments that often correspond to repeated
   details such as building windows.
   """
   if len(anns) == 0:
       return
  
   sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
   combined_mask = np.zeros_like(sorted_anns[0]['segmentation'], dtype=np.int32)
   already_masked = np.zeros_like(sorted_anns[0]['segmentation'], dtype=bool)
   boolean_mask = np.zeros_like(sorted_anns[0]['segmentation'], dtype=bool)
   count = 0


   for ann in sorted_anns:
       if ann['area'] < min_area:
           continue


       m = ann['segmentation']
       non_overlapping = m & ~already_masked
       non_overlap_area = non_overlapping.sum()
      
       if non_overlap_area >= min_non_overlap_area:
           combined_mask[non_overlapping] = count + 1
           boolean_mask[non_overlapping] = True
           already_masked[non_overlapping] = True 
           count += 1


   return combined_mask, boolean_mask


def labeled_mask_auto(anns):
   if len(anns) == 0:
       return
  
   sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
   combined_mask = np.zeros_like(sorted_anns[0]['segmentation'], dtype=np.int32)
   # already_masked = np.zeros_like(sorted_anns[0]['segmentation'], dtype=bool)
   boolean_mask = np.zeros_like(sorted_anns[0]['segmentation'], dtype=bool)
   count = 0


   for ann in sorted_anns:
       m = ann['segmentation']
       # non_overlapping = m & ~already_masked 
      
       # if non_overlapping.sum() > 0:
       combined_mask[m] = count + 1
       boolean_mask[m] = True
       # already_masked[non_overlapping] = True 
       count += 1


   # scaling from 0 to 1
   # max_id = combined_mask.max()
   # scaler = lambda x: x / max_id if max_id != 0 else 0
   # combined_mask = scaler(combined_mask)


   return combined_mask, boolean_mask


# generate superpixel
def generate_superpixel(  
                       rgb_img,
                       num_superpixels=1000,
                       num_iterations=5,
                       prior=2,
                       num_levels=4,
                       num_histogram_bins=5
                   ):
  
   height, width, channels = rgb_img.shape
   # superpixel seeds
   seeds = cv2.ximgproc.createSuperpixelSEEDS(
                                               width,
                                               height,
                                               channels,
                                               num_superpixels,
                                               num_levels,
                                               prior,
                                               num_histogram_bins
                                           )
                                              
   seeds.iterate(rgb_img, num_iterations)
   num_of_superpixels_result = seeds.getNumberOfSuperpixels()
   labels = seeds.getLabels()
   super_pixel_line = seeds.getLabelContourMask(False)
   contour_img_rgb = rgb_img.copy()
   contour_img_rgb[super_pixel_line == 255] = [255, 0, 0]
   return labels, contour_img_rgb, num_of_superpixels_result


# calc central point of the superpixel
def calculate_superpixel_centers(labels):
   superpixel_centers = {}
   points = []
   unique_labels = np.unique(labels)
   for label in unique_labels:
       mask = (labels == label)
       y_coords, x_coords = np.where(mask)
       center_y = np.median(y_coords).astype(int)
       center_x = np.median(x_coords).astype(int)
       superpixel_centers[label] = (center_y, center_x)
       points.append((center_x, center_y))
   return points


# superpixel automask
class SuperpixelSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):
   def __init__(self, **kwargs):
       super().__init__(**kwargs)
   def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
      
       # rgb_img = get_img_array(image, open_or_pil='open'
       # Generate superpixels and calculate their centers
       labels, contour_img_rgb, num_of_superpixels_result = generate_superpixel(image)
       superpixel_centers = calculate_superpixel_centers(labels)
       # point grid
       self.point_grids = [np.array(superpixel_centers) / np.array(image.shape[:2][::-1])]
       # generate masks
       mask_data = self._generate_masks(image)
       # filtering
       if self.min_mask_region_area > 0:
           mask_data = self.postprocess_small_regions(
               mask_data,
               self.min_mask_region_area,
               max(self.box_nms_thresh, self.crop_nms_thresh),
           )
          
       # Encode masks
       if self.output_mode == "coco_rle":
           mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
       if self.output_mode == "binary_mask":
           mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
       else:
           mask_data["segmentations"] = mask_data["rles"]
       # Write mask records
       curr_anns = []
       for idx in range(len(mask_data["segmentations"])):
           segmentation = mask_data["segmentations"][idx]
           # non_overlapping = segmentation & ~already_maske
           # if non_overlapping.sum() >= 10000:
          
               # already_masked[segmentation] = True
           ann = {
               "segmentation": segmentation,
               "area": area_from_rle(mask_data["rles"][idx]),
               "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
               "predicted_iou": mask_data["iou_preds"][idx].item(),
               "point_coords": [mask_data["points"][idx].tolist()],
               "stability_score": mask_data["stability_score"][idx].item(),
               "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
               "mask_label": idx + 1
           }
           curr_anns.append(ann)
           # sorted_anns = sorted(curr_anns, key=(lambda x: x['area']), reverse=True
       return curr_anns



