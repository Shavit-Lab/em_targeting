from skimage import measure, morphology
import numpy as np


def postprocess_mask(mask):
    label = measure.label(mask)
    props = measure.regionprops(label)

    mask_proc = np.zeros_like(label)

    mask_proc = label

    for prop in props:
        if prop.major_axis_length > 80:
            mask_proc[label == prop.label] = prop.label

    print("Removing small objects")
    mask_proc = morphology.remove_small_objects(mask_proc, min_size=1000)
    print("Closing")
    mask_proc = morphology.binary_closing(mask_proc, footprint=morphology.disk(5))
    print("Dilation")
    mask_proc = morphology.binary_dilation(mask_proc, footprint=morphology.disk(5))

    mask_proc = measure.label(mask_proc)
    print("Removing small objects")
    mask_proc = morphology.remove_small_objects(mask_proc, min_size=50000)

    mask_prop_binary = mask_proc > 0
    return mask_prop_binary
