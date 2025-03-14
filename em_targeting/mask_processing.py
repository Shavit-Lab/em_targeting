from skimage import measure, morphology
import numpy as np


def postprocess_mask_bell(mask, closing_disk=5, dilation_disk=5, min_size=5000, major_axis_length=200):
    mask_steps = []
    mask_proc = mask

    print("Closing")
    mask_proc = morphology.binary_closing(mask_proc, footprint=morphology.disk(closing_disk))
    mask_steps.append(mask_proc)

    print("Threshold major axis length")
    label = measure.label(mask_proc)
    props = measure.regionprops(label)
    mask_proc = np.zeros_like(label)
    for prop in props:
        if prop.major_axis_length > major_axis_length:
            mask_proc[label == prop.label] = prop.label
    mask_steps.append(mask_proc)

    print("Dilation")
    mask_proc = morphology.binary_dilation(mask_proc, footprint=morphology.disk(dilation_disk))
    mask_steps.append(mask_proc)

    mask_proc = measure.label(mask_proc)
    print("Removing small objects")
    mask_proc = morphology.remove_small_objects(mask_proc, min_size=min_size)
    mask_steps.append(mask_proc)

    mask_prop_binary = mask_proc > 0

    return mask_prop_binary, mask_steps


def postprocess_mask_tentacles(mask, closing_disk=5, dilation_disk=5, min_size=5000):
    mask_steps = []
    mask_proc = mask

    print("Closing")
    mask_proc = morphology.binary_closing(mask_proc, footprint=morphology.disk(closing_disk))
    mask_steps.append(mask_proc)

    print("Dilation")
    mask_proc = morphology.binary_dilation(mask_proc, footprint=morphology.disk(dilation_disk))
    mask_steps.append(mask_proc)

    mask_proc = measure.label(mask_proc)
    print("Removing small objects")
    mask_proc = morphology.remove_small_objects(mask_proc, min_size=min_size)
    mask_steps.append(mask_proc)

    lbl = measure.label(mask_proc)
    mask_prop_area = 0 * mask_proc
    for prop in measure.regionprops(lbl):
        mask_prop_area[lbl == prop.label] = prop.area

    mask_prop_binary = mask_proc > 0

    return mask_prop_binary, mask_steps
