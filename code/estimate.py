import numpy as np

# Estimate depth using focal length and radius of the circle
# def depth_estimate(area, focal_length, radius):
#     depth = (focal_length * radius) / np.sqrt(area)
#     return depth


def depth_estimate(area, area_ref, depth_ref):
    depth_estimate = depth_ref * np.sqrt(area_ref) / np.sqrt(area)
    return depth_estimate


def depth_estimate2(radius, radius_ref, depth_ref):
    depth_estimate = depth_ref * radius_ref / radius
    return depth_estimate
