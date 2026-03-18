from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

#rotates an array counter-clockwise about center
def rotate_array(array, angle):
    image = Image.fromarray(array, 'L')
    rotated_image = image.rotate(angle, expand=False, fillcolor='black')
    rotated_array = np.array(rotated_image)
    return rotated_array

#translates array given dx and dy
def translate_array(array, dx, dy):
    translated_array = np.empty_like(array)
    # asserts int arguments
    dx = round(dx)
    dy = round(dy)
    # converts dy from top-bottom into bottom-top
    dy = -dy
    num_rows = len(array)
    num_cols = len(array[0])
    for row in range(num_rows):
        for col in range(num_cols):
            if (row-dy>=0 and row-dy<len(array) and col-dx>=0 and col-dx<len(array[0])):
                translated_array[row][col] = array[row - dy][col - dx]
            else:
                translated_array[row][col] = 0
    return translated_array

# dilates array about center of image
# one method is to dilate entire thing from top left, translate image to the left, and crop
# first this would require making a dummy large array
# def dilate_array(array, scale):
#     dilated_array = np.empty_like(array)
#     num_rows = len(array)
#     num_cols = len(array[[0]])
#     for row in range(len(array)):
#         for col in range(len(array[0])):
#             if (row-dy>=0 and row-dy<len(array) and col-dx>=0 and col-dx<len(array[0])):
#                 translated_array[row][col] = array[row - dy][col - dx]
#             else:
#                 translated_array[row][col] = 0
#     return translated_array   

# scale up array starting from center, does not preserve bit density - output same size as input
def dilate_array(array, width_scale_factor, height_scale_factor):
    width = len(array[0])
    height = len(array)

    new_width = round(width_scale_factor * width)
    new_height = round(height_scale_factor * height)

    dilated_array = update_resolution(array, new_width, new_height)
    resized_array = resize_array(dilated_array, width, height)
    return resized_array

# scales entire shape up to new dimension, preserves bit density - output different size than input
def update_resolution(array, new_width, new_height):
    width = len(array[0])
    height = len(array)
    # asserts int arguments
    new_width = round(new_width)
    new_height = round(new_height)

    col_scale_factor = new_width / width
    row_scale_factor = new_height / height
    # print(col_scale_factor)
    new_array = np.zeros((new_height, new_width), dtype=np.uint8)

    for row in range(new_height):
        for col in range(new_width):
            # print(int(row / row_scale_factor))
            # print(int(col / col_scale_factor))
            # print(new_width)
            new_array[row][col] = array[int(row / row_scale_factor)][int(col / col_scale_factor)]

    return new_array

# crops array down to center (decreased size) or pads edges with zeros (increased size) - output different size than input
def resize_array(array, new_width, new_height):
    width = len(array[0])
    height = len(array)
    # asserts int arguments
    new_width = round(new_width)
    new_height = round(new_height)

    new_array = np.zeros((new_height, new_width), dtype=np.uint8)

    dcol = new_width - width # number of added cols (negative means cropped)
    drow = new_height - height # number of added rows (negative means cropped)
    irow = 0
    icol = 0
    if (drow < 0):
        irow = round(-drow / 2)
    if (dcol < 0):
        icol = round(-dcol / 2)

    for row in range(new_height):
        for col in range(new_width):
            if (row >= height or col >= width):
                new_array[row][col] = 0
            else:
                new_array[row][col] = array[row + irow][col + icol]
    translated_array = translate_array(new_array, round(dcol / 2) + icol, -round(drow / 2) - irow)
    return translated_array
            


# size = 10
# arr = np.zeros((size, size), dtype=np.uint8)
# for i in range(size):
#     arr[i, i] = 1
#     # arr[i, 5] = 1

# image = Image.fromarray(arr, 'L')

# # counter clockwise rotation
# angle = 60

# # expand parameter dictates cropping of array
# rotated_image = image.rotate(angle, expand=False, fillcolor='black')
# rotated_array = np.array(rotated_image)

# print(f"Original array shape: {arr.shape}")
# print(f"Rotated array shape: {rotated_array.shape}")

size = 8
arr = np.zeros((size, size), dtype=np.uint8)
for i in range(size):
    # arr[i, i] = 1
    arr[i, 3] = 1
    arr[i, 4] = 1
    # arr[i, 0] = 1
    # arr[i, 1] = 1

# plt.figure(1)
# plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
# # plt.imshow(translate_array(arr, 0, 1), cmap='gray', vmin=0, vmax=1)

# new = rotate_array(update_resolution(arr, 32, 32), 6)
# plt.figure(2)
# plt.imshow(new, cmap='gray', vmin=0, vmax=1)
# new = translate_array(arr, 3, 1)
# plt.figure(3)
# plt.imshow(new, cmap='gray', vmin=0, vmax=1)
# new = dilate_array(arr, 2, 2)
# plt.figure(4)
# plt.imshow(new, cmap='gray', vmin=0, vmax=1)
# new = update_resolution(arr, 16, 16)
# plt.figure(5)
# plt.imshow(new, cmap='gray', vmin=0, vmax=1)
# new = resize_array(arr, 16, 16)
# plt.figure(6)
# plt.imshow(new, cmap='gray', vmin=0, vmax=1)
# plt.show()

fig, axs = plt.subplots(2, 3, figsize=(10, 5))

axs[0][0].imshow(arr, cmap='gray', vmin=0, vmax=1)
axs[0][1].imshow(rotate_array(update_resolution(arr, 32, 32), 6), cmap='gray', vmin=0, vmax=1)
axs[0][2].imshow(translate_array(arr, 3, 1), cmap='gray', vmin=0, vmax=1)
axs[1][0].imshow(dilate_array(arr, 2, 2), cmap='gray', vmin=0, vmax=1)
axs[1][1].imshow(update_resolution(arr, 16, 16), cmap='gray', vmin=0, vmax=1)
axs[1][2].imshow(resize_array(arr, 16, 16), cmap='gray', vmin=0, vmax=1)
plt.show()