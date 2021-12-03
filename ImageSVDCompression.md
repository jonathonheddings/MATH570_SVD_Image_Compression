# Using SVD To Compress Images

In this Python Notebook, I wrote a short python program for compressing images by cutting out singular values from the Singular Value Decomposition matrix, and converting that back into an image. Below is the function that compresses a single channel of an image given a specified output rank.


```python
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
     
# Open a grayscale bmp file
img = Image.open(r'C:\Users\jonat\Documents\Code\FruitGrayscale.bmp')

# This function takes an image and decomposes the bitmap using SVD, then each composite matrix
#    is sliced by the given output matrix rank and multiplied together to create a compressed image
def compress_image(img, rank):
    
    # Apply SVD
    U, S, Vt = np.linalg.svd(img)
    
    # SLice matrices and create compressed image
    c_img = np.dot(U[:,:rank], np.dot(np.diag(S[:rank]), Vt[:rank, :]))
    
    return c_img
print(np.asarray(img))
print(compress_image(img, 600))

```

    [[  0   0   0 ...  65  65  65]
     [ 66  66   0 ...  65  65  65]
     [ 66  66  66 ...  65  65  65]
     ...
     [189 189 189 ...  67  67  66]
     [189 189 189 ...  66  66  67]
     [188 188 188 ...  67  67  67]]
    [[-5.42088076e-14  1.13843397e-13  2.48750497e-13 ...  6.50000000e+01
       6.50000000e+01  6.50000000e+01]
     [ 6.60000000e+01  6.60000000e+01 -7.12456561e-12 ...  6.50000000e+01
       6.50000000e+01  6.50000000e+01]
     [ 6.60000000e+01  6.60000000e+01  6.60000000e+01 ...  6.50000000e+01
       6.50000000e+01  6.50000000e+01]
     ...
     [ 1.89000000e+02  1.89000000e+02  1.89000000e+02 ...  6.70000000e+01
       6.70000000e+01  6.60000000e+01]
     [ 1.89000000e+02  1.89000000e+02  1.89000000e+02 ...  6.60000000e+01
       6.60000000e+01  6.70000000e+01]
     [ 1.88000000e+02  1.88000000e+02  1.88000000e+02 ...  6.70000000e+01
       6.70000000e+01  6.70000000e+01]]
    

These are just outputted arrrays showing the head and tail of the image bitmap and the compressed image bitmap.

---


Now we use the function to compress two black and white images. One 900x600 pixel image of a pomegranate, and one 225x225 image of a cat.


```python
## Initialize Matplotlib plots for displaying the images
fig2, ax2 = plt.subplots(1,2,figsize=(15,5))
fig, ax = plt.subplots(1,4,figsize=(16,7))

# Init a list of output image ranks to iterate through
rank = [1,5,10,250]

# Loop through the list of ranks and create and graph an SVD compressed image
for i, rk in enumerate(rank):
    c_img = compress_image(img, rk)
    ax[i].imshow(c_img, cmap='gray')
    ax[i].set_axis_off()
    title = 'rank = ' + str(rk)
    ax[i].set_title(title)
    

# Plot the Singular Values of the original image
U, S, Vt = np.linalg.svd(img, full_matrices=True)
ax2[0].plot(S)
ax2[0].set_title('Singular Values')
ax2[1].imshow(img)
ax2[1].set_axis_off()
ax2[1].set_title('Original Grayscale Image')
```




    Text(0.5, 1.0, 'Original Grayscale Image')




    
![png](output_4_1.png)
    



    
![png](output_4_2.png)
    


Now the other image


```python
def invert_image(c_img):
    for i, row in enumerate(c_img):
        for j, pixel in enumerate(row):
            c_img[i][j] = 255 - c_img[i][j]   
    return c_img

img = Image.open(r'C:\Users\jonat\Documents\Code\cat.bmp')
fig2, ax2 = plt.subplots(1,2,figsize=(15,5))
fig, ax = plt.subplots(1,4,figsize=(16,7))

#   For some reason this image needed to be inverted again
rank = [1,5,10,200]
for i, rk in enumerate(rank):
    c_img = invert_image(compress_image(img, rk))
    ax[i].imshow(c_img, cmap='gray')
    ax[i].set_axis_off()
    title = 'rank = ' + str(rk)
    ax[i].set_title(title)
    

# Plot the Singular Values of the original image
U, S, Vt = np.linalg.svd(img, full_matrices=True)
ax2[0].plot(S), ax2[0].set_title('Singular Values'), ax2[1].imshow(img)
ax2[1].set_axis_off(), ax2[1].set_title('Original Grayscale Image')
```




    (None, Text(0.5, 1.0, 'Original Grayscale Image'))




    
![png](output_6_1.png)
    



    
![png](output_6_2.png)
    


## Compressing Colored Images
To compress colored images the only difference is that each channel needs to get the SVD treatement seperately and then stitched back together. 


```python
img = np.asarray(Image.open(r'C:\Users\jonat\Documents\Code\colorveggies.bmp'))
red, green, blue = img[:,:,0], img[:,:,1], img[:,:,2]


fig, ax = plt.subplots(1, 4, figsize=(12, 8))
ax[0].imshow(img), ax[0].set_title('Original Image'),
ax[1].imshow(red, cmap='Reds'), ax[1].set_title('Red'),
ax[2].imshow(green, cmap='Greens'), ax[2].set_title('Green'),
ax[3].imshow(blue, cmap='Blues'), ax[3].set_title('Blue')

for i in range(4):
    ax[i].set_axis_off()
img1 = np.zeros_like(img)
rank = [30,5,1]
for i, rk in enumerate(rank):
    red1, green1, blue1 = (compress_image(red, rk)), (compress_image(green, rk)), (compress_image(blue, rk))
    fig1, ax1 = plt.subplots(1, 4, figsize=(12, 8))
    for i in range(4):
        ax1[i].set_axis_off()
    img1[:,:,0], img1[:,:,1], img1[:,:,2] = red1, green1, blue1
    title = 'Rank = ' + str(rk)
    ax1[0].imshow(img1), ax1[0].set_title(title), ax1[1].imshow(red1, cmap='Reds'), ax1[1].set_title(title),
    ax1[2].imshow(green1, cmap='Greens'), ax1[2].set_title(title), ax1[3].imshow(blue1, cmap='Blues'), ax1[3].set_title(title)
    
```


    
![png](output_8_0.png)
    



    
![png](output_8_1.png)
    



    
![png](output_8_2.png)
    



    
![png](output_8_3.png)
    

