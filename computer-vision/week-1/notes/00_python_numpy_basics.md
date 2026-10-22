# 00 · Just enough Python & NumPy for Week 1

You don't need to be a programmer. The lectures use **three libraries**, and you only need a small piece of each.

| Library | Imported as | Job in plain words |
|---------|-------------|--------------------|
| OpenCV | `import cv2` | The image toolbox: load, convert colours, rotate, blur, find edges… |
| NumPy | `import numpy as np` | Stores grids of numbers (an image *is* one) and does maths on them |
| Matplotlib | `import matplotlib.pyplot as plt` | Draws/shows pictures and charts on screen |

---

## 1. A NumPy array = a grid of numbers

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.shape)   # (2, 3)  -> 2 rows, 3 columns
print(a[0, 2])   # 3       -> row 0, column 2  (counting starts at 0!)
print(a[1])      # [4 5 6] -> the whole of row 1
```

- **`.shape`** tells you the size: `(rows, columns)`, or `(rows, columns, channels)` for colour images.
- **Counting starts at 0.** The first row is row 0.
- **`a[row, col]`**: the row always comes first.

## 2. Slicing = cutting out a piece

`start:stop` means "from start up to **but not including** stop".

```python
img[100:200, 50:300]   # rows 100..199, columns 50..299  -> a crop
img[:, 0]              # ":" alone means "all" -> every row, column 0
img[0:60, 0:60] = 255  # set a whole block at once (paint it white)
```

## 3. Data types: `uint8` and `float64`

- Images are usually **`uint8`**: whole numbers from **0 to 255** only.
- Going past 255 **wraps around** in NumPy: `250 + 10 = 4` (!). OpenCV's `cv2.add` **clips** instead: `250 + 10 = 255`.
- Maths that can go negative (edge detection) needs **`float64`** (`cv2.CV_64F` in OpenCV).
- `np.float32([...])`: OpenCV's transformation functions want their matrices as 32-bit decimals. That's why the lectures write `np.float32([[1,0,100],[0,1,50]])`.

## 4. Useful NumPy bits seen in the lectures

```python
np.zeros((3, 3))          # 3x3 grid of 0s
np.ones((3, 3)) / 9       # 3x3 grid of 1/9  (an "average" kernel)
np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])  # the sharpening kernel
a * b                     # multiply element by element (same-size grids)
np.sum(a)                 # add everything up
```

## 5. Showing an image with Matplotlib

```python
plt.imshow(img_rgb)            # colour image (must be RGB order!)
plt.imshow(gray, cmap='gray')  # 1-channel image: tell it to use grey shades
plt.title("My title")
plt.axis('off')                # hide the x/y ruler numbers
plt.show()                     # actually display it

plt.subplot(1, 2, 1)           # "1 row, 2 columns, draw in slot 1"
plt.subplot(1, 2, 2)           # "...now slot 2"  -> side-by-side pictures
```

> Without `cmap='gray'`, a 1-channel image is drawn in yellow-green-purple colours (matplotlib's default colour map). It's the same data, just painted with fake colours.

## 6. Python bits used in the mini-projects

```python
def image_viewer(img_path, mode="original"):   # a function with a default value
    if mode == "gray":
        ...
    elif mode == "edges":       # "else if"
        ...

f"Mode: {mode}"                 # f-string: puts the variable's value into the text

for i, cnt in enumerate(contours):   # loop that also gives the counter i = 0, 1, 2...
    ...

filters = {"blur": ..., "sharpen": ...}   # dictionary: name -> value
for name, output in filters.items():      # loop over (name, value) pairs
    ...

x = a if condition else b                 # one-line if/else
```

That's all the Python you need for Week 1. Next: `day1_images_and_colors.md`.
