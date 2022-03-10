from PIL import *
from PIL import Image
import numpy as np

def main():
    img = readPILimg()
    arr = PIL2np(img)
    print(arr)

    my_filter = np.array([[-1, 0, 1 ], [-1, 0, 1 ], [-1, 0, 1]])
    im_out = convolve(arr,my_filter) # 3. foto çıkıyor arrayı burası.
    print("Photo 4:")
    print(im_out)

    im_out = threshold(im_out, 60, 0, 100)
    new_img = np2PIL(im_out)
    new_img.show()


def readPILimg():
    # img dosyasının konumunu belirler.
    img = Image.open("uncompressed_cat2.png")
    # dosyanın içeriğini gösterir.
    img.show()
    # look at the function.
    img_gray = color2gray(img)
    print("Photo 2:")
    img_gray.show()
    img_gray.save("newImg",'png')
    #Return a resized copy of this img.
    new_img = img.resize((256,256))
    new_img.show()
    return img_gray

#converte the file
def color2gray(img):
    #returns converted copy of this image.For the "P" mode,
    #this method translates pixels through the palette.
    img_gray = img.convert('L')
    return img_gray

#arraya kaydetme
def PIL2np(img):
    #img.size yi dizi gibi düşün ilk elemanı[0] rows,
    # ikinci elemanı [1] column
    nrows = img.size[0]
    ncols = img.size[1]
    print("nrows, ncols : ", nrows, ncols)
    #array oluşturup değerleri içeri alır 2 D
    imgarray = np.array(img.convert("L"))
    return imgarray

def np2PIL(im):
    print("size of arr: ",im.shape)
    img = Image.fromarray(np.uint8(im))
    return img

def convolve(im,filter):
    (nrows, ncols) = im.shape
    (k1,k2) = filter.shape
    k1h = int((k1 -1) / 2)
    k2h = int((k2 -1) / 2)
    #np.zeros() Return a new array of given shape and type, filled with zeros.
    im_out = np.zeros(shape = im.shape)
    print("image size , filter size ", nrows, ncols, k1, k2)
    for i in range(1, nrows - 1):
        for j in range(1, ncols - 1):
            sum = 0.
            for l in range(-k1h, k1h + 1):
                for m in range(-k2h, k2h + 1 ):
                    sum += im[i - l][j - m] * filter[l][m]
            im_out[i][j] = sum
    return im_out

def threshold(im,T, LOW, HIGH):
    (nrows, ncols) = im.shape
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) <  T :
                im[i][j] = LOW
            else:
                im[i][j] = HIGH
    return im



if __name__=='__main__':
    main()
