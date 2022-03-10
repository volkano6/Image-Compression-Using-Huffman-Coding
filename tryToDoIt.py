import numpy
from PIL import *
from PIL import Image
import numpy as np
import os
import Huffman
import math

def main():
    img = readPILimg()
    arr = PIL2np(img)
    print("---------")
    print(arr)
    file = open("input.txt", "r+")
    writeMatrixToFile(file, img, arr)

    # print("compressionLevel1 :")
    print("---------")
    # compressionLevel1("input.txt")
    gray_level_list = readFileToList("input.txt")

    array = ListToNpArray(gray_level_list)
    print(calculate_probability(array))
    print(calculate_entropy(calculate_probability(array)))
    # my_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # im_out = convolve(arr, my_filter)  # 3. foto çıkıyor arrayı burası.
    # print("Photo 4:")
    # print(im_out)
    #
    # im_out = threshold(im_out, 60, 0, 100)
    # new_img = np2PIL(im_out)
    # new_img.show()
    #


def compressionLevel1(file):
    file_data = np.loadtxt(file, dtype=int)
    print(file_data)

    metin = ""
    for val in range(0, 240):
        for val2 in range(0, 384):
            metin = metin + ' ' + str(file_data[val][val2])

    print(metin)
    Huffman.Huffman_Encoding(metin)


def ListToNpArray(list):
    arr = numpy.array(list)
    return arr


def readFileToList(file):
    # opening the file in read mode
    my_file = open(file, "r")
    # reading the file
    data = my_file.read()
    # replacing end splitting the text
    # when newline ('\n') is seen.
    data_into_list = data.split(" ")
    del data_into_list[0]
    data_into_list = list(map(int, data_into_list))
    my_file.close()
    return data_into_list


def calculate_probability(arr):
    denominator = len(arr)
    unique, counts = numpy.unique(arr, return_counts=True)
    countOfVal = dict(zip(unique, counts))
    probArr = []

    for ind in range(0, len(countOfVal)):
        probArr.append(countOfVal.get(ind) / denominator)

    return probArr

def calculate_entropy(arr):
    entropyVal = 0
    for ind in range(0, len(arr)):
        entropyVal += arr[ind] * math.log(arr[ind])
    return entropyVal * (-1)

def readFileTo2DArray(file):
    file_data = np.loadtxt(file, dtype=int)
    return file_data


def writeMatrixToFile(file, img, arr):
    # input dosyasına data yazdırır.
    for ind in range(0, img.size[1]):
        if ind != 0:
            file.write("\n")
        for ind2 in range(0, img.size[0]):
            file.write(' ' + str(arr[ind][ind2]))


def readFile(fileName):
    fileObj = open("input.txt", "r")  # opens the file in read mode
    words = fileObj.read().splitlines()  # puts the file into an array
    fileObj.close()
    return words


def readPILimg():
    # img dosyasının konumunu belirler.
    img = Image.open("uncompressed_cat2.png")
    # dosyanın içeriğini gösterir.
    img.show()
    # look at the function.
    img_gray = color2gray(img)
    print("Photo 2:")
    img_gray.show()
    img_gray.save("newImg", 'png')
    # Return a resized copy of this img.
    # new_img = img.resize((256,256))
    # new_img.show()
    return img_gray


# converte the file
def color2gray(img):
    # returns converted copy of this image.For the "P" mode,
    # this method translates pixels through the palette.
    img_gray = img.convert('L')
    return img_gray


# arraya kaydetme
def PIL2np(img):
    # img.size yi dizi gibi düşün ilk elemanı[0] rows,
    # ikinci elemanı [1] column
    nrows = img.size[0]
    ncols = img.size[1]
    print("nrows, ncols : ", nrows, ncols)
    # array oluşturup değerleri içeri alır 2 D
    imgarray = np.array(img.convert("L"))
    return imgarray


def np2PIL(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(np.uint8(im))
    return img


def convolve(im, filter):
    (nrows, ncols) = im.shape
    (k1, k2) = filter.shape
    k1h = int((k1 - 1) / 2)
    k2h = int((k2 - 1) / 2)
    # np.zeros() Return a new array of given shape and type, filled with zeros.
    im_out = np.zeros(shape=im.shape)
    print("image size , filter size ", nrows, ncols, k1, k2)
    for i in range(1, nrows - 1):
        for j in range(1, ncols - 1):
            sum = 0.
            for l in range(-k1h, k1h + 1):
                for m in range(-k2h, k2h + 1):
                    sum += im[i - l][j - m] * filter[l][m]
            im_out[i][j] = sum
    return im_out


def threshold(im, T, LOW, HIGH):
    (nrows, ncols) = im.shape
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) < T:
                im[i][j] = LOW
            else:
                im[i][j] = HIGH
    return im


if __name__ == '__main__':
    main()
