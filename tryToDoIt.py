import fileinput
from pathlib import Path
from huffman2 import HuffmanCoding
import sys
import numpy
from PIL import *
from PIL import Image
import numpy as np
import os
from numpy import ndarray
import Huffman
import math
import cv2


def main():

    #Level1("test.txt")
    #Level2("muhi.png")
    Level3("muhi.png")
    # gray_level_list = readFileToList("input.txt")
    #
    # array = ListToNpArray(gray_level_list)
    #
    # my_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # im_out = convolve(arr, my_filter)  # 3. foto çıkıyor arrayı burası.
    # print("Photo 4:")
    # print(im_out)
    #
    # im_out = threshold(im_out, 60, 0, 100)
    # new_img = np2PIL(im_out)
    # new_img.show()


def Level1(file):

    path = file
    h = HuffmanCoding(path)

    with open(file, "r+") as f:
        data = f.read()
    encoding, tree = Huffman.Huffman_Encoding(data)
    print("Encoded output", encoding)
    with open("Level1_test.txt", "w+") as f:
        f.write(encoding)

    output_path = h.compress()
    print("Compressed file path: " + output_path)

    #text to str.
    bin_file = open("Level1_test.txt", "r")
    encode_str = bin_file.read()
    bin_file.close()

    decom_path = h.decompress(output_path)
    print("Decompressed file path: " + decom_path)
    print("Decoded Output", Huffman.Huffman_Decoding(encode_str, tree))


def Level2(input_image):

    img = readPILimg(input_image)
    arr = PIL2np(img)

    # Dosya açar ve içine veri aktarır.
    with open("Level2_grayLevelImage.txt", "w+") as file:
        pass
    with open("Level2_grayLevelImage.txt", "r+") as file:
        writeMatrixToFile(file, arr)

    print("--------------------")
    encoding, tree = Huffman.Huffman_Encoding(readFileToList("Level2_grayLevelImage.txt"))

    print("Encoded output", encoding)
    with open("Level2_Image_Encode.txt", "w+") as f:
        f.write(encoding)

    #gray_level_2DArray = readFileTo2DArray("Level2_Image_Encode.txt")

    # huffman_encoding = Huffman.Calculate_Codes(tree)
    # huffman_BinaryEncoding = [len(huffman_encoding)][len(huffman_encoding[0])]
    #
    # for ind in range(len(gray_level_2DArray)):
    #     for ind2 in range(len(gray_level_2DArray[0])):
    #         huffman_BinaryEncoding[ind][ind2] = huffman_encoding.values(1)
    # print(huffman_encoding)


    path = "Level2_grayLevelImage.txt"
    h = HuffmanCoding(path)

    output_path = h.compress()
    print("Compressed file path: " + output_path)

    # text to str.
    bin_file = open("Level2_Image_Encode.txt", "r")
    encode_str = bin_file.read()
    bin_file.close()

    decom_path = h.decompress(output_path)
    print("Decompressed file path: " + decom_path)
    print("Decoded Output", Huffman.Huffman_Decoding(encode_str, tree))

    decompressed_arr= readFileTo2DArray("Level2_grayLevelImage_decompressed.txt")

    convertMatrixToImage(decompressed_arr, "Level2_restoredImage.png")




def compressionLevel2(file, img):
    nrows = img.size[0]
    ncols = img.size[1]

    encoding, tree = Huffman.Huffman_Encoding(readFileToList(file))
    print("Encoded output", encoding)

    # saves binary code in new file.


    # saves binary codes ftrom new file to huffman decodding.
    compressedFile = open("compressed_file.txt", "r+")
    newEncoding = compressedFile.read()
    print("Decoded Output", Huffman.Huffman_Decoding(newEncoding, tree))

    # yeni string decodeyi iki boyutlu araya atar.
    newImgArr = strToArr2DWithSpace(Huffman.Huffman_Decoding(newEncoding, tree), nrows, ncols)
    print(newImgArr)
    # convertMatrixToImage(newImgArr)
    # stringToArray2DForImg(14, 12, newImgArr)
    convertMatrixToImage(newImgArr, "restoredImage.png")
    # -----------------------------------------------Level 3

def Level3(img):

    level3Img = readPILimg(img)
    level3ImgNpArray = PIL2np(level3Img)
    print(level3ImgNpArray)
    level3ImgArray = Array2DToList2D(level3ImgNpArray)
    diff_arr = difference(level3ImgArray)
    print("difference pixels :", difference(level3ImgArray))
    #
    # with open("Level3_diff.txt", "w+") as diff_file:
    #     writeMatrixToFile(diff_file, diff_arr)
    # with open("Level3_diff.txt", "r+") as diff_file:
    #     diff_encoding, diff_tree = Huffman.Huffman_Encoding(readFileToList("Level3_diff.txt"))
    #     print("Encoded output", diff_encoding)
    # os.remove("Level3_diff.txt")
    #





#-------------------
    # level3Img = readPILimg("muhi.png")
    #
    # level3ImgArray = PIL2np(level3Img)
    # print(level3ImgArray)
    # level3_nrows = len(level3ImgArray)
    # level3_ncols = len(level3ImgArray[0])
    #
    # diff_arr = difference(newImgArr)
    # print(diff_arr)
    # print("difference pixels :", difference(newImgArr))
    #
    # with open("Level3.txt", "w+") as diff_file:
    #     writeMatrixToFile(diff_file, diff_arr)
    # with open("Level3.txt", "r+") as diff_file:
    #     diff_encoding, diff_tree = Huffman.Huffman_Encoding(readFileToList("Level3.txt"))
    #     print("Encoded output", diff_encoding)
    # os.remove("Level3.txt")
    #
    # # # saves binary code in new file.
    # with open("level3_compressed_file.txt", "w+") as f:
    #     f.write(diff_encoding)
    # with open("level3_diffArray_file.txt", "w+") as f:
    #     f.write(diff_encoding)
    #
    # level3_compressedFile = open("level3_compressed_file.txt", "r+")
    # newDiffEncoding = level3_compressedFile.read()
    # print("Decoded Output", Huffman.Huffman_Decoding(newDiffEncoding, diff_tree))
    #
    # newlevel3ImgArray = strToArr2DWithSpace(Huffman.Huffman_Decoding(newDiffEncoding, diff_tree), level3_nrows,
    #                                         level3_ncols)
    # print(newlevel3ImgArray)
    #
    # # convertMatrixToImage(newImgArr)
    # # stringToArray2DForImg(14, 12, newImgArr)
    # convertMatrixToImage(newlevel3ImgArray, "level3restoredImg.png")

    # --------------------------------------------
def Array2DToList2D(npArray):
    list_of_lists = list()
    for row in npArray:
        list_of_lists.append(row.tolist())
    return list_of_lists

def difference(arr):
    # en boy
    nrow = len(arr)
    ncolumn = len(arr[0])

    darr = [[0] * ncolumn] * nrow
    for i in range(nrow):
        for j in range(1, ncolumn):
            darr[i][j] = arr[i][j] - arr[i][j - 1]

    diff_arr = darr
    pivot = darr[0][0]
    diff_arr[0][0] = darr[0][0] - pivot
    for i in range(1, nrow):
        diff_arr[i][0] = (darr[i][0]) - int(darr[i - 1][0])
    diff_arr = np.matrix(diff_arr)

    return diff_arr


def redifference(diff_arr):
    pass


def Array2D_To_Array1D(arr2D):
    arr1D = []
    for ind in range(0, len(arr2D)):
        for ind2 in range(0, len(arr2D[0])):
            arr1D.append(arr2D[ind][ind2])
    return arr1D


def convertMatrixToImage(arr, str):
    # convert 2D array to image
    cv2.imwrite(str, arr)


def my_function(filepath):
    data = open(filepath, "r+")


def strToArr2DWithSpace(str, row, colmn):
    arr = np.fromstring(str, sep=" ", dtype=int)
    arr2D = np.reshape(arr, (colmn, row))
    return arr2D


def ListToNpArray(list):
    arr = numpy.array(list)
    return arr


def stringToArray2DForImg(h, w, str):
    arr2D = [w][h]
    arr1D = readFileTo1DArray(str)
    allPixels = 0
    while allPixels <= h + w:
        for ind in range(0, h):
            for ind2 in range(0, w):
                arr2D[ind][ind2] = arr1D[allPixels]
                allPixels += 1
    return print(arr2D)


def Array2DToText(arr):
    file_data = np.loadtxt(arr, dtype=int)

    text = ""
    for val in range(0, len(arr)):
        for val2 in range(0, len(arr[0])):
            text = text + ' ' + str(file_data[val][val2])

    return text


def saveBinFile(str):
    f = open("compressed_file.bin", "wb+")
    textToArray = np.array(list(str))
    arr = bytearray(textToArray)
    f.write(arr)
    f.close()


def readBinFile(binFile):
    with open(binFile, "br") as f:
        buffer = f.read()
        print("Length of buffer is %d" % len(buffer))

        for i in buffer:
            print(int(i))


def readFileTo1DArray(file):
    file_data = np.loadtxt(file, dtype=int)
    return file_data


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


def calculate_entropy(arr):
    entropyVal = 0
    for ind in range(0, len(arr)):
        entropyVal += arr[ind] * math.log(arr[ind], 2)
    return entropyVal * (-1)


def readFileTo2DArray(file):
    file_data = np.loadtxt(file, dtype=int)
    return file_data


def writeMatrixToFile(file, arr):
    # input dosyasına data yazdırır.
    for ind in range(0, len(arr)):
        if ind != 0:
            file.write("\n")
        for ind2 in range(0, len(arr[0])):
            file.write(' ' + str(arr[ind][ind2]))


def readFile(fileName):
    fileObj = open("input.txt", "r")  # opens the file in read mode
    words = fileObj.read().splitlines()  # puts the file into an array
    fileObj.close()
    return words


def readPILimg(image):
    # img dosyasının konumunu belirler.
    img = Image.open(image)
    # dosyanın içeriğini gösterir.
    img.show()
    # look at the function.
    img_gray = color2gray(img)
    print(img_gray)
    img_gray.show()
    # img_gray.save("newImg", 'png')
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
