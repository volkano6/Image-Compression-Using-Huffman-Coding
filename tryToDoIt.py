import fileinput
from pathlib import Path
from huffman2 import HuffmanCoding
import sys
import numpy
from PIL import *
from PIL import Image, ImageTk
import numpy as np
import os
from numpy import ndarray
import Huffman
import math




def main():
    #
    # # img = readPILimg("red.png")
    # # arr = PIL2np(img)
    # # print(arr)
    # arr = [[21,21,21,95,169,243,243,243],
    #        [21,21,21,95,169,243,243,243],
    #        [25,0,21,95,169,243,243,243],
    #        [21,21,21,95,169,243,243,23]]
    #
    # Darr_arr = difference(arr)
    # print(Darr_arr)
    # print(len(Darr_arr[0]))
    # print(reDifference(Darr_arr))
    # #RGB_comp("muhi.png")
    # # img = readPILimg("muhi.png")
    # # arr = PIL2np(img)

    # Dosya açar ve içine veri aktarır.
    # with open("Level2_grayLevelImage.txt", "w+") as file:
    #     pass
    # with open("Level2_grayLevelImage.txt", "r+") as file:
    #     writeMatrixToFile(file, arr)

    # array =readFileTo2DArray("Level2_grayLevelImage.txt")
    # array2 = readFileTo2DArray("blue_diff_encode.txt")
    # print(difference(array))
    # print(difference(array2))
     #Level1("input.txt")
    Level2("muhi.png", "muhi2.txt", "Level2_restoredImage.png")
    # Level3("muhi.png", "test_encode.txt", "restored_image.png")
    #Level4("uncompressed_cat2.png")
    #Level5("muhi.png")
    #print(comparison("muhi.png","restored_image.png"))
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

    # text to str.
    bin_file = open("Level1_test.txt", "r")
    encode_str = bin_file.read()
    bin_file.close()

    decom_path = h.decompress(output_path)
    print("Decompressed file path: " + decom_path)
    decodedOutput = Huffman.Huffman_Decoding(encode_str, tree)
    print("Decoded Output", decodedOutput.replace(" ", ""))


def Level2(input_image, Level2_Image_Encode, Level2_restoredImage):
    img = readPILimg(input_image)
    arr = PIL2np(img)

    # Dosya açar ve içine veri aktarır.
    with open("Level2_grayLevelImage.txt", "w+") as file:
        pass
    with open("Level2_grayLevelImage.txt", "r+") as file:
        writeMatrixToFile(file, arr)

    print("------------------------------------------------COMPRESSION")
    encoding, tree = Huffman.Huffman_Encoding(readFileToList("Level2_grayLevelImage.txt"))
    print("Encoded output", encoding)

    # bu arrayden değerler okunacak
    gray_level_2DArray = arr
    huffman_encoding = Huffman.Calculate_Codes(tree)

    # bu arraye binary kodlar yazılacak
    huffman_BinaryEncoding = []
    for ind in range(len(gray_level_2DArray)):
        col = []
        for ind2 in range(len(gray_level_2DArray[0])):
            col.append((huffman_encoding[gray_level_2DArray[ind][ind2]]))
        huffman_BinaryEncoding.append(col)

    with open(Level2_Image_Encode, "w+") as f:
        writeMatrixToFile(f, huffman_BinaryEncoding)

    path = Level2_Image_Encode
    h = HuffmanCoding(path)

    output_path = h.compress()
    print("Compressed file path: " + output_path)

    print("------------------------------------------------DECOMPRESSION")

    decom_path = h.decompress(output_path)
    print("Decompressed file path: " + decom_path)
    bin_Array_fromDecompressfile = readFileTo2DArray(decom_path, str)
    binfileRow = (len(bin_Array_fromDecompressfile))
    binfileColmn = (len(bin_Array_fromDecompressfile[0]))
    # text to str.
    bin_file = open(decom_path)
    encode_str = bin_file.read()
    bin_file.close()

    decom_path = np.array(decom_path, copy=True)

    print("Decoded Output", Huffman.Huffman_Decoding(encode_str, tree))
    decode_str= Huffman.Huffman_Decoding(encode_str, tree)

    decode_arr = np.fromstring(decode_str, dtype=int, sep=' ')
    decode_array = decode_arr.reshape(binfileRow,binfileColmn)
    print(decode_array)

    convertMatrixToImage(decode_array, Level2_restoredImage)


def Level3(img, Level3_diff_encode, Level3_restoredImage):
    level3Img = readPILimg(img)
    level3ImgNpArray = PIL2np(level3Img)
    print(level3ImgNpArray)

    level3ImgArray = Array2DToList2D(level3ImgNpArray)
    diff_arr = difference(level3ImgArray)
    diff_arr = np.array(diff_arr)
    print("difference pixels :", difference(diff_arr))

    # Dosya açar ve içine veri aktarır.
    with open("Level3_diff.txt", "w+") as file:
        pass
    with open("Level3_diff.txt", "r+") as file:
        writeMatrixToFile(file, diff_arr)

    print("--------------------")
    encoding, tree = Huffman.Huffman_Encoding(readFileToList("Level3_diff.txt"))

    print("Encoded output", encoding)
    with open(Level3_diff_encode, "w+") as f:
        f.write(encoding)

    path = "Level3_diff.txt"
    h = HuffmanCoding(path)

    output_path = h.compress()
    print("Compressed file path: " + output_path)

    # text to str.
    bin_file = open(Level3_diff_encode, "r")
    encode_str = bin_file.read()
    bin_file.close()

    decom_path = h.decompress(output_path)
    print("Decompressed file path: " + decom_path)
    print("Decoded Output", Huffman.Huffman_Decoding(encode_str, tree))

    diff_array_new = readFileTo2DArray("Level3_diff_decompressed.txt")
    print(diff_array_new)

    convertMatrixToImage(diff_array_new, Level3_restoredImage)
    # -------------------
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


def Level4(img):

    RGB_comp(img, 'blue', "blue.png")
    RGB_comp(img, 'green', "green.png")
    RGB_comp(img, "red", "red.png")

    print(" ")
    print(" ")

    print("For Blue Img:")
    Level2(input_image="blue.png", Level2_Image_Encode="blue_Image_encode.txt", Level2_restoredImage="blue_restored_Image.png")

    print(" ")
    print(" ")

    print("For Green Img:")
    Level2(input_image="green.png", Level2_Image_Encode="green_Image_encode.txt", Level2_restoredImage="green_restored_Image.png")

    print(" ")
    print(" ")

    print("For Red Img:")
    Level2(input_image="red.png", Level2_Image_Encode="red_Image_encode.txt", Level2_restoredImage="red_restored_Image.png")


def Level5(img):

    RGB_comp(img, 'blue', "blue.png")
    RGB_comp(img, 'green', "green.png")
    RGB_comp(img, "red", "red.png")

    print("For Blue Img:")
    Level3(img="blue.png", Level3_diff_encode="blue_diff_encode.txt",
           Level3_restoredImage="blue_restored_DiffImage.png")

    print(" ")
    print(" ")

    print("For Green Img:")
    Level3(img="green.png", Level3_diff_encode="green_diff_encode.txt",
           Level3_restoredImage="green_restored_DiffImage.png")

    print(" ")
    print(" ")

    print("For Red Img:")
    Level3(img="red.png", Level3_diff_encode="red_diff_encode.txt",
           Level3_restoredImage="red_restored_DiffImage.png")


def Array2DToList2D(npArray):
    list_of_lists = list()
    for row in npArray:
        list_of_lists.append(row.tolist())

    return list_of_lists


def difference(arr):
    # en boy
    nrow = len(arr)
    ncolumn = len(arr[0])
    print(ncolumn)

    darr = numpy.array(arr, copy=True)
    for i in range(nrow):
        for j in range(1, ncolumn):
            darr[i][j] = arr[i][j] - arr[i][j - 1]

    pivot_arr= []
    diff_arr = numpy.array(darr, copy=True)
    for i in range(nrow):
        pivot_arr.append(darr[i][0])
    pivot = darr[0][0]
    diff_arr[0][0] = darr[0][0] - pivot
    for i in range(1, nrow):
        diff_arr[i][0] = (darr[i][0]) - int(darr[i - 1][0])

    with open("pivot.txt", "w+") as f:

        writeArrToFile(f, pivot_arr)

    return diff_arr


def reDifference(diff_arr):

    nrow = len(diff_arr)
    ncolumn = len(diff_arr[0])
    pivot = []

    with open("pivot.txt", "r+") as f:
        str_arr = ','.join([l.strip() for l in f])

    piv_arr = np.asarray(str_arr.split(' '), dtype=int)

    darr = np.array(diff_arr, copy=True)

    for i in range(0, nrow):
        darr[i][0] = piv_arr[i]
    print()
    arr = np.array(darr, copy=True)


    for ind in range(nrow):
        for j in range(1, ncolumn):
            arr[ind][j] = darr[ind][j] + arr[ind][j-1]

    return arr


def Array2D_To_Array1D(arr2D):
    arr1D = []
    for ind in range(0, len(arr2D)):
        for ind2 in range(0, len(arr2D[0])):
            arr1D.append(arr2D[ind][ind2])
    return arr1D


def convertMatrixToImage(arr, str):
    # convert 2D array to image
    array = np2PIL(arr)
    array.save(str)


def pil_to_np(img):
   img_array = np.array(img)
   return img_array


def my_function(filepath):
    data = open(filepath, "r+")


def strToArr2DWithSpace(str, row, colmn):
    arr = np.fromstring(str, sep=" ", dtype=int)
    arr2D = np.reshape(arr, (colmn, row))
    return arr2D


def ListToNpArray(list):
    arr = numpy.array(list)
    return arr


def stringToArray2DForImg(w, h, str):
    arr2D = [w][h]
    arr1D = readFileTo1DArray(str)
    allPixels = 0
    while allPixels <= h + w:
        for ind in range(0, h):
            for ind2 in range(0, w):
                arr2D[ind][ind2] = arr1D[allPixels]
                allPixels += 1
    return arr2D


def Array2DToText(arr):
    # file_data = np.loadtxt(arr, dtype=int)
    file_data = arr
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


# boşluklu txt yi listeye çevirir.
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


def readFileTo2DArray(file, type):
    if type == int:
        file_data = np.loadtxt(file, dtype=type)
    else:
        file_data = np.loadtxt(file, dtype=str)
    return file_data

def writeArrToFile(file, arr):
    # input dosyasına data yazdırır.
    for ind in range(0, len(arr)):
        file.write(' ' + str(arr[ind]))


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


def RGB_comp(image_file_path, channel,str):
   # red channel -> 0, green channel -> 1 and blue channel -> 2
   if channel == 'red':
      channel_index = 0
   elif channel == 'green':
      channel_index = 1
   else:
      channel_index = 2
   # open the current image as a PIL image
   img_rgb = Image.open(image_file_path)
   # convert the current image to a numpy array
   image_array = pil_to_np(img_rgb)
   # traverse all the pixels in the image array
   n_rows = image_array.shape[0]
   n_cols = image_array.shape[1]
   for row in range(n_rows):
      for col in range(n_cols):
         # make all the values 0 for the color channels except the given channel
         for rgb in range(3):
            if (rgb != channel_index):
               image_array[row][col][rgb] = 0
   # convert the modified image array (numpy) to a PIL image
   pil_img = np_to_pil(image_array)
   pil_img.save(str)


def np_to_pil(img_array):
   img = Image.fromarray(np.uint8(img_array))
   return img

def open_ready_file(file):
    f = open(file, "w+")
    f = open(file, "r+")

if __name__ == '__main__':
    main()
