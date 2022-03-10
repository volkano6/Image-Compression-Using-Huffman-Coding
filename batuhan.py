def Huffman_Encoding(data):
    symbol_with_probs = Calculate_Probability(data)
    symbols = symbol_with_probs.keys()
    probabilities = symbol_with_probs.values()
    print("symbols: ", symbols)
    print("probabilities: ", probabilities)

    summ = 0.0
    for i in probabilities:
        summ += i

    probs :[float] =[]

    for k in probabilities:
        x = k/summ
        floatX = "{:.2f}".format(x)
        probs.append(floatX)

    print ("All probabilities " ,probs)
    h = 0.00
    for i in range(len(probs)):
        h = h + math.log((float(probs[i])),2)*float(probs[i])
    h = -1 * h
    print("Entropy",h)
    nodes = []

    # converting symbols and probabilities into huffman tree nodes
    for symbol in symbols:
        nodes.append(Node(symbol_with_probs.get(symbol), symbol))

    while len(nodes) > 1:
        # sort all the nodes in ascending order based on their probability
        nodes = sorted(nodes, key=lambda x: x.prob)
        # for node in nodes:
        #      print(node.symbol, node.prob)

        # pick 2 smallest nodes
        right = nodes[0]
        left = nodes[1]

        left.code = 0
        right.code = 1

        # combine the 2 smallest nodes to create new node
        newNode = Node(left.prob + right.prob, left.symbol + right.symbol, left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    huffman_encoding = Calculate_Codes(nodes[0])
    print("symbols with codes", huffman_encoding)

    symbolsLength = []
    length: [int] = []
    for j in huffman_encoding.values():
        symbolsLength.append(j)
        for x in range(len(symbolsLength)):
            length.append((len(symbolsLength[x])))
            # lAve += ((len(symbolsLength[x])) * probs[x]) + lAve

        lAve = 0.00
        for x, y in zip(range(len(length)), range(len(probs))):
            lAve += int(length[x]) * float(probs[y])
        print("Average Code Length", lAve)

        Total_Gain(data, huffman_encoding)
        encoded_output = Output_Encoded(data, huffman_encoding)
        return encoded_output, nodes[0]