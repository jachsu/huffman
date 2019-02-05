"""
Code for compressing and decompressing using Huffman compression.
"""
from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    dict_ = {}
    for b in text:
        if b not in dict_:
            dict_[b] = 1
        else:
            dict_[b] += 1
    return dict_


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    d = freq_dict
    l_v = sorted(d.values())   # ascending frequencies
    l_n = [HuffmanNode(s) for s in sorted(d, key=d.get)] # ascending keys in HuffmanNodes
    if len(d) == 0:
        return HuffmanNode()
    elif len(d) == 1:
        return l_n[0]
    else:
        while len(l_v) > 2:
            prev = HuffmanNode(None, l_n[0], l_n[1])   # two with lowest freq
            l_v = [l_v[0]+l_v[1]]+l_v[2:]    # add the freq of the two and insert sum into the list.
            l_n = [prev]+l_n[2:]
            moving = l_v[0]
            l_v.sort()
            ind = l_v.index(moving)
            l_n.insert(ind, l_n.pop(0))
        return HuffmanNode(None, l_n[0], l_n[1])


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    if tree is None:
        return {}
    else:
        c = ''
        d = {}
        codes_helper(tree, d, c)
        return d


def codes_helper(tree, d, c):
    """
    A helper function for get_codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'.
    @param dict d: a dictionary recording codes.
    @param str c: code
    """
    if tree.is_leaf():
        d[tree.symbol] = c
    else:
        codes_helper(tree.left, d, c + '0')
        codes_helper(tree.right, d, c + '1')


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # post_order
    if tree is not None:
        l = [0]
        counter(tree, l)


def counter(tree, l):
    """
    A helper function for number_nodes.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'.
    @param list l: a list of number for next internal nodes.
    """
    if tree.is_leaf():
        pass
    else:
        counter(tree.left, l)
        counter(tree.right, l)
        tree.number = l[0]
        l[0] += 1


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    codes = get_codes(tree)
    total_l = sum([len(codes[n]) * freq_dict[n] for n in freq_dict])
    total_f = sum([freq_dict[n] for n in freq_dict])
    return total_l/total_f


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    bit = ''.join([codes[b] for b in text])
    return bytes(bits_to_byte(bit[i: i + 8]) for i in range(0, len(bit), 8))


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    # post order visit of internal nodes.
    # leaf:0
    # not leaf:1
    if tree.is_leaf():
        return bytes([])
    else:
        r = bytes([int(not tree.left.is_leaf()),
                   tree.left.symbol if tree.left.is_leaf()
                   else tree.left.number,
                   int(not tree.right.is_leaf()),
                   tree.right.symbol if tree.right.is_leaf()
                   else tree.right.number])
        return tree_to_bytes(tree.left)+tree_to_bytes(tree.right)+r


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def make_node(n, i):
    """
    Make the elements in node_lst HuffmanNodes.
    """
    t = HuffmanNode()
    if not n.l_type:
        t.left = HuffmanNode(n.l_data)
    else:
        t.left = HuffmanNode()
        t.left.number = n.l_data
    if not n.r_type:
        t.right = HuffmanNode(n.r_data)
    else:
        t.right = HuffmanNode()
        t.right.number = n.r_data
    t.number = i
    return t


def make_tree(tree, l_):
    """
     Make a tree by rooted at tree with subtrees from l_
     """

    if tree.is_leaf():
        pass
    else:
        if tree.left.number is not None:
            tree.left = make_node(l_[tree.left.number], tree.left.number)
            make_tree(tree.left, l_)
        if tree.right.number is not None:
            tree.right = make_node(l_[tree.right.number], tree.right.number)
            make_tree(tree.right, l_)


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)))
    """
    tree = make_node(node_lst[root_index], root_index)
    make_tree(tree, node_lst)
    return tree


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)))
    """
    l = [make_node(n, node_lst) for n in node_lst]
    l = l[:root_index+1]

    i = 0
    while i in range(len(l)):
        if l[i].left.symbol is None:
            l[i].left = l[i-2]
        if l[i].right.symbol is None:
            l[i].right = l[i-1]
        i += 1
    return l[root_index]


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes

    >>> text = bytes([216, 0])
    >>> size = 4
    >>> tree = HuffmanNode(None, HuffmanNode(None, HuffmanNode(3), \
HuffmanNode(None, HuffmanNode(1), HuffmanNode(4))), \
HuffmanNode(None, HuffmanNode(2), HuffmanNode(5)))
    >>> list(generate_uncompressed(tree, text, size))
    [5, 4, 3, 3]
    """

    codes = get_codes(tree)
    inv = {codes[k]: k for k in codes.keys()}
    s = ''.join([byte_to_bits(n) for n in text])
    l = []
    i = 1
    prev = 0
    while len(l) in range(size):
        if s[prev:i] in inv:
            l.append(inv[s[prev:i]])
            prev = i
        i += 1
    return bytes(l)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    d = freq_dict
    l = sorted(d, key=d.get, reverse=True)  # symbols by frequency, descending
    pre_order(tree, l)


def pre_order(tree, l):
    """
    Assign the nodes in tree with optimal symbol from l.

    @param HuffmanNode tree:Huffman tree rooted at 'tree'
    @param list l: a list of freq_dict symbols by frequency
    """
    if tree is None:
        pass
    else:
        if tree.is_leaf():
            tree.symbol = l[0]
            l.remove(l[0])
        pre_order(tree.left, l)
        pre_order(tree.right, l)


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
