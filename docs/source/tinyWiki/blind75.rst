Blind 75
=================

Let's do all leetcode blind 75 problems



Indexs
---------------------

1. `q49`_ `q49hints`_ `q49sol`_
2. `q133`_ `q133hints`_ `q133sol`_
3. `q152`_ `q152hints`_ `q152sol`_
4. `q153`_ `q153hints`_ `q153sol`_
5. `q191`_ `q191hints`_ `q191sol`_
6. `q208`_ `q208hints`_ `q208sol`_
7. `q217`_ `q217hints`_ `q217sol`_
9.  `q238`_ `q238hints`_ `q238sol`_
10. `q242`_ `q242hints`_ `q242sol`_
11. `q300`_ `q300hints`_ `q300sol`_



Tips
-----

.. code-block:: python

    # think outloud, use commets and strach
    """
       1. inital thought
       2. example run
       3. edge cases
       4. time and space complexity
       5. start write
       6. check using example for each feature
    """


    # new list, m
    new_list = [0]*m
    # new list of list, m*n
    new_list = [[0]*n for _ in range(m)]

    # bfs queue
    bfs_queue = [root]

    while bfs_queue:
        current_node = bfs_queue.pop(0)
        for neighbor in current_node.neighbors:
            if neighbor not in seen: # for graph
                bfs_queue.append(neighbor)
                seen.add(neighbor)
    
    # dfs stack

Questions
----------

.. _q49:
Q49. Group Anagrams
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Given an array of strings strs, group the anagrams together. 
    You can return the answer in any order. An Anagram is a word 
    or phrase formed by rearranging the letters of a different word 
    or phrase, typically using all the original letters exactly once.

    Example 1:
    Input: strs = ["eat","tea","tan","ate","nat","bat"]
    Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

    Example 2:
    Input: strs = [""]
    Output: [[""]]

    Example 3:
    Input: strs = ["a"]
    Output: [["a"]]
 
    Constraints:
    1 <= strs.length <= 10^4
    0 <= strs[i].length <= 100
    strs[i] consists of lower-case English letters.

------------------------------------------------------------------------------------------------------------------------

.. _q133:
Q133. Clone Graph
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Given a reference of a node in a connected undirected graph.
    Return a deep copy (clone) of the graph.
    Each node in the graph contains a value (int) and a list 
    (List[Node]) of its neighbors.

    class Node {
        public int val;
        public List<Node> neighbors;
    }

    Test case format:
    For simplicity, each node's value is the same as the node's 
    index (1-indexed). For example, the first node with val == 1, 
    the second node with val == 2, and so on. The graph is represented 
    in the test case using an adjacency list.
    An adjacency list is a collection of unordered lists used to 
    represent a finite graph. Each list describes the set of neighbors 
    of a node in the graph.
    The given node will always be the first node with val = 1. 
    You must return the copy of the given node as a reference to the cloned graph.

Hint: `q133hints`_ Solution: `q133sol`_

------------------------------------------------------------------------------------------------------------------------

.. _q152:
Q152. Maximum Product Subarray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Given an integer array nums, find a subarray
    that has the largest product, and return the product.
    The test cases are generated so that the answer will 
    fit in a 32-bit integer.

    Example 1:

    Input: nums = [2,3,-2,4]
    Output: 6
    Explanation: [2,3] has the largest product 6.
    Example 2:

    Input: nums = [-2,0,-1]
    Output: 0
    Explanation: The result cannot be 2, because [-2,-1] is not a subarray.


    Constraints:

    1 <= nums.length <= 2 * 10^4
    -10 <= nums[i] <= 10
    The product of any prefix or suffix of nums 
    is guaranteed to fit in a 32-bit integer.

Hint: `q152hints`_ Solution: `q152sol`_

------------------------------------------------------------------------------------------------------------------------

.. _q153:
Q153. Find Minimum in Rotated Sorted Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

    [4,5,6,7,0,1,2] if it was rotated 4 times.
    [0,1,2,4,5,6,7] if it was rotated 7 times.
    Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

    Given the sorted rotated array nums of unique elements, return the minimum element of this array.

    You must write an algorithm that runs in O(log n) time.

    

    Example 1:

    Input: nums = [3,4,5,1,2]
    Output: 1
    Explanation: The original array was [1,2,3,4,5] rotated 3 times.
    Example 2:

    Input: nums = [4,5,6,7,0,1,2]
    Output: 0
    Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
    Example 3:

    Input: nums = [11,13,15,17]
    Output: 11
    Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
    

    Constraints:

    n == nums.length
    1 <= n <= 5000
    -5000 <= nums[i] <= 5000
    All the integers of nums are unique.
    nums is sorted and rotated between 1 and n times.

Hint: `q153hints`_ Solution: `q153sol`_

------------------------------------------------------------------------------------------------------------------------

.. _q191:
Q191. Number of 1 Bits
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Write a function that takes an unsigned integer and returns 
    the number of '1' bits it has (also known as the Hamming weight).

    Example 1:
    Input: n = 00000000000000000000000000001011
    Output: 3
    Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.
    
    Example 2:
    Input: n = 00000000000000000000000010000000
    Output: 1
    Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.

------------------------------------------------------------------------------------------------------------------------

.. _q208:
Q208. Implement Trie (Prefix Tree)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    A trie (pronounced as "try") or prefix tree is a tree data structure used
    to efficiently store and retrieve keys in a dataset of strings. There 
    are various applications of this data structure, such as autocomplete and spellchecker.

    Implement the Trie class:

    Trie() Initializes the trie object.
    void insert(String word) Inserts the string word into the trie.
    boolean search(String word) Returns true if the string word is in the trie
     (i.e., was inserted before), and false otherwise.
    boolean startsWith(String prefix) Returns true if there is a previously 
    inserted string word that has the prefix prefix, and false otherwise.
    

    Example 1:

    Input
    ["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
    [[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
    Output
    [null, null, true, false, true, null, true]

    Explanation
    Trie trie = new Trie();
    trie.insert("apple");
    trie.search("apple");   // return True
    trie.search("app");     // return False
    trie.startsWith("app"); // return True
    trie.insert("app");
    trie.search("app");     // return True
    

    Constraints:

    1 <= word.length, prefix.length <= 2000
    word and prefix consist only of lowercase English letters.
    At most 3 * 10^4 calls in total will be made to insert, search, and startsWith.

------------------------------------------------------------------------------------------------------------------------

.. _q217:
Q217. Contains Duplicate
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Given an integer array nums, return true if any value appears at least twice
     in the array, and return false if every element is distinct.

    Example 1:
    Input: nums = [1,2,3,1]
    Output: true

    Example 2:
    Input: nums = [1,2,3,4]
    Output: false

    Example 3:
    Input: nums = [1,1,1,3,3,4,3,2,4,2]
    Output: true

------------------------------------------------------------------------------------------------------------------------

.. _q242:
Q242. Valid Anagram
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Given two strings s and t, return true if t is an anagram of 
    s, and false otherwise. An Anagram is a word or phrase formed 
    by rearranging the letters of a different word or phrase, 
    typically using all the original letters exactly once.

    Example 1:
    Input: s = "anagram", t = "nagaram"
    Output: true

    Example 2:
    Input: s = "rat", t = "car"
    Output: false
 
    Constraints:
    1 <= s.length, t.length <= 5 * 104
    s and t consist of lowercase English letters.

    Follow up: What if the inputs contain Unicode characters? 
    How would you adapt your solution to such a case?

------------------------------------------------------------------------------------------------------------------------

.. _q238:
Q238. Product of Array Except Self
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Given an integer array nums, return an array answer such that 
    answer[i] is equal to the product of all the elements of nums 
    except nums[i]. The product of any prefix or suffix of nums is 
    guaranteed to fit in a 32-bit integer. You must write an algorithm 
    that runs in O(n) time and without using the division operation.

    Example 1:
    Input: nums = [1,2,3,4]
    Output: [24,12,8,6]

    Example 2:
    Input: nums = [-1,1,0,-3,3]
    Output: [0,0,9,0,0]
 
    Constraints:
    2 <= nums.length <= 10^5
    -30 <= nums[i] <= 30

    The product of any prefix or suffix of nums is guaranteed to fit in 
    a 32-bit integer.
 
    Follow up: Can you solve the problem in O(1) extra space complexity? 
    (The output array does not count as extra space for space complexity 
    analysis.)

------------------------------------------------------------------------------------------------------------------------

.. _q300:
Q300. Longest Increasing Subsequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Given an integer array nums, return the length of the longest strictly 
    increasing subsequence.

    A subsequence is a sequence that can be derived from an array by deleting 
    some or no elements without changing the order of the remaining elements. 
    For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].

    Example 1:
    Input: nums = [10,9,2,5,3,7,101,18]
    Output: 4
    Explanation: The longest increasing subsequence is [2,3,7,101], therefore 
    the length is 4.

    Example 2:
    Input: nums = [0,1,0,3,2,3]
    Output: 4

    Example 3:
    Input: nums = [7,7,7,7,7,7,7]
    Output: 1
 
    Constraints:
    1 <= nums.length <= 2500
    -10^4 <= nums[i] <= 10^4
 
    Follow up: Can you come up with an algorithm that runs in O(n log(n)) time complexity?

------------------------------------------------------------------------------------------------------------------------




Hints
-----

.. _q49hints:
Q49. Group Anagrams
~~~~~~~~~~~~~~~~~~~~

dict of lists

1. We just need to find a way to represent each anagram in a unique way, so we can group them together. And using a hashmap (dict) to store the anagrams as anagrams[key] = [anagram1, anagram2, ...]
2. the easiest way is to sort each anagram, and use the sorted anagram as the key. But this will take O(nlogn) time for each anagram, and O(n) space for the hashmap.
3. using counts as key can reduce the time to O(n) for each anagram, and O(n) space for the hashmap. But since the string is only 100 long, i dont think its worth it.
4. use dict.get(key,[]) for faster creating dict of lists. dict.get(key,[]) will return the value of the key if it exists, otherwise it will return the default value, which is [] here.

------------------------------------------------------------------------------------------------------------------------

.. _q133hints:
Q133. Clone Graph
~~~~~~~~~~~~~~~~~~

use a hashmap to index the old and new nodes

------------------------------------------------------------------------------------------------------------------------

.. _q152hints:
Q152. Maximum Product Subarray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

use 2 numbers to store max positive and min negative product so far

------------------------------------------------------------------------------------------------------------------------

.. _q153hints:
Q153. Find Minimum in Rotated Sorted Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

binary search, just be clear about all 4 cases
also check edge cases using examples

------------------------------------------------------------------------------------------------------------------------

.. _q191hints:
Q191. Number of 1 Bits
~~~~~~~~~~~~~~~~~~~~~~~~

easy counting and divide by 2

------------------------------------------------------------------------------------------------------------------------

.. _q208hints:
Q208. Implement Trie (Prefix Tree)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

use a dict of list(27len) of dict to store the children of each node
dont foregt to mark the end of a word

------------------------------------------------------------------------------------------------------------------------

.. _q217hints:
Q217. Contains Duplicate
~~~~~~~~~~~~~~~~~~~~~~~~~

use a set to store the seen numbers, 
return TRUE if already in set

------------------------------------------------------------------------------------------------------------------------
.. _q242hints:
Q242. Valid Anagram
~~~~~~~~~~~~~~~~~~~~

array of counts

1. Since we know the input is only lowercase English letters, we can use a 26 length array to store the count of each letter in s.
2. check if the count of each letter in t is the same as the count of each letter in s. By creating 2 counts or use the same one and minus the count of each letter in t.
3. get the index by subtracting the ascii value of each letter by 'a'. The python way is to use ord() function. and chr() to convert back to letter, not that we need it here.
4. if contains unicode, we can use a hashmap (dict) instead, since it will take too much memory to use an array of size 1,114,112 (unicode characters) to store the count of each letter.
5. can also use python's counter: Counter(word1) == Counter(word2)

------------------------------------------------------------------------------------------------------------------------

.. _q238hints:
Q238. Product of Array Except Self
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

prefix and suffix

1. maintain 2 arrays, prefix and suffix
2. prefix[i] = nums[0]*nums[1]*...*nums[i-1]
3. suffix[i] = nums[i+1]*nums[i+2]*...*nums[n-1], or reversed to easier initalize
4. Follow up: Can you solve the problem in O(1) extra space complexity? (The output array does not count as extra space for space complexity analysis.)
5. we can use the output array to store the prefix, and use a variable to store the suffix

------------------------------------------------------------------------------------------------------------------------

.. _q300hints:
Q300. Longest Increasing Subsequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dp with n^2 time

128493,7 = max(7,17,127,1287,12847,128497,1284937)

------------------------------------------------------------------------------------------------------------------------




Solutions
---------

.. _q49sol:
Q49. Group Anagrams
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Solution:
        def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
            indexed_anagram = {}
            for str_item in strs:
                index = "".join(sorted(str_item))
                indexed_anagram[index] = indexed_anagram.get(index,[]) + [str_item]
            return list(indexed_anagram.values())

------------------------------------------------------------------------------------------------------------------------

.. _q133sol:
Q133. Clone Graph
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Definition for a Node.
    """
    class Node:
        def __init__(self, val = 0, neighbors = None):
            self.val = val
            self.neighbors = neighbors if neighbors is not None else []
    """

    from typing import Optional
    class Solution:
        def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
            # create dict, indexing every old node to new node, for easy triversing
            # empty case
            # bfs queue need to be repopulated somewhere, somewhere will not create-
            # infinite loop, but make sure it's not too early
            # 
            if not node:
                return node
            new_root = Node(node.val,[])
            old_new_index = {node:new_root}

            bfs_queue = [node]

            while bfs_queue:
                c_old_node = bfs_queue.pop(0)
                for neighbor in c_old_node.neighbors:
                    if neighbor not in old_new_index:
                        bfs_queue.append(neighbor)
                        old_new_index[neighbor] = Node(neighbor.val,[])
                    old_new_index[c_old_node].neighbors.append(old_new_index[neighbor])

            return new_root

------------------------------------------------------------------------------------------------------------------------

.. _q152sol:
Q152. Maximum Product Subarray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Solution:
        def maxProduct(self, nums: List[int]) -> int:
            # dp?
            # maintain 2 lists. max pos and max neg, keep updates on final answer
            # list[i] = max pos product of subarray nums[:i+1],include nums[i]
            # for nums = 2 3 -2 4
            # [2,6,-2,4]
            # [2,3,-2,-8]
            # update everything at the end in case of stuff changes in the middle

            # max_product = nums[0]
            # pos_max = [nums[0]] # [2,6,-2,4]
            # neg_max = [nums[0]] # [2,3,-2,-8]
            # for num in nums[1:]:
            #     if num >= 0:

            #         pos_max.append(max(pos_max[-1]*num,num))
            #         neg_max.append(min(neg_max[-1]*num,num))
            #         max_product = max(max_product,pos_max[-1])
            #     else:

            #         pos_max.append(max(neg_max[-1]*num,num))
            #         neg_max.append(min(pos_max[-2]*num,num)) # already appended
            #         max_product = max(max_product,pos_max[-1])

            # after this, it turns out no list is needed, make it 2 numbers afterwards.
            # no need to access history, therefore no need to maintain 2 lists

            max_product = nums[0]
            pos_max = nums[0] # [2,6,-2,4]
            neg_max = nums[0] # [2,3,-2,-8]
            for num in nums[1:]:
                if num >= 0:

                    pos_max = max(pos_max*num,num)
                    neg_max = min(neg_max*num,num)
                    max_product = max(max_product,pos_max)
                else:
                    pos_max_new = max(neg_max*num,num)
                    neg_max_new = min(pos_max*num,num)
                    pos_max,neg_max = pos_max_new,neg_max_new
                    max_product = max(max_product,pos_max)

            return max_product

------------------------------------------------------------------------------------------------------------------------

.. _q153sol:
Q153. Find Minimum in Rotated Sorted Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    class Solution:
        def findMin(self, nums: List[int]) -> int:
            # split
            # this type of array can only split into 
            # 1. 2 sorted half, l < r [1,2,3,4,5,6,7], return a[0]
            # 2. 2 sorted half, l > r [5,6,7,1,2,3,4], return a[len(a)//2]
            # 3. rotated sorted half, sorted half [6,7,1,2,3,4,5], do first half
            # 4. sorted half, rotated sorted half [3,4,5,6,7,1,2], do 2nd half
            # remeber to update everything after a loop

            while True:
                middle_idx = len(nums)//2
                left = nums[:middle_idx]
                right = nums[middle_idx:]                
                if not right:
                    return left[0]
                if not left:
                    return right[0]

                if left[0]<=left[-1]:
                    if right[0]<=right[-1]:
                        return min(left[0],right[0])
                    else:
                        nums = right
                else:
                    nums = left

------------------------------------------------------------------------------------------------------------------------

.. _q191sol:
Q191. Number of 1 Bits
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Solution:
        def hammingWeight(self, n: int) -> int:
            ham = 0
            while n > 0:
                ham += n%2
                n = n//2
            return ham
            
------------------------------------------------------------------------------------------------------------------------

.. _q208sol:
Q208. Implement Trie (Prefix Tree)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # just use dict of dicts instead of treenode next time
    class TrieNode:
        def __init__(self):

            self.next = [None]*27 # end symbol


    class Trie:
        #! need some sort of end symbol
        def __init__(self):
            self.root = TrieNode()

        def insert(self, word: str) -> None:
            current_node = self.root
            for c in word:
                # dont overwrite when building the tree
                if current_node.next[ord(c)-ord('a')] is None:
                    current_node.next[ord(c)-ord('a')]=TrieNode()
                current_node = current_node.next[ord(c)-ord('a')]
            current_node.next[-1] = TrieNode()

        def search(self, word: str) -> bool:
            current_node = self.root
            for c in word:
                if current_node.next[ord(c)-ord('a')] is None:
                    return False
                current_node = current_node.next[ord(c)-ord('a')]
            # ! check if it ends
            print(current_node.next)
            return current_node.next[-1] is not None

            

        def startsWith(self, prefix: str) -> bool:
            # same with search but no check
            current_node = self.root
            for c in prefix:
                if current_node.next[ord(c)-ord('a')] is None:
                    return False
                current_node = current_node.next[ord(c)-ord('a')]
            return True

------------------------------------------------------------------------------------------------------------------------

.. _q217sol:
Q217. Contains Duplicate
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Solution:
        def containsDuplicate(self, nums: List[int]) -> bool:
            # temp_set = set({}), temp_set.add(num)
            number_set = set({})
            for num in nums:
                if num not in number_set:
                    number_set.add(num)
                else:
                    return True
            return False      

------------------------------------------------------------------------------------------------------------------------

.. _q238sol:
Q238. Product of Array Except Self
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Solution:
        def productExceptSelf(self, nums: List[int]) -> List[int]:
            # 1 2 3 4 5
            # result = [(1)*2*3*4*5, 1* 3*4*5, 1*2* 4*5, 1*2*3* 5, 1*2*3*4*(1)]
            prefix_products = [1]*len(nums)
            suffix_products = [1]*len(nums)

            for i in range(len(nums)-1):
                prefix_products[i+1] = prefix_products[i]*nums[i]
                suffix_products[i+1] = suffix_products[i]*nums[-1*i-1]
            
            results = [prefix_products[i]*suffix_products[-1*i-1] for i in range(len(nums))]
            return results

"""
                
------------------------------------------------------------------------------------------------------------------------

.. _q242sol:
Q242. Valid Anagram
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Solution:
        def isAnagram(self, s: str, t: str) -> bool:
            char_counts = [0]*26
            for current_char in s:
                char_counts[int(ord(current_char)-ord('a'))] += 1
            for current_char in t:
                char_counts[int(ord(current_char)-ord('a'))] -= 1
            return char_counts == [0]*26

------------------------------------------------------------------------------------------------------------------------

.. _q300sol:
Q300. Longest Increasing Subsequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Solution:
        def lengthOfLIS(self, nums: List[int]) -> int:
            # dp because:
            #   we have to make decisions that may depend on previously made decisions
            #   decisions on n impact future decisions
            #   if we select a number as the sequece, it can be too large
            #   so that we can choose later suitable numbers
            # also 2500 so n^2 (maybe)
            # to choose a new number, we have to calculate every case to make sure its 
            # the best
            # 128493,7 = max(7,17,127,1287,12847,128497,1284937)
            lls_list = [1]*len(nums) # lls[i] = lls(nums[:i+1]), 1 is worst case
            lls_list[0] = 1
            max_lls = 1
            for i in range(1,len(nums),1):
                for j in range(i):
                    if nums[i] > nums[j]:
                        lls_list[i] = max(lls_list[i],lls_list[j]+1)
                max_lls = max(max_lls,lls_list[i])
            return max_lls
            
