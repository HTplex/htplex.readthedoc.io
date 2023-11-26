Blind 75
=================

Let's do all leetcode blind 75 problems



Indexs
---------------------
1. `q1`_ 
2. `q49`_ `q49hints`_ `q49sol`_
3. `q242`_ `q242hints`_ `q242sol`_
4. `q238`_ `q238hints`_ `q238sol`_



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


Questions
----------


.. _q1:

Q1. Two Sum
~~~~~~~~~~~~~~

.. _q49:

Q49. Group Anagrams
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Given an array of strings strs, group the anagrams together. You can return the answer in any order.
    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

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


.. _q242:

Q242. Valid Anagram
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Given two strings s and t, return true if t is an anagram of s, and false otherwise.
    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

    Example 1:
    Input: s = "anagram", t = "nagaram"
    Output: true

    Example 2:
    Input: s = "rat", t = "car"
    Output: false
 
    Constraints:
    1 <= s.length, t.length <= 5 * 104
    s and t consist of lowercase English letters.

    Follow up: What if the inputs contain Unicode characters? How would you adapt your solution to such a case?

.. q238:

Q238. Product of Array Except Self
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
    The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
    You must write an algorithm that runs in O(n) time and without using the division operation.

    Example 1:
    Input: nums = [1,2,3,4]
    Output: [24,12,8,6]

    Example 2:
    Input: nums = [-1,1,0,-3,3]
    Output: [0,0,9,0,0]
 
    Constraints:
    2 <= nums.length <= 10^5
    -30 <= nums[i] <= 30
    The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
 
    Follow up: Can you solve the problem in O(1) extra space complexity? (The output array does not count as extra space for space complexity analysis.)





Hints
----------

.. _q49hints:

Q49. Group Anagrams
~~~~~~~~~~~~~~~~~~~~

dict of lists

1. We just need to find a way to represent each anagram in a unique way, so we can group them together. And using a hashmap (dict) to store the anagrams as anagrams[key] = [anagram1, anagram2, ...]
2. the easiest way is to sort each anagram, and use the sorted anagram as the key. But this will take O(nlogn) time for each anagram, and O(n) space for the hashmap.
3. using counts as key can reduce the time to O(n) for each anagram, and O(n) space for the hashmap. But since the string is only 100 long, i dont think its worth it.
4. use dict.get(key,[]) for faster creating dict of lists. dict.get(key,[]) will return the value of the key if it exists, otherwise it will return the default value, which is [] here.

.. _q242hints:

Q242. Valid Anagram
~~~~~~~~~~~~~~~~~~~~

array of counts

1. Since we know the input is only lowercase English letters, we can use a 26 length array to store the count of each letter in s.
2. check if the count of each letter in t is the same as the count of each letter in s. By creating 2 counts or use the same one and minus the count of each letter in t.
3. get the index by subtracting the ascii value of each letter by 'a'. The python way is to use ord() function. and chr() to convert back to letter, not that we need it here.
4. if contains unicode, we can use a hashmap (dict) instead, since it will take too much memory to use an array of size 1,114,112 (unicode characters) to store the count of each letter.
5. can also use python's counter: Counter(word1) == Counter(word2)

.. _q238hints:

Q238. Product of Array Except Self
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

prefix and suffix

1. maintain 2 arrays, prefix and suffix
2. prefix[i] = nums[0]*nums[1]*...*nums[i-1]
3. suffix[i] = nums[i+1]*nums[i+2]*...*nums[n-1], or reversed to easier initalize
4. Follow up: Can you solve the problem in O(1) extra space complexity? (The output array does not count as extra space for space complexity analysis.)
5. we can use the output array to store the prefix, and use a variable to store the suffix





Solutions
------------

.. _q49sol:

.. code-block:: python

    class Solution:
        def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
            indexed_anagram = {}
            for str_item in strs:
                index = "".join(sorted(str_item))
                indexed_anagram[index] = indexed_anagram.get(index,[]) + [str_item]
            return list(indexed_anagram.values())

.. _q242sol:

.. code-block:: python

    class Solution:
        def isAnagram(self, s: str, t: str) -> bool:
            char_counts = [0]*26
            for current_char in s:
                char_counts[int(ord(current_char)-ord('a'))] += 1
            for current_char in t:
                char_counts[int(ord(current_char)-ord('a'))] -= 1
            return char_counts == [0]*26

.. _q238sol:

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