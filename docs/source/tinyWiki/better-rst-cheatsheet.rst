Better reStructuredText Cheatsheet
=================
This is a chest sheet for rst, main refereces: chatgpt and https://sphinx-tutorial.readthedocs.io/cheatsheet/

Please referece the source code for the rst file to see how each section is made.

.. note:: 
    Example on how leveled titles work


This is a Subsection
---------------------
This is a subSubsection
^^^^^^^^^^^^^^^^^^^^^^^^^^
This is a subsubsubsection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can use any underlines for anything, it automatically becomes a 
levels based on which one used first. my preferece levels are =, -, ^, ~, and +

This is a Subsection 2
---------------------------
Jibber jibber jabber jabber.

.. note:: 
    Example on how section refereces work


Subsection A
------------

This is some content in subsection A.

.. _my-subsection-label:

Subsection B
------------

This is some content in subsection B.

Referencing Subsections
-----------------------

As discussed in `Subsection B`_, there are important points to consider.


.. note:: 
    Styles Example


.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:


The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, it will raise an exception.


For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']
