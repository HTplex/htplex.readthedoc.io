Linux Shell Tips
================

Linux shell is often a lot more capable than you think. 
During my daily work, I often have the moment of "Wow, 
I didn't this is possible! It would save me so much time and effort if only I know this eariler.".
So here is a collection of tips, tricks and snippets that I found useful.

Auto retry a command
--------------------

For a command that may fail due to network issue, 
or any command that returns non-zero exit code when it fails, 
you can use the following snippet to retry it until it succeeds.

.. code-block:: bash

    until <command>; do sleep <wait time>; done
    # example
    until rsync --rsh=ssh -rP ~/data/llms yda4:/data/; do sleep 60; done

