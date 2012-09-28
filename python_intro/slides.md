Introduction to Python
======================

Danny George

.fx: titleslide

---

What is Python?
===============

**Python is a dynamic, interpreted, garbage-collected programming language**

The reference interpreter is written in C (CPython), but other interpreters exist:

- Jython (JVM)
- IronPython (.NET)
- Pypy (Python)

Useless Trivia
--------------

- Created in 1991
- Named after Monty Python
- Google employs its creator, Guido van Rossum

---

Who uses it?
============

- Google
- YouTube
- Pinterest
- Instagram
- Yahoo!
- NASA
- Industrial Light & Magic
- Micron
- MIT
- *... who doesn't use it?*

---

What do people like about it?
=============================

Simple, easy, readable syntax.  
The language "gets out of the way" and let's you program.

There are standard libraries for the most common operations.

Very, _very_ high programmer productivity (fast development turn around).

Fast execution speed.

Excellent documentation.

Supports multiple programming styles (OO, imperative, functional, ...).

Interfaces with C (and everything else through that).

Very portable code; runs on Linux, OSX, Windows, ...

---

Python is Dynamic
=================

Variables are simply references to typed objects.  They can be reassigned to
other objects that may have different types.

Java:

    !java
    int x = 10;
    String s = "Java is static";

Python:

    !python
    x = 10
    x = "Python is awesome!"

**DEMO**

Python also allows for object introspection.  That is, you can look at an object
to see what it is capable of during runtime.

---

Python is Interpreted
=====================

print_time.py:

    !python
    import datetime

    print("the current date and time is: %s" % datetime.datetime.now())

output:

    user@host> python print_time.py
    the current date and time is: 2012-09-25 22:34:08.867020

  
There is no manual compilation step, which creates a rapid development cycle.

Modules will be compiled to byte-code files (`.pyc`)
upon demand and only be recompiled if the source is newer than the `.pyc` file.

You can also manually insert and compile code as the program is running.

---

Basic Types
===========

---

None and Bools
--------------

None
----
`None` is Python's keyword for NULL, nil, whatever; indicating the absence of a value.

Boolean keywords
----------------
True

False

...very self explanatory.

---

Types - Integer
---------------
Simple whole numbers.  There aren't any signed/unsigned variants.

    !python
    dec_num = 42
    hex_num = 0x200
    oct_num = 0o777     # (lowercase oh)
    bin_num = 0b111001
    neg_num = -8675309

You do not need to worry about overflow.  Python takes care of
the backend storage for you.

    !python
    >>> 2 ** 1000
    10715086071862673209484250490600018105614
    04811705533607443750388370351051124936122
    49319837881569585812759467291755314682518
    71452856923140435984577574698574803934567
    77482423098542107460506237114187795418215
    30464749835819412673987675591655439460770
    62914571196477686542167660429831652624386
    837205668069376

---

Types - Float
-------------
Numbers with a fractional component.  Again, you don't need to worry about the storage (e.g.`double` vs. `float`)

    !python
    x = 3.14
    y = 0.1

Floating point storage and rounding issues still exist, but Python does pretty well
on formatting your number as you would expect.

---

Types - String
--------------
There is no `char` type, a string is a string.  
Strings can be surrounded by double-quotes, single-quotes, and others.  
You can also use character escapes if you want.

    !python
    s1 = "It's easy"
    s2 = 'The cow says "Moo!"'
    s3 = "The students say \"This presentation is awesome.\""

Raw strings are useful for regular expression coding.

    !python
    regex = r"(\d+):\s*(\S+)"
    reg2 = """The farmer's cow says "Moo" often"""

As of Python 3.0, strings are Unicode (UTF-8) by default.

---

Types - String
--------------
Strings have many methods to make data processing trivial.

    !python
    s.split()   # split a string by whitespace
    s[0]        # access substrings (also slicing)
    s.find('substring')
    s.count("a")

Powerful printf-ish formatting

    !python
    # positional arguments
    s = "There are %d %s" % (10, cats)

    # keyword arguments
    s = "%(mynoun)s is very %(myadj)s times %(mynum)d" %
            {
                "mynoun": "python",
                "myadj": "intuitive",
                "mynum": 10,
            }

---

Basic Containers
================

---

List
=======

Python's list is the standard array-ish type.  
Its elements can be of mixed types.
Of course they use zero-based indexing.

    !python
    l = [10, 3.14, "string", [1, 2, 3, 4]]
    print(l[3])
    last_elem = l.pop()
    l.append("new_end")
    l.insert(3, 'new_fourth_element')
    print(len(l))

Slicing notation is useful to get certain subsets of the data.

    !python
    l = [0, 1, 2, 3, 4, 5]

    a = l[:]
    b = l[:3]   # b is now [0, 1, 2]
    c = l[3:]   # c is now [3, 4, 5]
    d = l[4:4]  # d is now [4]

--- 

Tuple
=======

Tuples are similar to lists except they are immutable.

    !python
    t = (1, 2, "string")

They support slicing, index access, etc..., but no operations
that would change the tuple itself.

    !python
    l = t[:]    # creates a list [1, 2, "string"]

Also, be aware of possible parentheses parsing issues - it doesn't
hurt to add an extra comma!

    t1 = (3)    # t is the integer 3
    t2 = (3,)   # t is a tuple (3,)

---

Dictionary
==========

The dictionary is probably Python's most useful and versatile type.  
It is a hash table mapping unique keys to values.  Keys can be of any type
that is hashable, and values can be whatever.

    !python
    d = {1: "one",
         "mylist": [1, 2, 3],
         "mydict": {"key": "value"}}

You can store and retrieve values through index notation

    !python
    x = d[1]            # x now references the string "one"
    l = d["mylist"]     # l now references [1, 2, 3]

Dicts also have many, many other useful methods

    !python
    k = d.keys()
    i = d.items()
    myval = d.get("questionable", None)
    # read the documentation for more!

---

Syntax
======

Python's most immediately abrasive, awesome feature:  
**Block scope is denoted by indentation.**  
Whitespace is _important_!  Good programmers should already do this anyway, and it
enforces a clean, consistent style.

C/C++/Java:

    !c
    if (x) {
        if (y)
            do_something1();
            do_something2();    // added absent-mindedly
        do_something3();
    }

Python:

    !python
    if x:
        if y:
            do_something1()
            do_something2()
        do_something3()

---

Syntax
======

Python eliminates many braces and semicolons.  
Don't try to be too clever, it's good to have only one operation per line.  
IMHO This makes code much easier to read, and faster to write.  
(also, no arguing over the one true "brace style")

C/C++/Java:

    !c
    if (x)
    {
        if (y) {
            do_something1(); do_something2();
        }
        do_something3();
    }

Python:

    !python
    if x:
        if y:
            do_something1()
            do_something2()
        do_something3()

---

Syntax - Loops
==============

Python has the familiar while loop, useful for counters and the like

    !python
    x = 0
    while x < 10:
        # do something imporant
        x += 1

In C/C++/Java, the `for` loop is essentially a convenient while loop.  
In Python the for loop is used to iterate over collections or generators  
(which is usually what you are doing with C/C++/Java for loops.)

    !python
    l = [10.5, 22.0, 3.5, 4.2, 5.5]
    sum = 0
    for x in l:
        sum += x

    # if you need the index as well use 'enumerate'
    for i, x in enumerate(l):
        sum += x * i

---

Functions
=========

Functions are declared with the `def` keyword, and are called with the familiar
parentheses notation.

    !python
    def myfunc(a, b, c):
        if a > 10:
            return b + c
        else:
            return a + b + c

    x = myfunc(3, 4, 5)

The dynamic typing makes the code much smaller (no return type, no variable types),
**but** requires more discipline on the programmer's part.

    !python
    y = myfunc("runtime", "failure", 10)
    z = myfunc("another", "fail")

---

Functions - Default Arguments
=============================

Argument defaults are a good way to make the calling API easier, as well as
provide self-documentation and concise, explicit argument passing.

    !python
    # default arguments
    def foo(a, mode="RGBA", fullscreen=True):
        if fullscreen:
            draw_fullscreen(a, mode)
        else:
            draw_window(a, mode)

    foo('planet')
    foo('comet', 'grayscale')
    foo('spaceship', 'RGB', False)
    foo('star', fullscreen=False)

---

Functions - Varargs
===================

Extra arguments are passed into the function as a list.  
The standard convention is to use the name `args`.

    !python
    # varargs
    def sum(*args):
        sum = 0
        for arg in args:
            sum += arg
        return sum

    sum(1, 2, 3)    # returns 6
    sum(10)         # returns 10
    sum()           # returns 0

---

Functions - Keyword Args
========================

Keyword args are passed into the function as a dictionary with string keys.  
These can be useful for passing many named arguments, but you cannot assume what keys
exist in the dict.  
The standard convention is to use the name `kwargs`

    !python
    # keyword args
    def cross_bridge(**kwargs):
        if kwargs.get('name', None) == None:
            throw_into_pit()
        else:
            allow_to_cross()
 
    cross_bridge(name="Sir Lancelot of Camelot",
                 quest="I seek the Holy Grail",
                 favorite_color="blue")

---

Liftoff
=======

Wow, that was a lot of boring syntax information. (there's much more)  
Where does the fun part come in?

Libraries!
----------

Much of Python's power and pervasiveness is due to the "batteries included"
attitude of the standard libraries.

There are many modules/libraries available in the default installation that
you can string together to solve problems very quickly.

- string processing
- network / socket programming  
- database apis  
- GUI builders  
- Files / operating system logic  
- webservers  
- specialized containers  

---

Libraries
=========

The standard distribution includes utilities for most programming tasks,
but there are even more very heavily used 3rd party libraries available online.

**SciPy / NumPy** - Widely used for scientific programming and data processing.

**Twisted** - Event driven network server framework (web, IM, ftp, etc...).

**Django, Plone, Zope** - Dynamic web page frameworks that connect databases to webpages.

**pyglet, PyOpenGL** - Create video games or other OpenGL applications.

**PyCUDA** - Python interface to CUDA programming

**Hundreds more...**

---

What else should I know?
========================

More syntax (for, while, with, list comprehensions, class, ...)  
Object-oriented features of Python (classes, modules, ...)  
Multi-paradigm programming (imperative, OO, functional, ...)  
Duck-typing, run-time function & class creation  
Exception based programming  
Interfacing with C/C++ programs  
**DEMO**

Alright, what about the "bad"?
------------------------------
Performance concerns (yes and no)  
When memory storage matters  
Multi-threading performance  
Python 2 vs. 3

**Only optimize your code where you need it.**

---

Questions?
==========

Resources:

**ipython** - better interactive Python shell (comes with scipy)

[**PEP8**](http://www.python.org/dev/peps/pep-0008/) - Python official suggested style guide

- [http://python.org](http://python.org)
- [http://docs.python.org/py3k/](http://docs.python.org/py3k/)
    - excellent tutorials
    - library reference
    - language reference
- [http://google.com](http://google.com)
- [http://freecomputerbooks.com](http://freecomputerbooks.com)

