CUDA Thrust
===========

Danny George

.fx: titleslide

---

What is Thrust?
===============

Thrust is a parallel algorithms library designed similar to C++'s STL,
used to abstract away many of CUDA's lower level details.

It was created outside of NVIDIA, but now is part of the standard
CUDA toolkit distribution. (If you are using version CUDA 4.0+, you
already have it on your computer).

Thrust source is also available on github, and is distributed under the
Apache license.

But forget about all of that for now...

---

Let's learn about C++!
=============================

(If you already know the C++ STL, you now get to sit through a review.)

One driving feature of C++ are its templates and the STL library.  
C++11 is further pushing these ideas and shows no sign of slowing.

Templates/Generic programming is used to write **type-safe** functions where
the logic is the same regardless of the type(s) passed to it.  The C++ compiler
actually duplicates the template code for each type that it was instantiated with.

    !cpp
    template<typename T>
    T sum(T a, T b, T c) {
        return a + b + c;
    }

    //...

    int   x = sum(10, 20, 30);
    float f = sum(1.0f, 3.14f, 2.6f);

---

C++ STL Containers
==================

C++ templates are probably most widely used through the STL containers.  
**std::vector**, **std::string**, **std::map**, **std::list**, etc...

Besides the OO features and convenience, these containers are designed to
rise-above basic C pointers, providing more safety from memory violations,
while maintaining the bare-metal performance.

I'm going to focus on `std::vector` from here on out, but the operations should
work for most of the other containers as well.

---

std::vector
-----------

The vector template is designed to replace C's arrays.

Arrays in C are basically typed pointers.  They have no notion of length,
*they are just pointers*.  If you need to store more elements than initially allocated, you
have to manually reallocate memory and move the original elements.

The `std::vector` template does keep track of its length, and it also takes care of memory
management behind the scenes upon demand.  Its elements are guaranteed to be stored
contiguously, so it is every bit as efficient as a C array.

    !cpp
    std::vector<int> my_ints(4, 100);   // four ints with value 100

    size_t cur_size = my_ints.size();
    my_ints.push_back(30);
    my_ints.push_back(60);

---

Iterator logic
==============

Let's talk about looping over elements in a collection.  
How about we multiply a set of numbers by 2.

C
---

    !c
    int myset[10];
    // ... initialize myset ...

    // using looping and indexes
    for (int i=0; i<10; ++i) {
        myset[i] *= 2;
    }

    // OR

    // using pointer arithmetic
    int *cur = myset;
    int *end = myset + 10;
    for (; cur != end; ++cur) {
        *cur *= 2;
    }

---

Iterator logic
==============

Now the same thing in C++ using a `std::vector`.  
Multiply a set of numbers by 2.

C++
---

    !cpp
    std::vector<int> myset(10);
    // ... initialize myset ...

    // using looping and indexes
    for (int i=0; i<10; ++i) {
        myset[i] *= 2;
    }

    // OR

    // using STL iterators
    std::vector<int>::iterator it;
    for (it = myset.begin(); it != myset.end(); ++it) {
        *it *= 2;
    }

---

Iterator logic
==============

So what do STL iterators give us?  **Flexibility!**

Iterating through vectors isn't that impressive, but the _exact same_
logic will work for containers that aren't contiguous in memory.  
`(list, map, set, ...)`

You can also create iterators that aren't part of any memory-backed container at all.

- Counting iterators take a single number and can keep returning
  a never ending incrementing set of numbers.  
- Constant iterators take a single number and can keep returning that
  same number forever.

...but let's keep using regular `vector`s and their iterators for now.

---

STL algorithms
==============

OK, we have a generic "begin" and "end" to our collections as well
as a way to iterate from one end to the other.

Here's where smart people came up with the idea of building generic
algorithms using only that functionality.

Our simple for loop can be turned into one of these generic algorithms.

    !cpp
    void doubleIt(int &val) {
        val *= 2;
    }

    std::vector<int> myset(10);
    std::for_each(myset.begin(), myset.end(), doubleIt);

---

STL algorithms
==============

    !cpp
    void doubleIt(int &val) {
        val *= 2;
    }
   
    std::vector<int> myset(10);
    std::for_each(myset.begin(), myset.end(), doubleIt);

Now if you didn't pay attention, you just missed some awesomeness.

We are now programming _'what I want done'_ instead of _'this is how I want it done'_.
I just want every number in `myset` doubled, I don't care how it's done.  
This is getting into _functional_ programming within C++

I could also change from using a `vector` to a `list`, and it would still work.

Warning:  blatant foreshadowing ahead
-------------------------------------
Do I even care if each number in the set is doubled in parallel?  
C++11 added a `std::parallel_for_each` algorithm to the STL.

---

Another Meander - C++ Functors
==============================

Argh, we were just getting into parallel stuff!

Alright, what is a functor?
---------------------------
A C++ functor is a "function-object".  In other words, it is an object
that can be called and treated just like a regular function.  Let's see an example.

    !cpp
    class Doubler {
        public:
            void operator()(int &val) const {
                val *= 2;
            }
    };

    Doubler d1, d2; // two instances of the Doubler class
    int i=20, j=30;

    d1(i);         // both instances can be called as
    d2(j);         //   if they were regular functions

    // now i=40 and j=60

---

C++ Functors
============

OK, so what do functors give us? **LOTS!**

\- We can maintain per-instance state.  
\- We can pass functors around just like variables.  
\- Compiler optimizations (vs. function pointers)  

    !cpp
    class AddX {
        public:
            AddX(int x) : _x(x) { }

            int operator()(const int &val) const {
                return val + _x;
            }
        private:
            const int _x;
    };

    AddX add5(5), add10(10);                // two instances
    std::cout << add5(3)  << std::endl;     // prints 8
    std::cout << add10(3) << std::endl;     // prints 13


---

Back to std::for\_each
======================

We can also pass functor instances into the STL algorithms.

    !cpp
    void doubleIt(int &val) {
        val *= 2;
    }

    class Doubler {
        public:
            void operator()(int &val) const {
                val *= 2;
            }
    };
    
    std::vector<int> myset(10);
    Doubler myDoubler;

    // the next two lines are functionally equivalent
    std::for_each(myset.begin(), myset.end(), doubleIt);
    std::for_each(myset.begin(), myset.end(), myDoubler);

---

Other STL Algorithms
====================

The STL provides much more than the basic `std::for_each`

    !cpp
    std::binary_search(begin, end, value);
    std::copy(src_begin, src_end, dst_begin);
    std::count(begin, end, value);
    std::count_if(begin, end, pred);  // pred is a bool functor
    std::fill(begin, end, value);
    std::sort(begin, end, cmp);       // similar to C's qsort
    std::transform(src1_begin, src1_end, dst_begin, unaryOp);
    std::transform(src1_begin, src1_end, src2_begin,
                    dst_begin, binaryOp);
    // many more...

`std::transform` should be explained more thoroughly.  
Note that it does not modify the input collection, but stores all results of the
function into an output collection.  
If you've ever heard of Map/Reduce from functional programming, this is the 'Map' function.

---

STL Algorithms
==============

One last note about STL algorithms... They also work on raw, dirty pointers.

    !cpp
    int multiplyBy2(int &val) {
        return val * 2;
    }

    int src[20];
    int dst[20];

    std::fill(src, src+20, 42);     // set all 20 src elements to 42
    std::transform(src, src+20, dst, multiplyBy2);

    // NOTE: you must ensure that 'dst' is big enough
    //       to fit the results (dirty pointers)

Do not use this as an excuse to not prefer std::vector wherever possible!  
...also eat your vegetables.

---

Am I in the wrong room?
=======================

I thought this presentation was about CUDA Thrust.

Alright then! Let's learn about Thrust.

Thrust provides two vector template containers:  
**`thrust::host_vector`**   uses memory on the host
**`thrust::device_vector`** uses memory on the device (GPU)

Thrust also provides many algorithms modeled after what's in the STL.

That was the learning curve.

But wait, there's more!
=======================


Back to the beginning:  
_Thrust is a parallel algorithms library designed similar to C++'s STL,
used to abstract away many of CUDA's lower level details._

---

Thrust - Memory
===============

With raw CUDA, you have to deal with a lot of low-level memory management.  

    !cpp
    cudaMalloc(void ** devPtr, size_t size);
    cudaMemcpy(void * dst, void * src, size_t size, direction);
    cudaFree(void * devPtr);
    // ...

With Thrust, let the library do the work for you.

    !cpp
    thrust::host_vector<int> h_nums(1024);
    thrust::device_vector<int> d_nums;

    // copy from host to device
    d_nums = h_nums;    // calls cudaMemcpy in background

    // do some number crunching on the GPU

    // copy from device to host
    h_nums = d_nums;


---

Thrust - Memory
===============

Let's explore more functionality that Thrust provides to make
memory operations easier.

    !cpp
    thrust::host_vector<int>   h1(10);
    thrust::device_vector<int> d1;

    // instantiate and copy in one step
    thrust::device_vector<int> d2(h1)
    thrust::device_vector<int> d3(h1.begin(), h1.begin()+5)

    // remember: vector memory will resize on demand

    d1 = h1;    // host to device
    h1 = d1;    // device to host
    d1 = d2;    // device to device

    // access individual elements from a device vector
    std::cout << d1[4] << std::endl;

---

Thrust - Memory
===============

Be sure to keep in mind what is actually happening in the background.

    !cpp
    thrust::device_vector<int> d1(65536);

    // the for loop below is OK for quick debugging,
    //  but is obnoxiously slow - Why?
    for (int i=0; i<d1.size(); ++i) {
        std::cout << d1[i] << std::endl;
    }

    // there are d1.size() cudaMemcpy calls being made!
    // it is usually better to copy the whole vector or 
    //  decent sized chunks into host_vectors first

    thrust::host_vector<int> h1(d1);    // one cudaMemcpy
    thrust::host_vector<int> h2(d1.begin(), d1.begin()+512);

    for (int i=0; i<h1.size(); ++i) {
        std::cout << h1[i] << std::endl;
    }

---

Thrust - Kernel launching
=========================

With raw CUDA, you have to be aware of Grid and Block dimensions.  
Optimal launch parameters can depend on many things, including what
hardware you are running on.

    !cpp
    // call some device code
    add_some_vectors<<<dim3 gridSize, dim3 blockSize>>>(
            int *in1, int *in2, int *out
        );

With Thrust, use template algorithms!
All of the logic for grid/block dimensions is contained in the library,
so let it make smart guesses on what to use.

    !cpp
    thrust::device_vector<int> d_src1, d_src2, d_dst;

    thrust::transform(d_src1.begin(), d_src1.end(),
                      d_src2.begin(),
                      d_dst.begin(),
                      addFunc);             // functor!

---

Thrust Algorithms
=================

Thrust provides many built-in algorithms to use, and the cool thing
is that they will be run in parallel on the GPU!

Most of the algorithms are modeled after the STL algorithms for
familiarity.

    !cpp
    thrust::find(begin, end, value);
    thrust::find_if(begin, end, Predicate);
    thrust::copy(src_begin, src_end, dst_begin);
    thrust::copy_if(src_begin, src_end, dst_begin, Predicate);
    thrust::count(begin, end, value);
    thrust::count_if(begin, end, Predicate);
    thrust::equal(src1_begin, src1_end, src2_begin);
    thrust::min_element(begin, end, [Cmp])
    thrust::max_element(begin, end, [Cmp])
    thrust::merge(src1_begin, src1_end, src2_begin, src2_end, dst_begin);
    thrust::sort(begin, end, [Cmp])
    // Map/Reduce   (transform === map)
    thrust::transform(src1_begin, src1_end, src2_begin,
                        dst_begin, BinaryOp);
    thrust::reduce(begin, end, init, BinaryOp);
    // ... and many more ...

---

Thrust - Kernel launching
=========================

    !cpp
    thrust::device_vector<int> d_src1, d_src2, d_dst;
    thrust::transform(d_src1.begin(), d_src1.end(), d_src2.begin(),
                      d_dst.begin(), addFunc);

Another cool thing about the Thrust algorithms is that because
of the magic of C++ templates, you can use the same algorithms
on `thrust::host_vectors`!  

When called with a `host_vector` the computation will be done by the CPU,
and when called with a `device_vector` it will be done by the GPU.

    !cpp
    int x, y;
    thrust::host_vector<int> hvec;
    thrust::device_vector<int> dvec;

    // (thrust::reduce is a sum operation by default)
    x = thrust::reduce(hvec.begin(), hvec.end());   // on CPU
    y = thrust::reduce(dvec.begin(), dvec.end());   // on GPU

---

Thrust - Functors
=================

OK, so Thrust provides some vectors and generic algorithms for me.  But what if
I want to do **more** than just sort, count, and sum my data?

Well, look at the title of the slide! - You can make your own functors
and pass these into Thrust's generic algorithms.

    !cpp
    // calculate result[] = (a * x[]) + y[]
    struct saxpy {
        const float _a;
        saxpy(int a) : _a(a) { }

        __host__ __device__
        float operator()(const float &x, const float& y) const {
            return a * x + y;
        }
    };

    thrust::device_vector<float> x, y, result;
    // ... fill up x & y vectors ...
    thrust::transform(x.begin(), x.end(), y.begin(),
                        result.begin(), saxpy(a));

---

Thrust - Functors
=================

    !cpp
    struct saxpy {
        const float _a;
        saxpy(int a) : _a(a) { }

        __host__ __device__
        float operator()(const float &x, const float& y) const {
            return a * x + y;
        }
    };

Let's look more at that.
The `operator()` function is prefixed with `__host__ __device__`.
This is CUDA compiler notation, but to Thrust it means that it can be called
with a `host_vector` OR `device_vector`.

Also notice that in this form of programming you don't need to worry about
`threadIdx` and `blockIdx` index calculations in the kernel code.  You can just
focus on what needs to happen to each element.

There are many excellent Thrust examples (including this saxpy one)
in the installed distribution and online.

---

What else is there?
===================

Fancy Iterators
---------------

    !cpp

    // continually returns a constant value
    thrust::constant_iterator
    // provides a counter - useful for getting index
    thrust::counting_iterator
    // zip together multiple iterators to pass more arguments
    thrust::zip_iterator

CUDA intermixing
----------------

There are some tasks that may not be easily fit into Thrust's algorithms.  
For this you can always fall back to using standard CUDA, but you will
need to be more aware of what is happening with memory and the CUDA scheduler.

You may also need to use different memory regions (texture mem) for large
performance gains - for this you will need to manage the memory yourself
with regular CUDA **but**  
You can also use raw pointers allocated with `cudaMalloc` as iterators in
Thrust algorithms as well.


---

So what's the bad?
===================

C++ templates in general cause slower compile times and even small typos can
create unreadable walls of error messages from the compiler.

It may be difficult to shoehorn some structures into `vector` form -
fallback to CUDA in those parts when necessary.  
For example, problems where you need to know your current index/position,
such as image processing or matrix multiplication.

When working with more than 2 input vectors to an algorithm, you often
have to work with `zip_iterator`s - not difficult, but does get a bit tedious.
This is required to maintain compile time type safety.

---

Closing Thoughts
=================

For portability and flexibility, strive towards programming _"what you want
done"_, and not _"how you want it done"_.  This is only going to get more important
in the heterogeneous CPU/GPU computing future.

Be aware of the memory operations that are going on.  
To make CUDA/GPU computing "worth it" you have to have enough
number crunching speedup to outweigh the memory transfers.

There are some tasks that may be unsuited for Thrust's algorithms,
but you can always fall back to raw CUDA if needed and intermix the two styles.

When writing parallel functions, be aware of possible race conditions.  
Use immutable/const data whenever possible to minimize errors.

And remember - if you are currently using CUDA, you probably already
have Thrust installed.  You don't need to do anything special, you still use
`nvcc` to compile everything.

[http://thrust.github.com](http://thrust.github.com)
