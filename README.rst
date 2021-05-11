
What is Kriging
===============

Kriging is a geostatistical technique to interpolate data in a sparse matrix of data.  Unlike
other methods like linear interpolation and IDW, kriging attempts to model in a more sophisticated manner
the spatial behavior of isotropy and stationarity.  For more detailed information on kriging, please see
https://desktop.arcgis.com/en/arcmap/10.3/tools/3d-analyst-toolbox/how-kriging-works.htm


Motivation
==========

In Oil & Gas, we deal with 3-dimensional data sets where data is very dense in the z- dimension, 
and extremely sparse in the x and y dimensions. It is of interest to exploration companies to intelligently 
interpolate across the space to figure out the values in between the observed data points. This is where kriging comes in. 
The problem with kriging is that is yields a deterministic imputation on the space. However, I believe there 
is a way to supercharge this idea to create a much richer output.


Workflow
========

The workflow for this project is fairly linear.  Data from logs, whether it be in numpy form or LAS files are loaded
into a `Log` object.  That log object houses the log data, as well as the metadata that describe where in space that
log exists, and other pertinent information about the log.  Many log objects can be compiled into a list and fed into
a `Grid` object.  The `Grid` object will manage the ingestion of those logs, and place them appropriately in 3D space 
given the associated metadata.  That `Grid` object behaves exactly like a numpy array in 2D or 3D, as it is a sublass.
On that object, special methods are added to allow us to perform stochastic kriging.  We go one step further by loading
the grids into a multiprocessing orchestrator object called `WorkForce`.  This allows us to distribute the work of kriging
the grids over many processes and speeds up the time to create many grids.  The general workflow diagram is below. 

.. image:: ../docs/workflow.png
  :width: 600
  :alt: Alternative text

Going Further
=============

There is a method of kriging that involves kriging random samples of data many times.  After many random samples,
the entire sample space is kriged.  This creates one actualization of a possible grid.  When many grids are kriged
this way, the result is an estimation of mean and standard deviation across the kriged space.  This can be very 
useful for contextualizing the information available, and this is what this package attempts to solve.

Stochasic Kriging
=================
.. image:: ../docs/krig.gif
  :width: 600
  :alt: Alternative text


There are two main python workflows that dominated this project: Subclassing numpy, and elegant multiprocessing.
I will go over in detail how each one of these was accomplished here.


Subclassing Numpy
=================

Motivation
~~~~~~~~~~

Since kriging is mostly linear algebra, it makes sense to preform most of the work using numpy.
However, it is a bit awkward to have an attribute on a custom object where the numpy array grid
is stored.  It would make more sense to inherit numpy so that the methods therein are more directly
available, and so that it is more intuitive for users of numpy using this project.  For example, 
calling `.shape` or other numpy methods on arrays.

Overriding __new__
~~~~~~~~~~~~~~~~~~


In order to subclass numpy, we have to override the `__new__` method.  This is why a GridConstructor
pattern was used to create the grids in this project.  By using a constructor class, we can simply create
the grid in the `__new__` method, then cast it as a numpy view and add in our own methods.  See below:

.. code-block:: python

    class Grid(np.ndarray):
        def __new__(cls, *args, **kwargs):
            grid = GridConstructor(*args, **kwargs)  # Make the grid
            obj = np.asarray(grid.grid).view(cls) # Cast as a view
            for k, v in grid.__dict__.items():
                if k != "grid":
                    setattr(obj, k, v)
            return obj


However, we can note that an issue occurs when we attempt to copy our class or subset it.

.. code-block:: python

    >>> grid2 = grid.copy()
    >>> grid2.logs
    Tracebacck (most recent call last):
      File "~stdin>", line 1, in ~module>
    AttibuteError: 'Grid' object has no attribute 'logs'

This is because separate special methods are responsible for copying numpy arrays or for
view casting.  Going a little deeper, we find that the `__array_finalize__` needs to be
overridden in order to successfully preserve our attributes across copies.

.. code-block:: python

    def __array_finalize__(self, obj):
        if obj is None:
            return
        try:
            for k, v in obj.__dict__.items():
                setattr(self, k, v)
        except AttributeError:
            pass

By overriding this special method, we can ensure that our attributes are preserved across copies.

In order to speed up the kriging process, multiprocessing is emplyed to distribute the work
across many workers.  During the transfer of memory from the main process to the child process,
objects must be pickled, or something comparable, and then unpickled on the child process.  We 
see that our numpy subclassing is not sufficient, as we get errors like the ones below on a sub process.

.. code-block:: python

    >>> def print_logs(grid):
    ...     print(grid.logs)
    >>> 
    >>> from multiprocessing import Process
    >>> 
    >>> process = Process(target=print_logs, arg=[grid])
    >>> process.start()
    ...
    AttributeError: 'Grid' object has no attribute 'logs'

In order to persist our subclassing to child processes, more special methods need to be overridden
to ensure that our object is pickled and unpickled properly.  These methods for numpy are the `__reduce__`
and `__setstate__` methods.  

Pickling

The following should make sure that our objects end up in the pickled state.

.. code-block:: python

    def __reduce__(self):
        pickled_state = super(Grid, self).__reduce__()
        extras = {k: v for k, v in self.__dict__.items()}
        new_state = pickled_state[2] + (extras, )
        return (pickled_state[0], pickled_state[1], new_state)


Unpickling

The next bit should unpack our pickled object to ensure that our objects make it into the new object.

.. code-block:: python

    def __setstate__(self):
        self.__dict__.update(state[-1])
        super(Grid, self).__setstate__(state[:-1])


Success!
~~~~~~~~

We've now successfully subclassed numpy so that our custom class will persist accross copies and pickling.
Now when we interact with our `Grid` object, it feel exactly like numpy.

.. code-block:: python

    >>> grid = Grid(logs=logs)
    >>> grid
    Grid([[[1.49924105],
           [1.54893128],
            ...
        ]])
    >>> grid.shape
    (345, 322, 1)

Multiprocessing (The elegant way)
=================================


