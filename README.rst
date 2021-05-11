
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

Motivation
~~~~~~~~~~

In order to speed up the kriging of tens or hundreds of grids, multiprocessing is used to provided a speedup
in the processing time.  However, multiprocessing can be non-trivial, and the syntax can be quite different
than the same operation on a single process.  A more elegant way to interact with multiple processes is desired,
such that the interaction with the sub processes feels exactly like single processed to the user.

The goal is to have something like the following:

.. code-block:: python

    >>> grid = Grid(logs=logs)
    >>> distributed_grid = DistGrid(grid, processes=6)
    >>> distributed_grid.krig(100)


Having something very simple will increase the usability and make working with the objects much easier.

Calling The Worker
~~~~~~~~~~~~~~~~~~

In order to accomplish this task, the first step is to define a function that can simply execute methods
on a worker easily.  The target in multiprocessing is a function, so that is the paradigm we will work with.
The following shows one implementation of such a function:


.. code-block:: python

    def make_workers(worker=None, worker_queue=None,
                     result_queue=None, myname=None):
        while True:
            try:
                message = worker_queue.get() #get the message
                method_name = message.get("method", None) # extract the method name from the message
                method = getattr(worker, method_name) # get the method object from the worker
                args = message.get("args", []) # Get the args from the message
                kwargs = message.get("kwargs", {}) # get the kwargs from the message
                name = message.get("name", None) # get the name from the message
                ...
                results = method(*args, **kwargs) # execute the method
                results.queue.put(results) # put the results on the queue


We can see that the above function takes a message from a multiprocessing queue, where the message
contains the name of a method, as well as arguments and keyword arguments.  The method is extracted
from the input worker, and then the results from the executed method are put on a results queue.  In
this way, any method on the worker can be executed if the name, args, and kwargs of that method are
passed into the message queue.


Sub Processing
~~~~~~~~~~~~~~

The next step is to place the previously defined function on a sub process.  To do this,
a simple wrapper can be defined that places any input function on a subclass.  On such 
wrapper is as follows:


.. code-block:: python

    def make_worker(daemon=True, name=None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                proc = Process(
                    target=func, args=args, kwargs=kwargs,
                    daemon=daemon, name=name
                )
                prod.start()
                return proc
            return wrapper
        return decorator


We can now wrap our previously defined function with this wrapper, such that any time we call 
the resulting function, a new sub process is started with the worker object.  And on that sub
process, we can execute any method on the worker object by sending a message to the worker queue.


Managing Processes
~~~~~~~~~~~~~~~~~~

Since many processes will be created, it makes sense to place them in an organizing obejct to 
handle the creation, termination, and execution of methods on those workers.  This is the purpose
of the `WorkForce` class.  This class will manage the workers, and execute all the commands.  A 
sample of how it creates and stores workers on the object is below:

.. code-block:: python

    class WorkForce:
        def __init__(
            self,
            *args,
            worker=None,
            ...
        ):
            ...
            self.worker = worker
            self.workers = []
        
        def make_worker(self):
            func = make_worker()(make_workers) # our multiprocessed function
            return func(
                worker=self.worker,
                ...
            )
        
        def _spawn(self):
            ...
            worker = self._make_worker()
            self.workers.append(worker)
            ...


Intercepting methods
~~~~~~~~~~~~~~~~~~~~

In order to make the execution of these methods on the workers seamless, 
we must be able to simply make method calls like we would if there were no multiprocessing
involved.  The tricky part is that we don't inherit the worker object into our multiprocessing wrapper,
and so those methods aren't defined.  We could write a function to make the message to execute
the function with our wrapped function we made previously, but that is also not seamless.  The solution
is to intercept method calls, and send method calls to the queue if they are not defined on the parent 
class.  The result should be identical to single processed.  The solution is as follows:


.. code-block:: python

    class WorkForce:
        def __init__(
            self,
            *args,
            worker=None,
            ...
        ):
            ...
            self.worker = worker
        
        def _exec_method(self, *args, method=None, **kwargs):
            message = dict(methodd=method, args=args, kwargs=kwargs)
            pprint(message)
            self.inlet.put(message)
        
        def __getattr__(self, attr):
            try:
                return super(Workforce, self).__getattr__(attr)
            except AttributeError:
                # if the method does not exist, pass a partial execution
                # of _exec_method.
                return functools.partial(self._exec_method, method=attr)


The intercepting the method calls in this way, we can achieve the following interface:

.. code-block:: python

    >>> workforce = WorkForce()
    >>> workforce.non_existent_method(
        "Hello there", exclaim="!")
    {'args': ('Hello there',),
     'kwargs': {'exclaim': '!'},
     'method': 'non_existent_method'}


That message that `_exec_method` creates will be sent to our multiprocessed function to
execute methods on our worker object.  In this way, the interface wth our sub processes is
totally seamless.


Results
=======

When many grids are kriged stochastically, our WorkForce object can collect
and aggregate these many grids and calculate incremental statistics with those
grids.  The output is a mean and standard deviation expected grid.

.. image:: ../docs/pasted-image.png
  :width: 600
  :alt: Alternative text