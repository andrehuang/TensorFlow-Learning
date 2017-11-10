

```python
import tensorflow as tf
```

### Initialize a TensorFlow variable


```python
weights = tf.Variable(tf.random_normal([300, 200], stddev=0.5), 
                     name = 'weights')
```

Here we pass two arguments to tf.Variable. 
- The first, *tf.random_normal*, is an operation that produces a tensor initialized using a normal distribution with standard deviation 0.5. We’ve specified that this tensor is of size 300x200, implying that the weights connect a layer with 300 neurons to a layer with 200 neurons. 
- We’ve also passed a name to our call to tf.Variable. The name is a unique identifier that allows us to refer to the appropriate node in the computation graph. 

- In this case, weights is meant to be trainable, or in other words, we will automatically compute and apply gradients to weights. If weights is not meant to be trainable, we may pass an optional flag when we call tf.Variable:
```python
weights = tf.Variable(tf.random_normal([300,200], stddev=0.5),
                        name="weights", trainable=False)
```

#### Several other methods

```python
# Common tensors from the TensorFlow API docs
tf.zeros(shape, dtype=tf.float32, name=None)
tf.ones(shape, dtype=tf.float32, name=None)
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32,
seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32,
seed=None, name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32,
seed=None, name=None)
```

When we call tf.Variable, three operations are added to the computation graph:
1. The operation producing the tensor we use to initialize our variable (*tf.random_noraml*, in our case)
2. The *tf.assign* operation, which is responsible for filling the variable with the initializing
tensor prior to the variable’s use
3. The variable operation, which holds the current value of the variable

Before we use any TensorFlow variable, the _tf.assign_ operation must be run so that the variable is appropriately initialized with the desired value. WE can do this by running  ``` tf.initialize_all_variables()```, which will trigger all of the _tf.assign_ operations in our graph.

### TensorFlow Operaions 

On a high-level, TensorFlow operations represent abstract transformations
that are applied to tensors in the computation graph. Operations may have
attributes that may be supplied a priori or are inferred at runtime. For example, an
attribute may serve to describe the expected types of the input (adding tensors of type
float32 vs. int32). 

Just as variables are named, operations may also be supplied with
an optional name attribute for easy reference into the computation graph.

### Placeholder Tensors

#### how we pass the input to our deep model (during both train and test time)?

A variable is insufficient because it is only meant to be initialized
once. We instead need a component that we populate every single time the
computation graph is run.

TensorFlow solves this problem using a construct called a _placeholder_. A placeholder
is instantiated as follows and can be used in operations just like ordinary TensorFlow
variables and tensors.


```python
x = tf.placeholder(tf.float32, name='x', shape=[None, 784])
# Here we define a placeholder where x represents a mini-batch of data 
# stored as float32’s.
W = tf.Variable(tf.random_uniform([784, 10], -1, 1), name="W")
multiply = tf.matmul(x, W)
```

We notice that x has 784 columns, which means that each data sample has 784 dimensions. We also notice that x has an undefined number of rows. This means that x can be initialized with an arbitrary number of data samples.

The result is that the *ith* row of the ```multiply``` tensor corresponds to  ```W```  multiplied with *ith* data sample

Just as variables need to be initialized the first time the computation graph is built, placeholders need to be __filled__ every time the computation graph (or a subgraph) is run.

### Sessions in TensorFlow

A TensorFlow program interacts with a computation graph using a ___session___. The TensorFlow
session is responsible for _building the initial graph_, can be used to _initialize all variables_ appropriately, and to _run the computational graph. _

```python
from read_data import get_minibatch()

x = tf.placeholder(tf.float32, name="x", shape=[None, 784])
W = tf.Variable(tf.random_uniform([784, 10], -1, 1), name="W")
b = tf.Variable(tf.zeros([10]), name="biases")
output = tf.matmul(x, W) + b
# decribe the computational graph that is bulit by the session when it is finally instantiated

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

feed_dict = {"x" : get_minibatch()}
sess.run(output, feed_dict= feed_dict)   
# run the subgraph, pass in the tensors we want to compute along with a feed_dict that fills the placeholders with the neceassary input data.
```

<div class='alert aler-block alert-info'> __Note__: ***feed_dict*** is used for filling the placeholders with the necessary input data.

The ```sess_run``` interface can also be used to train networks.

### Navigating Variable Scopes and Sharing Variables

building complex models often requires re-using and sharing large sets of variables that we’ll want to instantiate together in one place.

Unfortunately, trying to enforce modularity and readability can
result in unintended results if we aren’t careful. Let’s consider the following example:


```python
def my_network(input):
    W_1 = tf.Variable(tf.random_uniform([784, 100], -1, 1),
                        name="W_1")
    b_1 = tf.Variable(tf.zeros([100]), name="biases_1")
    output_1 = tf.matmul(input, W_1) + b_1
    
    W_2 = tf.Variable(tf.random_uniform([100, 50], -1, 1),
                        name="W_2")
    b_2 = tf.Variable(tf.zeros([50]), name="biases_2")
    output_2 = tf.matmul(output_1, W_2) + b_2

    W_3 = tf.Variable(tf.random_uniform([50, 10], -1, 1),
                        name="W_3")
    b_3 = tf.Variable(tf.zeros([10]), name="biases_3")
    output_3 = tf.matmul(output_2, W_3) + b_3

    # printing names
    print ("Printing names of weight parameters")
    print (W_1.name, W_2.name, W_3.name)
    print ("Printing names of bias parameters")
    print (b_1.name, b_2.name, b_3.name)
    return output_3
```

If we wanted to use this network multiple times, we’d prefer to encapsulate it into a compact function like ***my_network***, which we can call multiple times. However, when we try to use this network on two different inputs, we get something unexpected:


```python
i_1 = tf.placeholder(tf.float32, [1000, 784], name="i_1")
```


```python
my_network(i_1)
```

    Printing names of weight parameters
    W_1_4:0 W_2_3:0 W_3_3:0
    Printing names of bias parameters
    biases_1_3:0 biases_2_3:0 biases_3_3:0
    




    <tf.Tensor 'add_11:0' shape=(1000, 10) dtype=float32>




```python
i_2 = tf.placeholder(tf.float32, [1000, 784], name='i_2')
```


```python
my_network(i_2)
```

    Printing names of weight parameters
    W_1_5:0 W_2_4:0 W_3_4:0
    Printing names of bias parameters
    biases_1_4:0 biases_2_4:0 biases_3_4:0
    




    <tf.Tensor 'add_14:0' shape=(1000, 10) dtype=float32>



If we observe closely, our second call to `my_network` doesn’t use the same variables as
the first call (in fact __the names are different__!). Instead, we’ve created a second set of
variables!

In many cases, we don’t want to create a copy, but instead, we want to reuse
the model and its variables. It turns out, in this case, we shouldn’t be using
__`tf.Variable`__. Instead, we should be using a more advanced naming scheme that
takes advantage of TensorFlow’s variable scoping.

TensorFlow’s variable scoping mechanisms are largely controlled by two functions:
1. __`tf.get_variable(<name>, <shape>, <initializer>)`__: checks if a variable with
this name exists, retrieves the variable if it does, creates it using the shape and
initializer if it doesn’t
2. __`tf.variable_scope(<scope_name>)`__: manages the namespace and determines
the scope in which `tf.get_variable` operates

Let’s try to rewrite my_network in a cleaner fashion using TensorFlow variable scoping.
The new names of our variables are namespaced as __`"layer1/W", "layer2/b",
"layer2/W"`__, etc.


```python
def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_uniform_initializer(minval=-1, maxval=1)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    return tf.matmul(input, W) + b

def my_network(input):
    with tf.variable_scope("layer_1"):
        output_1 = layer(input, [784, 100], [100])

    with tf.variable_scope("layer_2"):
        output_2 = layer(output_1, [100, 50], [50])

    with tf.variable_scope("layer_3"):
        output_3 = layer(output_2, [50, 10], [10])
    
    return output_3
```


```python
i_1 = tf.placeholder(tf.float32, [1000, 784], name="i_1")
```


```python
my_network(i_1)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-27-ba0558216212> in <module>()
    ----> 1 my_network(i_1)
    

    <ipython-input-25-f51e46e8cf81> in my_network(input)
         10 def my_network(input):
         11     with tf.variable_scope("layer_1"):
    ---> 12         output_1 = layer(input, [784, 100], [100])
         13 
         14     with tf.variable_scope("layer_2"):
    

    <ipython-input-25-f51e46e8cf81> in layer(input, weight_shape, bias_shape)
          3     bias_init = tf.constant_initializer(value=0)
          4     W = tf.get_variable("W", weight_shape,
    ----> 5                         initializer=weight_init)
          6     b = tf.get_variable("b", bias_shape,
          7                         initializer=bias_init)
    

    d:\python\lib\site-packages\tensorflow\python\ops\variable_scope.py in get_variable(name, shape, dtype, initializer, regularizer, trainable, collections, caching_device, partitioner, validate_shape, use_resource, custom_getter)
       1063       collections=collections, caching_device=caching_device,
       1064       partitioner=partitioner, validate_shape=validate_shape,
    -> 1065       use_resource=use_resource, custom_getter=custom_getter)
       1066 get_variable_or_local_docstring = (
       1067     """%s
    

    d:\python\lib\site-packages\tensorflow\python\ops\variable_scope.py in get_variable(self, var_store, name, shape, dtype, initializer, regularizer, reuse, trainable, collections, caching_device, partitioner, validate_shape, use_resource, custom_getter)
        960           collections=collections, caching_device=caching_device,
        961           partitioner=partitioner, validate_shape=validate_shape,
    --> 962           use_resource=use_resource, custom_getter=custom_getter)
        963 
        964   def _get_partitioned_variable(self,
    

    d:\python\lib\site-packages\tensorflow\python\ops\variable_scope.py in get_variable(self, name, shape, dtype, initializer, regularizer, reuse, trainable, collections, caching_device, partitioner, validate_shape, use_resource, custom_getter)
        365           reuse=reuse, trainable=trainable, collections=collections,
        366           caching_device=caching_device, partitioner=partitioner,
    --> 367           validate_shape=validate_shape, use_resource=use_resource)
        368 
        369   def _get_partitioned_variable(
    

    d:\python\lib\site-packages\tensorflow\python\ops\variable_scope.py in _true_getter(name, shape, dtype, initializer, regularizer, reuse, trainable, collections, caching_device, partitioner, validate_shape, use_resource)
        350           trainable=trainable, collections=collections,
        351           caching_device=caching_device, validate_shape=validate_shape,
    --> 352           use_resource=use_resource)
        353 
        354     if custom_getter is not None:
    

    d:\python\lib\site-packages\tensorflow\python\ops\variable_scope.py in _get_single_variable(self, name, shape, dtype, initializer, regularizer, partition_info, reuse, trainable, collections, caching_device, validate_shape, use_resource)
        662                          " Did you mean to set reuse=True in VarScope? "
        663                          "Originally defined at:\n\n%s" % (
    --> 664                              name, "".join(traceback.format_list(tb))))
        665       found_var = self._vars[name]
        666       if not shape.is_compatible_with(found_var.get_shape()):
    

    ValueError: Variable layer_1/W already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
    
      File "d:\python\lib\site-packages\tensorflow\python\framework\ops.py", line 1204, in __init__
        self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access
      File "d:\python\lib\site-packages\tensorflow\python\framework\ops.py", line 2630, in create_op
        original_op=self._default_original_op, op_def=op_def)
      File "d:\python\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 767, in apply_op
        op_def=op_def)
    



```python
i_2 = tf.placeholder(tf.float32, [1000, 784], name="i_2")
```


```python
my_network(i_2)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-24-f3cfaa3a5387> in <module>()
    ----> 1 my_network(i_2)
    

    <ipython-input-20-f51e46e8cf81> in my_network(input)
         10 def my_network(input):
         11     with tf.variable_scope("layer_1"):
    ---> 12         output_1 = layer(input, [784, 100], [100])
         13 
         14     with tf.variable_scope("layer_2"):
    

    <ipython-input-20-f51e46e8cf81> in layer(input, weight_shape, bias_shape)
          3     bias_init = tf.constant_initializer(value=0)
          4     W = tf.get_variable("W", weight_shape,
    ----> 5                         initializer=weight_init)
          6     b = tf.get_variable("b", bias_shape,
          7                         initializer=bias_init)
    

    d:\python\lib\site-packages\tensorflow\python\ops\variable_scope.py in get_variable(name, shape, dtype, initializer, regularizer, trainable, collections, caching_device, partitioner, validate_shape, use_resource, custom_getter)
       1063       collections=collections, caching_device=caching_device,
       1064       partitioner=partitioner, validate_shape=validate_shape,
    -> 1065       use_resource=use_resource, custom_getter=custom_getter)
       1066 get_variable_or_local_docstring = (
       1067     """%s
    

    d:\python\lib\site-packages\tensorflow\python\ops\variable_scope.py in get_variable(self, var_store, name, shape, dtype, initializer, regularizer, reuse, trainable, collections, caching_device, partitioner, validate_shape, use_resource, custom_getter)
        960           collections=collections, caching_device=caching_device,
        961           partitioner=partitioner, validate_shape=validate_shape,
    --> 962           use_resource=use_resource, custom_getter=custom_getter)
        963 
        964   def _get_partitioned_variable(self,
    

    d:\python\lib\site-packages\tensorflow\python\ops\variable_scope.py in get_variable(self, name, shape, dtype, initializer, regularizer, reuse, trainable, collections, caching_device, partitioner, validate_shape, use_resource, custom_getter)
        365           reuse=reuse, trainable=trainable, collections=collections,
        366           caching_device=caching_device, partitioner=partitioner,
    --> 367           validate_shape=validate_shape, use_resource=use_resource)
        368 
        369   def _get_partitioned_variable(
    

    d:\python\lib\site-packages\tensorflow\python\ops\variable_scope.py in _true_getter(name, shape, dtype, initializer, regularizer, reuse, trainable, collections, caching_device, partitioner, validate_shape, use_resource)
        350           trainable=trainable, collections=collections,
        351           caching_device=caching_device, validate_shape=validate_shape,
    --> 352           use_resource=use_resource)
        353 
        354     if custom_getter is not None:
    

    d:\python\lib\site-packages\tensorflow\python\ops\variable_scope.py in _get_single_variable(self, name, shape, dtype, initializer, regularizer, partition_info, reuse, trainable, collections, caching_device, validate_shape, use_resource)
        662                          " Did you mean to set reuse=True in VarScope? "
        663                          "Originally defined at:\n\n%s" % (
    --> 664                              name, "".join(traceback.format_list(tb))))
        665       found_var = self._vars[name]
        666       if not shape.is_compatible_with(found_var.get_shape()):
    

    ValueError: Variable layer_1/W already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
    
      File "d:\python\lib\site-packages\tensorflow\python\framework\ops.py", line 1204, in __init__
        self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access
      File "d:\python\lib\site-packages\tensorflow\python\framework\ops.py", line 2630, in create_op
        original_op=self._default_original_op, op_def=op_def)
      File "d:\python\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 767, in apply_op
        op_def=op_def)
    


Here we will have __ValueError__! Unlike `tf.Variable`, the __`tf.get_variable`__ command checks that a variable of the given name hasn't already been instantiated.

By default, sharing is not allowed, but if you want to enable sharing within a variable scope, we can say so explicitly:


```python
with tf.variable_scope("shared___variables") as scope:
    i_1 = tf.placeholder(tf.float32, [1000, 784], name="i_1")
    my_network(i_1)
    scope.reuse_variables() # Attention!
    i_2 = tf.placeholder(tf.float32, [1000, 784], name="i_2")
    my_network(i_2)
```

This allows us to retain modularity while still allowing variable sharing! And as a nice
byproduct, our naming scheme is cleaner as well.

### Managing Models over the CPU and GPU

1. "/cpu:0": The CPU of our machine.
2. "/gpu:0": The first GPU of our machine, if it has one.
3. "/gpu:1": The second GPU of our machine, if it has one.
4. ... etc ...

To inspect which devices are used by the computational graph, we can initialize our TensorFlow session with the `log_device_placement` set to `True`:
```python
sess = tf.Session(config=tf.COnfigProto(log_device_placement=True))
```

If we desire to use a specific device, we may do so by using with __`tf.device`__ to select
the appropriate device. If the chosen device is not available, however, an error will be
thrown. If we would like TensorFlow to find another available device if the chosen
device does not exist, we can pass the __`allow_soft_placement`__ flag to the session variable
as follows:


```python
with tf.device('/gpu:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], name='a')
    b = tf.constant([1.0, 2.0], shape=[2, 1], name='b')
    c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True, log_device_placement=True))
sess.run(c)
```




    array([[  5.],
           [ 11.]], dtype=float32)



TensorFlow also allows us to build models that span multiple GPUs by building models in a "tower" like fashion. Sample code for multi-GPU code is shown below:


```python
c = []

for d in ['/gpu:0', '/gpu:1']:
    with tf.device(d):
        a = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], name='a')
        b = tf.constant([1.0, 2.0], shape=[2, 1], name='b')
        c.append(tf.matmul(a, b))

with tf.device('/cpu:0'):
    sum = tf.add_n(c)

sess = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,log_device_placement=True))

sess.run(sum)
```




    array([[ 10.],
           [ 22.]], dtype=float32)



### Specifying the Logistic Regression Model in TensorFlow 

Our goal is to identify handwritten
digits from 28 x 28 black and white images. The first network that we’ll build
implements a simple machine learning algorithm known as logistic regression.

You’ll notice that the network interpretation for logistic regression is rather primitive.
It doesn’t have any hidden layers, meaning that it is limited in its ability to learn complex
relationships!

We’ll build the the logistic regression model in four phases:
1. __inference__: which produces a probability distribution over the output classes
given a minibatch
2. __loss__: which computes the value of the error function (in this case, the cross
entropy loss)
3. __training__: which is responsible for computing the gradients of the model’s
parameters and updating the model
4. __evaluate__: which will determine the effectiveness of a model


```python
def inference(x):
    init=tf.constant_initializer(value=0)
    W = tf.get_variable("W", [784, 10],
                        initializer=init)
    b = tf.get_variable("b", [10],
                        initializer=init)
    output = tf.nn.softmax(tf.matmul(x, W) + b)
    return output
```


```python
def loss(output, y):
    dot_product = y * tf.log(output) # to calculate cross-entropy
    
    # Reduction along axis 0 collapses each column into a single
    # value, whereas reduction along axis 1 collapses each row
    # into a single value. In general, reduction along axis i
    # collapses the ith dimension of a tensor to size 1.

    xentropy = -tf.reduce_sum(dot_product, axis=1)
    loss = tf.reduce_mean(xentropy) # return a single value
    
    # Here, "reduce" means reduction in DIMENSIONS
    # Reduces input_tensor along the dimensions given in axis.
    # the rank of the tensor is reduced by 1 for each entry in axis.
    # If axis has no entries, all dimensions are reduced, 
    # and a tensor with a single element is returned.
    return loss
```


```python
def training(cost, global_step):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op
```

Note that when we create the
training operation, we also pass in a variable that represents the number of minibatches
that has been processed.

Each time the training operation is run, this step variable is incremented so that we can keep track of progress.

***global_step*** refer to the number of batches seen by the graph. Everytime a batch is provided, the weights are updated in the direction that minimizes the loss. global_step just __keeps track of the number of batches seen so far__. When it is passed in the minimize() argument list, the variable is increased by one (after other variables have been updated).


```python
def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1),
                                    tf.argmax(y, 1))
    # recall: every row of the output(defined in inference funciton) 
    # is the probability distribution over output classes for each
    # corresponding data sample in the minibatch
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
```

### Logging and Training the Logistic Regression Model

In order to
log important information as we train the model, we log several summary statistics.


```python
def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op
```

An old version:

Every epoch, we run the __tf.merge_all_summaries__ in order to collect all summary
statistics we’ve logged and use a __tf.train.SummaryWriter__ to write the log to disk. In
the next section, we’ll describe how we can use visualize these logs with the built-in
TensorBoard tool.

__IN ADDITION,__ we also sace the model parameters using the __tf.train.Saver__ model saver. By default, the saver maintains the latest 5 check-points, and we can restore them for future use.


```python
import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 100
display_step = 10

with tf.Graph().as_default():

    # mnist data image of shape 28*28=784
    x = tf.placeholder("float", [None, 784])
    # 0-9 digits recognition => 10 classes
    y = tf.placeholder("float", [None, 10])
    
    output = inference(x)
    cost = loss(output, y)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = training(cost, global_step)
    eval_op = evaluate(output, y)
    
    summary_op = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter("logistic_logs/",
                                            graph_def=sess.graph_def)

    init_op = tf.initialize_all_variables()

    sess.run(init_op)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            feed_dict = {x : minibatch_x, y : minibatch_y}
            
            sess.run(train_op, feed_dict=feed_dict)
            # Compute average loss
            minibatch_cost = sess.run(cost, feed_dict=feed_dict)
            avg_cost += minibatch_cost/total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            val_feed_dict = {
                x : mnist.validation.images,
                y : mnist.validation.labels
            }
            accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
        
            print("Validation Error:", (1 - accuracy))
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str,
                                        sess.run(global_step))
            saver.save(sess, "logistic_logs/model-checkpoint",
                                        global_step=global_step)
        
        print ("Optimization Finished!")

        test_feed_dict = {
                x : mnist.test.images,
                y : mnist.test.labels
        }
        
        accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
        print ("Test Accuracy:", accuracy)
```

### Leveraging TensorBoard to Visualize Computation Graphs and Learning

Once we set up the logging of summary statistics as described in the previous section,
we are ready to visualize the data we’ve collected. Lauching TensorBoard is as easy as running:

`
tensorboard --logdir=<your-absolute-path-to-log-dir>
`

### Builiding a Multilayer Model for MNIST in TensorFlow 

尽管是multilayer，但并不是CNN，用的还是矩阵乘积


```python
def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=w_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)
```


```python
def inference(x):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [784, 256], [256])
    
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [256, 256], [256])
    
    with tf.variable_scope("output"):
        output = layer(hidden_2, [256, 10], [10])
    
    return output
```

The performance of deep neural networks very much depends
on an effective initialization of its parameters。

For ReLU units, a study published in 2015 by He et al. demonstrates that the variance
of weights in a network should be 2/
n_in
, where n_in is the number inputs coming into
the neuron. (__Here explains why in w_init, the stddev (2.0/weight_shape[0])**0.5__)

The curious reader should investigate what happens when we change our
initialization strategy. For example, changing __`tf.random_normal_initializer`__ back
to the __`tf.random_uniform_initializer`__ we used in the logistic regression example
significantly hurts performance.

Finally, for slightly better performance, we perform the softmax __while computing the
loss__ instead of during the inference phase of the network. This results in the modification
below:


```python
def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
    loss = tf.reduce_mean(xentropy)
    return loss
```

#### The second version(multilayer) should be read side by side with the first version. Understanding the changes and improvements is very essential!
