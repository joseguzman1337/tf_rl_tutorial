tf_rl_tutorial
===================

| ![tf](pics/tf1_small.png) | ![tsne](pics/tsne_small.png) | ![transe](pics/transe2.png) |
| ------------------------- | ---------------------------- | --------------------------- |

Accompanying code for "Relational Learning with TensorFlow" tutorial

The tutorial can be be viewed here: http://nbviewer.jupyter.org/github/fireeye/tf_rl_tutorial/blob/master/tf_rl_tutorial.ipynb

Please use the nbviewer link above instead of viewing the notebook directly from this GitHub repo. The GitHub renderer won't correctly display the images, equations, or the scatter plot.

### Dependencies
* Python 3: http://www.python.org
* TensorFlow: http://www.tensorflow.org
* Numpy: http://www.numpy.org
* Pandas: http://pandas.pydata.org

### Running the WordNet Example
In addition to the tutorial notebook, there is also a script which demonstrates training and evaluating a model.

1. Clone this repository, make sure all dependencies are installed
2. Download and unpack the WordNet dataset into the /data directory (link is in the tutorial)
3. Make sure that this project directory is in your PYTHONPATH
4. cd tf_rl_tutorial
5. python wordnet_eval.py
