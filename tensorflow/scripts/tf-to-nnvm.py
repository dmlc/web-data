"""
Sample Utility to convert tensorflow model compatible to TVM
============================================================

Args:

<model_file> <model_out> <out_node>

Ex: tf-to-nnvm.py mobilenet_v1_1.0_224_frozen.pb mobilenet_v1_1.0_224_frozen-shapes.pb 'MobilenetV1/Predictions/Reshape_1'

"""

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
import sys

model_file=sys.argv[1]
model_out=sys.argv[2]
out_node=sys.argv[3]

def _ProcessGraphDefParam(graph_def):
    """Type-checks and possibly canonicalizes `graph_def`."""
    if not isinstance(graph_def, graph_pb2.GraphDef):
        # `graph_def` could be a dynamically-created message, so try a duck-typed
        # approach
        try:
            old_graph_def = graph_def
            graph_def = graph_pb2.GraphDef()
            graph_def.MergeFrom(old_graph_def)
        except TypeError:
            raise TypeError('graph_def must be a GraphDef proto.')
    return graph_def

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        graph_def = _ProcessGraphDefParam(graph_def)

# Creates graph from saved GraphDef.
create_graph()

with tf.Session() as sess:

    output_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess,
                    sess.graph.as_graph_def(add_shapes=True),
                    [out_node],
                    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(model_out, "wb") as f:
        f.write(output_graph_def.SerializeToString())
